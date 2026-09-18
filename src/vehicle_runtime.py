"""
Vehicle Detector Runtime Abstraction (Fase C2/C3).
Supports configurable runtimes:
- 'ultralytics': Standard Ultralytics YOLO inference (default)
- 'ort_iobinding': ONNX Runtime with CUDA I/O Binding for zero-copy GPU inference
- 'trt': TensorRT Engine or TensorrtExecutionProvider
"""
import os
import sys
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

logger = logging.getLogger("alpr.vehicle_runtime")

class VehicleDetectorRunner:
    """Encapsulates vehicle detection runtime execution and performance profiling."""
    def __init__(
        self,
        model_name: str,
        runtime: str = "ultralytics",
        imgsz: int = 416,
        conf_threshold: float = 0.25,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
        model_factory: Optional[Any] = None
    ):
        self.model_name = model_name
        self.runtime = runtime.lower()
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.classes = classes or [0, 1, 2, 3, 5, 7]
        self.device = device
        self.names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.model = None
        self.ort_session = None
        self.io_binding = None
        self._init_runtime(model_factory=model_factory)

    def warmup(self):
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.predict(dummy)
        except Exception:
            pass

    def __call__(self, *args, **kwargs):
        if self.model is not None:
            return self.model(*args, **kwargs)
        img = args[0] if args else kwargs.get('source')
        return self.predict(img)

    def _init_runtime(self, model_factory: Optional[Any] = None):
        if model_factory is not None:
            self.model = model_factory(self.model_name)
            if hasattr(self.model, 'names') and self.model.names:
                self.names = self.model.names
            return
        if self.runtime == "ort_iobinding":
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                onnx_path = self.model_name if self.model_name.endswith('.onnx') else f"{self.model_name}.onnx"
                if os.path.exists(onnx_path):
                    opts = ort.SessionOptions()
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.ort_session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
                    if 'CUDAExecutionProvider' in self.ort_session.get_providers():
                        self.io_binding = self.ort_session.io_binding()
                        logger.info("✅ Initialized ONNX Runtime with CUDA I/O Binding.")
                    else:
                        logger.info("ℹ️ CUDAExecutionProvider not active; using standard CPU ORT.")
                    return
                else:
                    logger.warning(f"ONNX model file {onnx_path} not found. Falling back to Ultralytics.")
            except Exception as e:
                logger.warning(f"Error initializing ORT I/O Binding ({e}). Falling back to Ultralytics.")

        # Default fallback to Ultralytics YOLO
        from ultralytics import YOLO
        logger.info(f"Loading vehicle detector ({self.model_name}) via Ultralytics on {self.device}...")
        self.model = YOLO(self.model_name)
        if hasattr(self.model, 'names') and self.model.names:
            self.names = self.model.names

    def predict(
        self,
        crop_roi: np.ndarray,
        imgsz: Optional[int] = None,
        conf: Optional[float] = None,
        classes: Optional[List[int]] = None
    ) -> Tuple[List[List[float]], List[float], List[int], Dict[str, float]]:
        """
        Executes inference on image crop.
        Returns:
            boxes: [[x1, y1, x2, y2], ...]
            confidences: [0.95, ...]
            class_ids: [2, ...]
            timing: {'preprocess': sec, 'inference': sec, 'postprocess': sec, 'total': sec}
        """
        imgsz = imgsz or self.imgsz
        conf = conf or self.conf_threshold
        classes = classes or self.classes

        t_start = time.perf_counter()

        # Branch A: Ultralytics Runtime
        if self.model is not None:
            import torch
            with torch.inference_mode():
                res = self.model(
                    crop_roi,
                    imgsz=imgsz,
                    verbose=False,
                    conf=conf,
                    classes=classes,
                    device=self.device
                )[0]

            boxes, confs, cls_ids = [], [], []
            for box in res.boxes:
                rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
                boxes.append([float(rx1), float(ry1), float(rx2), float(ry2)])
                confs.append(float(box.conf[0].item()))
                cls_ids.append(int(box.cls[0].item()))

            speed = getattr(res, 'speed', {}) or {}
            prep_s = float(speed.get('preprocess', 0.0)) / 1000.0
            inf_s = float(speed.get('inference', 0.0)) / 1000.0
            post_s = float(speed.get('postprocess', 0.0)) / 1000.0
            total_s = time.perf_counter() - t_start

            return boxes, confs, cls_ids, {
                'preprocess': prep_s,
                'inference': inf_s,
                'postprocess': post_s,
                'total': total_s
            }

        # Branch B: ONNX Runtime I/O Binding
        if self.ort_session is not None:
            t0 = time.perf_counter()
            # Preprocessing (Resize & Normalization)
            h0, w0 = crop_roi.shape[:2]
            scale = min(imgsz / w0, imgsz / h0)
            nw, nh = int(w0 * scale), int(h0 * scale)
            resized = cv2.resize(crop_roi, (nw, nh), interpolation=cv2.INTER_LINEAR)
            pad_img = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
            pad_img[:nh, :nw] = resized
            blob = (pad_img.transpose(2, 0, 1).astype(np.float32) / 255.0)[None, ...]

            t_prep = time.perf_counter() - t0
            t1 = time.perf_counter()

            input_name = self.ort_session.get_inputs()[0].name
            if self.io_binding is not None:
                import torch
                # Zero-copy upload via PyTorch CUDA tensor
                t_cuda = torch.from_numpy(blob).cuda()
                self.io_binding.bind_input(
                    name=input_name,
                    device_type='cuda',
                    device_id=0,
                    element_type=np.float32,
                    shape=tuple(t_cuda.shape),
                    buffer_ptr=t_cuda.data_ptr()
                )
                output_name = self.ort_session.get_outputs()[0].name
                self.io_binding.bind_output(output_name, 'cuda')
                self.ort_session.run_with_iobinding(self.io_binding)
                outputs = [out.numpy() for out in self.io_binding.copy_outputs_to_cpu()]
            else:
                outputs = self.ort_session.run(None, {input_name: blob})

            t_inf = time.perf_counter() - t1
            t2 = time.perf_counter()

            # Post-processing (NMS / Output decoding)
            out = outputs[0]  # shape (1, 84, N) or (1, N, 84)
            if out.shape[1] < out.shape[2]:
                out = out.transpose(0, 2, 1)
            preds = out[0]  # shape (N, 84)

            boxes, confs, cls_ids = [], [], []
            boxes_raw = preds[:, :4]
            scores = preds[:, 4:]
            max_scores = np.max(scores, axis=1)
            max_cls = np.argmax(scores, axis=1)

            mask = max_scores >= conf
            if np.any(mask):
                filt_boxes = boxes_raw[mask]
                filt_scores = max_scores[mask]
                filt_cls = max_cls[mask]

                # Convert cx, cy, w, h to xyxy on original image
                for b, s, c in zip(filt_boxes, filt_scores, filt_cls):
                    if int(c) in classes:
                        cx, cy, w, h = b
                        x1 = max(0.0, (cx - w / 2.0) / scale)
                        y1 = max(0.0, (cy - h / 2.0) / scale)
                        x2 = min(float(w0), (cx + w / 2.0) / scale)
                        y2 = min(float(h0), (cy + h / 2.0) / scale)
                        boxes.append([x1, y1, x2, y2])
                        confs.append(float(s))
                        cls_ids.append(int(c))

            t_post = time.perf_counter() - t2
            total_s = time.perf_counter() - t_start

            return boxes, confs, cls_ids, {
                'preprocess': t_prep,
                'inference': t_inf,
                'postprocess': t_post,
                'total': total_s
            }

        # Fallback empty
        return [], [], [], {'preprocess': 0.0, 'inference': 0.0, 'postprocess': 0.0, 'total': 0.0}

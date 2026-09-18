"""
Debug script to inspect why Clip 61 events were not registered.
Runs ByteTrack and FSM on clip 61 frames 7500-8100 (Hino truck passage).
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import cv2
import yaml
import numpy as np
import supervision as sv
from ultralytics import YOLO
from src.crossing_logic import CrossingFSM

with open('config/camera_config.yaml') as f:
    cfg = yaml.safe_load(f)

crop_rect = cfg['roi']['crop_rect']
poly_gravel = np.array(cfg['roi']['polygon_gravel'], dtype=np.int32)
cx1, cy1 = crop_rect['x_min'], crop_rect['y_min']
cx2, cy2 = crop_rect['x_max'], crop_rect['y_max']

line_cfg = cfg['roi']['crossing_line']
fsm = CrossingFSM(
    line_p1=line_cfg['p1'],
    line_p2=line_cfg['p2'],
    dir_increasing_y=line_cfg.get('direction_increasing_y', 'ENTRADA'),
    dir_decreasing_y=line_cfg.get('direction_decreasing_y', 'SALIDA')
)

tracker = sv.ByteTrack(
    track_activation_threshold=0.20,
    lost_track_buffer=45,
    minimum_matching_threshold=0.65,
    frame_rate=8
)

model = YOLO('yolov8n.pt')
video_path = 'Camara Placas 2_20260909105651-20260909163038(61).mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

class MockTrack:
    def __init__(self, tid):
        self.track_id = tid
        self.trajectory = []
        self.state = 'OUTSIDE'
        self.last_side = None
        self.crossing_timestamp = None
        self.crossing_point = None
        self.direction = None
        self.has_emitted = False

tracks = {}
cap.set(cv2.CAP_PROP_POS_FRAMES, 7500)

for f_idx in range(7500, 8100, 3):  # sampling ~8 fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret:
        break
    ts = f_idx / fps
    crop_roi = frame[cy1:cy2, cx1:cx2]

    res = model(crop_roi, imgsz=416, verbose=False, conf=0.30, classes=[2, 3, 5, 7], device='cpu')[0]
    valid_boxes, valid_confs, valid_classes = [], [], []
    for box in res.boxes:
        rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        fx1, fy1, fx2, fy2 = int(rx1 + cx1), int(ry1 + cy1), int(rx2 + cx1), int(ry2 + cy1)
        pt = ((fx1 + fx2) // 2, fy2)
        if cv2.pointPolygonTest(poly_gravel, (float(pt[0]), float(pt[1])), False) >= 0:
            valid_boxes.append([rx1, ry1, rx2, ry2])
            valid_confs.append(conf)
            valid_classes.append(cls_id)

    if valid_boxes:
        detections = sv.Detections(
            xyxy=np.array(valid_boxes, dtype=np.float32),
            confidence=np.array(valid_confs, dtype=np.float32),
            class_id=np.array(valid_classes, dtype=np.int32)
        )
        tracked = tracker.update_with_detections(detections)
    else:
        tracked = tracker.update_with_detections(sv.Detections.empty())

    for i in range(len(tracked)):
        trk_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None and len(tracked.tracker_id) > i else None
        if trk_id is None:
            continue
        rx1, ry1, rx2, ry2 = tracked.xyxy[i]
        fx1, fy1, fx2, fy2 = int(rx1 + cx1), int(ry1 + cy1), int(rx2 + cx1), int(ry2 + cy1)
        contact_pt = ((fx1 + fx2) / 2.0, float(fy2 - 15))

        if trk_id not in tracks:
            tracks[trk_id] = MockTrack(trk_id)
        tr = tracks[trk_id]
        committed = fsm.update_track(tr, contact_pt, ts)
        if committed and not tr.has_emitted:
            tr.has_emitted = True
            print(f'*** COMMITTED EVENT: Track {trk_id} dir={tr.direction} ts={ts:.1f}s ***')

cap.release()
print('Tracks summary:')
for tid, t in tracks.items():
    min_y = min(p[1] for p in t.trajectory) if t.trajectory else 0
    max_y = max(p[1] for p in t.trajectory) if t.trajectory else 0
    print(f'Track {tid}: state={t.state}, points={len(t.trajectory)}, min_y={min_y:.1f}, max_y={max_y:.1f}, dir={t.direction}, emitted={t.has_emitted}')

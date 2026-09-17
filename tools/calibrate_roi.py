"""
Visual ROI & Line calibration preview tool.
Loads config/camera_config.yaml, overlays the Crop Rect, Polygon, and Virtual Line onto sample_frame.jpg,
and saves benchmarks/roi_calibration_preview.jpg for inspection.
"""
import os
import yaml
import cv2
import numpy as np

def generate_roi_preview(config_path="config/camera_config.yaml", sample_img_path="benchmarks/sample_frame.jpg", out_path="benchmarks/roi_calibration_preview.jpg"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    frame = cv2.imread(sample_img_path)
    if frame is None:
        raise FileNotFoundError(f"Sample image not found: {sample_img_path}")
        
    overlay = frame.copy()
    
    # 1. Draw Crop Rect (Cyan)
    crop = cfg['roi']['crop_rect']
    cv2.rectangle(overlay, (crop['x_min'], crop['y_min']), (crop['x_max'], crop['y_max']), (255, 255, 0), 3)
    cv2.putText(overlay, "CROP RECT (PHYSICAL ROI)", (crop['x_min'] + 20, crop['y_min'] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    # 2. Draw Polygon (Green fill with transparency + boundary)
    poly = np.array(cfg['roi']['polygon_gravel'], dtype=np.int32)
    cv2.fillPoly(overlay, [poly], (0, 200, 0))
    cv2.polylines(overlay, [poly], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.putText(overlay, "POLIGONO CARRIL LASTRE", (1400, 750), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Blend overlay with frame (opacity 0.3 for green polygon)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # 3. Draw Virtual Crossing Line (Red)
    line = cfg['roi']['crossing_line']
    p1 = tuple(line['p1'])
    p2 = tuple(line['p2'])
    cv2.line(frame, p1, p2, (0, 0, 255), 5)
    cv2.circle(frame, p1, 10, (0, 0, 255), -1)
    cv2.circle(frame, p2, 10, (0, 0, 255), -1)
    cv2.putText(frame, "LINEA DE CRUCE VIRTUAL (ENTRADA v / SALIDA ^)", (p1[0], p1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"Calibration preview saved to {out_path}")

if __name__ == '__main__':
    generate_roi_preview()

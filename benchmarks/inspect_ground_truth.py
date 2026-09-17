"""
Ground truth generator and visual verifier for benchmark clips (60) and (61).
Scans videos at 2 fps with YOLO to find all vehicle passages in the gravel road polygon.
Saves crops and timestamps to benchmarks/ground_truth_candidates/ for human audit.
"""
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

POLYGON = np.array([
    [1484, 636],
    [1750, 696],
    [1854, 858],
    [1913, 1066],
    [2076, 1302],
    [2372, 1554],
    [2579, 1664],
    [370, 1664],
    [340, 1125],
    [622, 918],
    [1036, 770],
    [1332, 681]
], dtype=np.int32)

VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

def inspect_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample every 12 frames (~2 FPS)
    sample_interval = int(round(fps / 2.0))
    
    frame_idx = 0
    detected_events = []
    
    print(f"Scanning {video_path} (total {total_frames} frames)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_interval == 0:
            timestamp = frame_idx / fps
            # Crop to ROI bounding box to speed up detection:
            # x: 300 to 2600, y: 600 to 1664
            roi = frame[600:1664, 300:2600]
            results = model(roi, verbose=False, conf=0.35, classes=list(VEHICLE_CLASSES.keys()))
            
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                # Bbox in ROI coordinates -> convert to full frame
                rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
                fx1, fy1, fx2, fy2 = int(rx1 + 300), int(ry1 + 600), int(rx2 + 300), int(ry2 + 600)
                
                # Bottom contact point of vehicle
                bottom_center = ((fx1 + fx2) // 2, fy2)
                
                # Exclude the static parked white van at bottom-left:
                # Parked van sits at x < 500, y > 1200
                if fx1 < 500 and fy2 > 1300 and (fx2 - fx1) > 300:
                    # Static parked van at camera base, ignore
                    continue
                    
                if point_in_polygon(bottom_center, POLYGON):
                    # Save candidate frame
                    event_info = {
                        'video': os.path.basename(video_path),
                        'frame_idx': frame_idx,
                        'timestamp_sec': round(timestamp, 2),
                        'class': VEHICLE_CLASSES[cls_id],
                        'confidence': round(conf, 3),
                        'bbox': [fx1, fy1, fx2, fy2],
                        'bottom_center': bottom_center
                    }
                    detected_events.append(event_info)
                    
                    # Save crop for inspection
                    crop = frame[max(0, fy1):min(frame.shape[0], fy2), max(0, fx1):min(frame.shape[1], fx2)]
                    if crop.size > 0:
                        crop_filename = f"{os.path.basename(video_path)}_f{frame_idx}_{timestamp:.1f}s_{VEHICLE_CLASSES[cls_id]}.jpg"
                        cv2.imwrite(os.path.join(output_dir, crop_filename), crop)
                        
        frame_idx += 1
        
    cap.release()
    print(f"Finished {video_path}: Found {len(detected_events)} candidate vehicle detections.")
    return detected_events

if __name__ == '__main__':
    for v in [
        'Camara Placas 2_20260909105651-20260909163038(60).mp4',
        'Camara Placas 2_20260909105651-20260909163038(61).mp4'
    ]:
        events = inspect_video(v, 'benchmarks/ground_truth_candidates')
        with open(f"benchmarks/{os.path.basename(v)}_candidates.json", "w") as f:
            json.dump(events, f, indent=2)

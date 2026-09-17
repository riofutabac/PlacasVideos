"""
Benchmark video decoding speed: OpenCV vs PyAV vs PyAV VideoToolbox.
Measures time to decode 500 frames of 3K video (2960x1664).
"""
import time
import cv2
import av

VIDEO_PATH = "Camara Placas 2_20260909105651-20260909163038(60).mp4"
NUM_FRAMES = 300

def test_opencv():
    cap = cv2.VideoCapture(VIDEO_PATH)
    t0 = time.perf_counter()
    count = 0
    for _ in range(NUM_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
    t1 = time.perf_counter()
    cap.release()
    fps = count / (t1 - t0)
    print(f"OpenCV VideoCapture:  {t1-t0:.2f}s for {count} frames ({fps:.1f} fps, {(t1-t0)/count*1000:.1f} ms/frame)")

def test_pyav_default():
    container = av.open(VIDEO_PATH)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    t0 = time.perf_counter()
    count = 0
    for frame in container.decode(stream):
        # Convert to numpy
        img = frame.to_ndarray(format='bgr24')
        count += 1
        if count >= NUM_FRAMES:
            break
    t1 = time.perf_counter()
    container.close()
    fps = count / (t1 - t0)
    print(f"PyAV Default Multi:   {t1-t0:.2f}s for {count} frames ({fps:.1f} fps, {(t1-t0)/count*1000:.1f} ms/frame)")

def test_pyav_videotoolbox():
    try:
        # Open with videotoolbox hardware acceleration
        container = av.open(VIDEO_PATH, options={'hwaccel': 'videotoolbox'})
        stream = container.streams.video[0]
        t0 = time.perf_counter()
        count = 0
        for frame in container.decode(stream):
            img = frame.to_ndarray(format='bgr24')
            count += 1
            if count >= NUM_FRAMES:
                break
        t1 = time.perf_counter()
        container.close()
        fps = count / (t1 - t0)
        print(f"PyAV VideoToolbox:    {t1-t0:.2f}s for {count} frames ({fps:.1f} fps, {(t1-t0)/count*1000:.1f} ms/frame)")
    except Exception as e:
        print(f"PyAV VideoToolbox error: {e}")

if __name__ == '__main__':
    print(f"Benchmarking video decoding on {NUM_FRAMES} frames of 3K video...")
    test_opencv()
    test_pyav_default()
    test_pyav_videotoolbox()

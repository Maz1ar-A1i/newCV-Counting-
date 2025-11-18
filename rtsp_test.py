import cv2, time
url = "rtsp://admin:dl2025123@192.168.2.112:554/cam/realmonitor?channel=1&subtype=0"

backends = [
    ("Default", 0),
    ("FFMPEG", cv2.CAP_FFMPEG),
    ("GSTREAMER", cv2.CAP_GSTREAMER if hasattr(cv2, 'CAP_GSTREAMER') else None)
]

for name, backend in backends:
    if backend is None:
        print(name, "backend not available, skipping")
        continue
    print("Testing backend:", name, backend)
    cap = cv2.VideoCapture(url, backend)
    print("isOpened:", cap.isOpened())
    if cap.isOpened():
        ret, frame = cap.read()
        print("read:", ret, "frame is None?", frame is None)
        if ret and frame is not None:
            print("frame shape:", frame.shape)
        cap.release()
    time.sleep(1)
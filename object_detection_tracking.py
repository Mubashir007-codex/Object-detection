"""
TASK 4: Object Detection and Tracking
--------------------------------------
- Real-time video input (webcam or video file) using OpenCV
- Object detection using a pre-trained YOLOv8 model
- Draws bounding boxes on detected objects
- Tracks objects across frames using Deep SORT
- Displays labels + tracking IDs in real time

Run this on your OWN computer (needs a webcam/display) -- see README.md
for setup instructions.
"""

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def main(video_source=0):
    """
    video_source:
        0            -> use your default webcam
        "video.mp4"  -> use a video file instead
    """

    # 1. Load a pre-trained YOLOv8 model (downloads automatically on first run)
    #    "yolov8n.pt" = nano version: smallest & fastest, good for real-time use
    model = YOLO("yolov8n.pt")

    # 2. Initialize the Deep SORT tracker
    #    max_age = how many frames a lost track is kept before being deleted
    tracker = DeepSort(max_age=30)

    # 3. Open the video source (webcam or file)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # 4. Run YOLO detection on the current frame
        results = model(frame, verbose=False)[0]

        # 5. Convert YOLO's output into the format Deep SORT expects:
        #    a list of ([x, y, w, h], confidence, class_name)
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, label))

        # 6. Update the tracker with this frame's detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # 7. Draw bounding boxes + tracking IDs on the frame
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            class_name = track.get_det_class() or "object"
            l, t, r, b = track.to_ltrb()  # left, top, right, bottom

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{class_name} ID:{track_id}",
                (int(l), int(t) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # 8. Display the output live
        cv2.imshow("Object Detection & Tracking", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Change video_source to a filename (e.g. "test.mp4") to use a video file
    # instead of your webcam.
    main(video_source=0)
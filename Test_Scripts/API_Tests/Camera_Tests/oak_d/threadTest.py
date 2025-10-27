import cv2
import time
from API.Camera.oakd_api import OAKD_LR
from pathlib import Path

MODEL_PATH = str((Path(__file__).parent / Path('../../../../Perception/Models/test_model/yolov8n_coco_640x352.blob')).resolve().absolute())

# Label Map for detection classes
LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

def main():
    # Initialize the camera API
    print("[INFO] Initializing OAKD LR camera...")
    oakd = OAKD_LR(MODEL_PATH, LABELS)
    
    # Start the camera
    oakd.startCapture()
    print("[INFO] Camera started successfully!")

    try:
        while True:
            # Get latest frames
            frames = oakd.getLatestBuffers()
            detections = oakd.getLatestDetection()

            if frames:
                rgb_frame, depth_frame = frames

                # Convert depth frame to color map for visualization
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)

                # Draw detections on RGB frame
                if detections:
                    for detection in detections:
                        x_min, y_min, x_max, y_max = (
                            int(detection.xmin * rgb_frame.shape[1]),
                            int(detection.ymin * rgb_frame.shape[0]),
                            int(detection.xmax * rgb_frame.shape[1]),
                            int(detection.ymax * rgb_frame.shape[0]),
                        )
                        label = LABELS[detection.label] if detection.label < len(LABELS) else "Unknown"
                        confidence = detection.confidence

                        # Draw bounding box
                        cv2.rectangle(rgb_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(
                            rgb_frame,
                            f"{label} ({confidence:.2f})",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                # Show images
                cv2.imshow("RGB Frame", rgb_frame)
                cv2.imshow("Depth Map", depth_colormap)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Exiting...")
                break

            time.sleep(0.01)  # Small delay to reduce CPU usage

    except KeyboardInterrupt:
        print("[INFO] Stopping capture due to keyboard interrupt...")

    # Cleanup
    oakd.stopCapture()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from API.Camera.oakd_api import OAKD_LR
from pathlib import Path
import cv2
import time
import numpy as np
# Path to the model blob
nnPath = str((Path(__file__).parent / Path('../../../../Perception/Models/test_model/yolov8n_coco_640x352.blob')).resolve().absolute())

# Label Map for detection classes
labelMap = [
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

# Create OAKD_LR object and start capturing frames
cam = OAKD_LR(model_path=nnPath, labelMap=labelMap)
cam.startCapture()


def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            # TODO: Investigate into the label index.
            # print(f"label index: {detection.label}")
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            # print(bbox)
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)  # I added -1 because the label is one whne I only have one detect object
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)
while(True):
    try:
        frame_rgb, _ = cam.getBuffers()  # Get RGB frame and depth frame
        if frame_rgb is None:
            print("No frame received")
            continue
        
        detections = cam.getDetection()  # Get object detections
        print(detections)
          
        displayFrame("camera",frame=frame_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
            break
    except Exception as e:
        print(f"Error: {e}")

# Stop capture and release resources
cam.stopCapture()
cv2.destroyAllWindows()

import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model_path = "/home/chaser/Downloads/best(1).pt"  # Change to your .pt file path
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path).to(device)

# Load the video file
video_path = "/media/chaser/Extreme SSD/RoboBoat data/02_h264.mp4"  # Change to your video file path
cap = cv2.VideoCapture(video_path)

def balance(frame, reference_Y_mean):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    current_Y_mean = ycrcb[:, :, 0].mean()

    if reference_Y_mean is not None and current_Y_mean > 0:
        gamma = reference_Y_mean / current_Y_mean
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], table)

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def find_refYmean(ref_y_values):
    reference_Y_mean = np.mean(ref_y_values)
    return reference_Y_mean

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

counter = 0 
imglist = []
num_sample = 0
Ymean = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when the video ends
    

    # balance frame color
    # Run YOLO model inference
    if (counter>100 and num_sample<50):
        height, width, _ = frame.shape
        top_region = frame[:height // 10, :, :]

        ycrcb = cv2.cvtColor(top_region, cv2.COLOR_BGR2YCrCb)
        Y_channel = ycrcb[:, :, 0]
        imglist.append(Y_channel.mean())
        num_sample +=1
    if(counter>150):
        Ymean =np.mean(imglist)
    
    balanced = balance(frame,Ymean) if Ymean >0 else balance(frame,244)
    results = model(frame)
    results_balanced = model(balanced)

    # Display the results (YOLOv8 has built-in visualization)
    for result in results:
        frame = result.plot()  # Draw bounding boxes and labels on frame

    for result in results_balanced:
        balanced = result.plot()

    # Show the frame with detections
    cv2.imshow("YOLOv8 Detection", frame)
    cv2.imshow("balanced",balanced)

    counter +=1
    # Quit with 'q' key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

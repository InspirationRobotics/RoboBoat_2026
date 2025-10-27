import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

# Load the trained YOLO model
model_path = Path("Perception/Models/sign_model/sign.pt")  # Change this to your model's path
print(f"Loading model from {model_path.resolve()}")
if(model_path.exists):   
    model = YOLO(str(model_path.resolve()))  # Loads the .pt model

    # Open the video file
    video_path = "/home/chaser/Downloads/target_h264.mp4"  # Change this to your video path
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the output video file
    output_path = Path("Perception/ML_Model_Core/modelOutputs/output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(str(output_path.resolve()), fourcc, fps, (frame_width, frame_height))

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit if video is done

        # Run YOLO detection on the frame
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                label = f"{model.names[cls]}: {conf:.2f}"  # Class label
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Show the video (optional)
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Object detection complete. Output saved to:", output_path)

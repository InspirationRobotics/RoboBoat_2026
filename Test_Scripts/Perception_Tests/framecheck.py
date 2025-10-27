import cv2
import time

cap = cv2.VideoCapture("/media/chaser/C2A5-B700/docking_trial_two.mp4")

# Get FPS and total frame count
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")


while cap.isOpened():
    ret, frame = cap.read()  # Corrected variable assignment
    
    if not ret:  # Stop if video ends or there's an issue
        break
    
    cv2.imshow("frame", frame)  # Display the frame

    time.sleep(0.01)
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the video capture
cv2.destroyAllWindows()  # Close all OpenCV windows

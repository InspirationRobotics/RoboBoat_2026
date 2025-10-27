import numpy as np
import cv2

def gamma_correction(image, gamma=1.2):
    """ Apply gamma correction to an image. """
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Video input
video_path = "/home/chaser/Downloads/RoboBoat data/02_h264.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# Video writer for output
out = cv2.VideoWriter("output_gamma_corrected.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends
    
    # Apply gamma correction
    corrected = gamma_correction(frame, gamma=0.8)  # Adjust gamma as needed

    # Write the processed frame to output video
    out.write(corrected)

    # Display the frames
    cv2.imshow("Original", frame)
    cv2.imshow("Gamma Corrected", corrected)

    # Press 'q' to exit early
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

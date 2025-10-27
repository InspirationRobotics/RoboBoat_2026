import cv2
import numpy as np

video_path = "/home/chaser/Downloads/RoboBoat data/02_h264.mp4"
cap = cv2.VideoCapture(video_path)

# Parameters
num_reference_frames = 10
ref_y_values = []
threshold = 500  # Minimum contour area threshold
paused = False  # Control playback
frame_count = 0

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# Create VideoWriter object
out = cv2.VideoWriter(f"{video_path}_balanced.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, frame_size)

def detect_buoy_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
           cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > threshold:
            lowest_point = max(largest_contour, key=lambda point: point[0][1])  # Find lowest y point
            x, y = lowest_point[0]
            cv2.drawContours(frame, [largest_contour], -1, (0, 0, 255), 2)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            return (x, y), frame
    return None, frame

def detect_buoy_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > threshold:
            lowest_point = max(largest_contour, key=lambda point: point[0][1])
            x, y = lowest_point[0]
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            return (x, y), frame
    return None, frame

def navigate_boat(frame):
    red_buoy, frame = detect_buoy_red(frame)
    green_buoy, frame = detect_buoy_green(frame)
    
    h, w, _ = frame.shape
    center_x = w // 2
    
    if red_buoy and green_buoy:
        gate_center_x = (red_buoy[0] + green_buoy[0]) // 2
        if gate_center_x < center_x - 40:
            command = "Turn Left"
        elif gate_center_x > center_x + 40:
            command = "Turn Right"
        else:
            command = "Move Forward"
    elif green_buoy:
        command = "Turn Left"
    elif red_buoy:
        command = "Turn Right"
    else:
        command = "Searching for buoys"

    cv2.imshow("Buoy Detection", frame)
    return command, frame

def balance(frame, reference_Y_mean):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    current_Y_mean = ycrcb[:, :, 0].mean()
    print(reference_Y_mean)

    if reference_Y_mean is not None and current_Y_mean > 0:
        gamma = reference_Y_mean / current_Y_mean
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], table)

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def find_refYmean(ref_y_values):
    reference_Y_mean = np.mean(ref_y_values)
    return reference_Y_mean

    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break  # Stop if video ends

    height, width, _ = frame.shape
    top_region = frame[:height // 10, :, :]

    ycrcb = cv2.cvtColor(top_region, cv2.COLOR_BGR2YCrCb)
    Y_channel = ycrcb[:, :, 0]

    if frame_count < num_reference_frames:
        ref_y_values.append(Y_channel.mean())
        frame_count += 1
        if frame_count == num_reference_frames:
            reference_Y_mean = np.mean(ref_y_values)
            print(f"Reference Y Mean: {reference_Y_mean}")
    else:
        balanced = balance(frame, reference_Y_mean)
        command, DetectFrame = navigate_boat(balanced)
        out.write(balanced)  # Save frame

        cv2.imshow("Original", frame)
        cv2.imshow("Balanced", balanced)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused

cap.release()
out.release()
cv2.destroyAllWindows()

from threading import Thread, Lock
import numpy as np
import cv2
import time
from API.Camera.oakd_api import OAKD_LR

class CameraCore:
    def __init__(self, model_path: str, labelMap: list):
        self.cam = OAKD_LR(model_path=model_path, labelMap=labelMap)
        self.cam_lock = Lock()
        self.labelMap = labelMap
        
        # Shared resources (protected by lock)
        self.rgb_frame = None
        self.depth_frame = None
        self.detections = []
        
        self.running = False
        self.capture_thread = None
    
    def start(self):
        """Start camera streaming in a separate thread."""
        if self.running:
            print("Camera is already running.")
            return
        
        self.running = True
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("Camera capture started.")
    
    def stop(self):
        """Stop camera streaming."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
            self.capture_thread = None
        print("Camera capture stopped.")
    
    def _capture_loop(self):
        """Continuously fetch frames and detections in the background."""
        self.cam.startCapture()
        while self.running:
            with self.cam_lock:
                self.rgb_frame, self.depth_frame = self.cam.getBuffers()
                self.detections = self.cam.getDetection()
            time.sleep(1 / self.cam.FPS)  # Sleep to match frame rate
    
    def get_latest_frames(self):
        """Retrieve the latest RGB and depth frames."""
        with self.cam_lock:
            return self.rgb_frame, self.depth_frame
    
    def get_latest_detections(self):
        """Retrieve the latest object detections."""
        with self.cam_lock:
            return self.detections
    
    def visualize(self):
        """Return a labeled OpenCV frame with bounding boxes and labels."""
        rgb, _ = self.get_latest_frames()
        if rgb is None:
            return None
        
        color = (255, 0, 0)
        try:
            for detection in self.get_latest_detections():
                bbox = self._frameNorm(rgb, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(rgb, self.labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), 
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(rgb, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        except Exception as e:
            print(f"Visualization Error: {e}")
        
        return rgb
    
    def _frameNorm(self, frame, bbox):
        """Normalize bounding box coordinates to match frame size."""
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesis, ObjectHypothesisWithPose, BoundingBox2D, Point2D, Pose2D

import cv2
import numpy as np
import depthai as dai
import threading
import time

class OAKD_LR(Node):
    def __init__(self, model_path: str, labelMap: list):
        """
        Arguments: 
            model_path  :str  -> path to the blob file
            labelMap    :list -> A list of classes, can be found from json file in Model folder

        """
        super().__init__('OAKD_LR')
        # config for stereo camera
        self.FPS = 20
        self.extended_disparity = True
        self.subpixel = True
        self.lr_check = True

        # config for NN detection
        self.syncNN = True
        self.nnPath = model_path
        self.labelMap = labelMap
        self.confidenceThreshold = 0.5

        if(self._findCamera()):
            self.device = dai.Device()
        else:
            print("ERROR: DID NOT FOUND OAK_D CAMERA")
            self.device = None

        # image config
        self.COLOR_RESOLUTION = dai.ColorCameraProperties.SensorResolution.THE_1200_P
        self.imageWidth = 1920
        self.imageHeight = 1200

        # Threading components
        self.running = False
        self.lock = threading.Lock()

        self.capture_thread = None

        # Publisher for RGB. depth image and bbox
        self.logger = self.get_logger()
        self.bridge = CvBridge()
        self.rgb_pub = self.create_publisher(Image,'oakd/rgb', 10)
        self.depth_pub = self.create_publisher(Image,'oakd/depth', 10)
        self.bbox_pub = self.create_publisher(Detection2DArray,'/oak/bbox',10)

    def _initPipeline(self):
        self.pipeline = dai.Pipeline()
        # 3 cameras
        self.leftCam = self.pipeline.create(dai.node.ColorCamera)
        self.rightCam = self.pipeline.create(dai.node.ColorCamera)
        self.centerCam = self.pipeline.create(dai.node.ColorCamera)

        # depth map
        self.stereo = self.pipeline.create(dai.node.StereoDepth)

        # Neural network
        self.detection = self.pipeline.create(dai.node.YoloDetectionNetwork)
        self.manip = self.pipeline.create(dai.node.ImageManip)

        # Output node
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        self.xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        self.xoutYolo = self.pipeline.create(dai.node.XLinkOut)

        # set stream name
        self.xoutRgb.setStreamName("rgb")
        self.xoutDepth.setStreamName("depth")
        self.xoutYolo.setStreamName("yolo")

    def _setProperties(self):
        self.leftCam.setIspScale(2, 3)
        self.leftCam.setPreviewSize(640, 352) # the size should be 640,400 for future models
        self.leftCam.setCamera("left")
        self.leftCam.setResolution(self.COLOR_RESOLUTION)
        self.leftCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.leftCam.setFps(self.FPS)

        self.rightCam.setIspScale(2, 3)
        self.rightCam.setPreviewSize(640, 352)
        self.rightCam.setCamera("right")
        self.rightCam.setResolution(self.COLOR_RESOLUTION)
        self.rightCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.rightCam.setFps(self.FPS)

        self.centerCam.setIspScale(2, 3)
        self.centerCam.setPreviewSize(640, 352)
        self.centerCam.setCamera("center")
        self.centerCam.setResolution(self.COLOR_RESOLUTION)
        self.centerCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.centerCam.setFps(self.FPS)

        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
        self.stereo.setLeftRightCheck(self.lr_check)
        self.stereo.setExtendedDisparity(self.extended_disparity)
        self.stereo.setSubpixel(self.subpixel)

        self.detection.setConfidenceThreshold(self.confidenceThreshold)
        self.detection.setNumClasses(len(self.labelMap))
        self.detection.setCoordinateSize(4)
        self.detection.setIouThreshold(0.5)
        self.detection.setBlobPath(self.nnPath)
        self.detection.setNumInferenceThreads(2)
        self.detection.input.setBlocking(False)

        self.manip.initialConfig.setResize(640, 352)
        self.manip.initialConfig.setCropRect(0, 0, 640, 352)
        self.manip.setFrameType(dai.ImgFrame.Type.BGR888p)

    def _linkStereo(self):
        self.leftCam.isp.link(self.stereo.left)
        self.rightCam.isp.link(self.stereo.right)
        self.stereo.depth.link(self.xoutDepth.input)

    def _linkNN(self):
        self.centerCam.preview.link(self.manip.inputImage)
        self.manip.out.link(self.detection.input)
        
        if self.syncNN:
            self.detection.passthrough.link(self.xoutRgb.input)
        else:
            self.leftCam.preview.link(self.xoutRgb.input)

        self.detection.out.link(self.xoutYolo.input)

    def _initQueues(self):
        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.qDet = self.device.getOutputQueue(name="yolo", maxSize=4, blocking=False)
        self.qDepth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    def _captureLoop(self):
        """ Threaded function to continuously grab frames and put them in queue"""
        while self.running:
            inRgb = self.qRgb.get().getCvFrame()
            inDepth = self.qDepth.get().getCvFrame()
            inDet = self.qDet.get()

            # pubish RGB image frame
            RGBmsg = self.bridge.cv2_to_imgmsg(inRgb, encoding='bgr8')
            RGBmsg.header.stamp = self.get_clock().now().to_msg()
            RGBmsg.header.frame_id = "camera_RGB_frame"
            self.rgb_pub.publish(RGBmsg)

            # publish Depth image frame
            Depthmsg = self.bridge.cv2_to_imgmsg(inDepth, encoding="16UC1")  # or "32FC1" --> convert to meters, oakd-LR is in mm
            Depthmsg.header.stamp = self.get_clock().now().to_msg()
            Depthmsg.header.frame_id = "camera_depth_frame"
            self.depth_pub.publish(Depthmsg)

            # publich yolo detection
            detection_array = Detection2DArray()
            detection_array.header.stamp = self.get_clock().now().to_msg()
            for detection in inDet.detections:
                # Calculate bounding box coordinates
                bbox = self._frame_norm(inRgb, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                
                # Calculate the center of the bounding box
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                
                # Define a smaller bounding box for depth calculation
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                bbox_msg = BoundingBox2D()
                bbox_msg.center = Pose2D(position=Point2D(x=center_x, y=center_y), theta=0.0)
                bbox_msg.size_x = width
                bbox_msg.size_y = height

                hypothesis = ObjectHypothesis()
                hypothesis.class_id = str(self.labelMap[detection.label])
                hypothesis.score = detection.confidence

                detection_msg = Detection2D()
                detection_msg.header = detection_array.header
                detection_msg.bbox = bbox_msg
                detection_msg.results.append(ObjectHypothesisWithPose(hypothesis=hypothesis))
                detection_array.detections.append(detection_msg)

            self.bbox_pub.publish(detection_array)
            
            time.sleep(1 / self.FPS)  # Sleep to match frame rate
    
    def _findCamera(self) -> bool:
        """ Check if a DepthAI device exists """
        try:
            # Get available devices
            available_devices = dai.Device.getAllConnectedDevices()

            # Check if any device is found
            if len(available_devices) == 0:
                print("[ERROR] No DepthAI devices found.")
                return False
            
            # If a device is found, print the device info and return True
            print(f"Found DepthAI device: {available_devices[0].getMxId()}")
            return True

        except RuntimeError as e:
            # Handle exceptions (e.g., if the device cannot be found)
            print(f"[ERROR] Failed to find DepthAI camera: {e}")
            return False

    def _frame_norm(self, frame, bbox):
        """Convert bbox from percentage to pixels base on frame size"""
        try:
            norm_vals = np.full(len(bbox), frame.shape[0])
            norm_vals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)
        except Exception as e:
            print("[ERROR] Normoalize bbox Error: {e}")

    def startCapture(self):
        if not self.device:
            print("[ERROR] Device is not running!")
            return
        else:
            print("[DEBUG] Device running.")

        print("[DEBUG] Starting pipeline...")
        self._initPipeline()
        self._setProperties()
        self._linkNN()
        self._linkStereo()
        self.device.startPipeline(self.pipeline)
        print("[DEBUG] Pipeline initialized.")
        self._initQueues()


        # Start thread
        self.running = True
        self.capture_thread = threading.Thread(target=self._captureLoop, daemon=True)
        self.capture_thread.start()
        time.sleep(1)  # wait for frame to arrive queue

    def stopCapture(self):
        self.logger.info("Shutting down OAKD_LR node...")
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.device:
            self.device.close()

        # destroy ros2 publishers
        self.rgb_pub.destroy()
        self.depth_pub.destroy()
        self.bbox_pub.destroy()

        self.logger.info("OAKD_LR node shutdown complete.")

if __name__=="__main__":
    from API.Util.get_labelMap import load_label_map

    rclpy.init()
    node = OAKD_LR(model_path="API/Camera/Models/competition_model/competition_openvino_2022.1_6shave.blob", labelMap=load_label_map("API/Camera/Models/competition_model/competition.json"))
    rclpy.spin(node)
    node.stopCapture()
    node.destroy_node()
    rclpy.shutdown()
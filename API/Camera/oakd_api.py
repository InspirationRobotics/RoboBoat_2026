import cv2
import numpy as np
import depthai as dai
import threading
import queue
import time
"""
Note from creator:
I think I overcomplicates things
self.frame_queue for frames may not be necessary
"""
class OAKD_LR:
    def __init__(self, model_path: str, labelMap: list):
        """
        Arguments: 
            model_path  :str  -> path to the blob file
            labelMap    :list -> A list of classes, can be found from json file in Model folder

        """
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
        self.frame_queue = queue.Queue(maxsize=4)
        self.det_queue = queue.Queue(maxsize=4)
        self.lock = threading.Lock()

        self.capture_thread = None

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
            inRgb = self.qRgb.get()
            inDepth = self.qDepth.get()
            inDet = self.qDet.get()

            if inRgb and inDepth:
                frame = (inRgb.getCvFrame(), inDepth.getCvFrame())

                # Store the latest frame safely
                with self.lock:
                    if self.frame_queue.full():
                        self.frame_queue.get()
                    self.frame_queue.put(frame)

            if inDet:
                with self.lock:
                    if self.det_queue.full():
                        self.det_queue.get()
                    self.det_queue.put(inDet.detections)

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
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.device:
            self.device.close()

    def getLatestBuffers(self):
        with self.lock:
            if not self.frame_queue.empty():
                return self.frame_queue.queue[-1]  # Get latest frame
            else:
                print("[ERROR] Queue empty")
        return None

    def getLatestDetection(self):
        with self.lock:
            if not self.det_queue.empty():
                return self.det_queue.queue[-1]  # Get latest detection
            else:
                print("[ERROR] Queue empty")
        return None

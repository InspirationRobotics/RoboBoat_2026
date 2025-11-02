import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import time
import cv2
import depthai as dai
from ultralytics import YOLO
from std_msgs.msg import Float32MultiArray
from src.read_yaml import extract_configuration
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesis, ObjectHypothesisWithPose, BoundingBox2D, Point2D, Pose2D

class DaiNode(Node):
    def __init__(self):
        super().__init__('dai_node')

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        config_folder = config_file['general']['config_folder']
        self.img_size = [config_file['camera']['image_size']['height'], config_file['camera']['image_size']['width']]

        # Load YOLOv8 model
        model_path = config_file['yolo']['cuda_path']
        self.model = YOLO(model_path)
        self.model.to('cuda')
        print("Inference device:", self.model.device)

        # Publisher for annotated image
        self.image_pub = self.create_publisher(Image,'dai_node/annotated_image', 10)


        self.bbox_publisher = self.create_publisher(
            Detection2DArray,
            '/oak/rgb/bounding_boxes',
            10
        )

        # Bridge for CV <-> ROS2 image conversion
        self.bridge = CvBridge()

        # Setup DepthAI pipeline
        self.pipeline = dai.Pipeline()
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setFps(30)

        xout_video = self.pipeline.createXLinkOut()
        xout_video.setStreamName("video")
        cam_rgb.video.link(xout_video.input)

        self.device = dai.Device(self.pipeline)
        self.video_queue = self.device.getOutputQueue(name="video", maxSize=30, blocking=True)
        self.device.setTimesync(True)

        # Create a timer to repeatedly call the inference function
        self.timer = self.create_timer(0.03, self.process_frame)  # ~30 FPS

    def process_frame(self):
        in_video = self.video_queue.get()
        while self.video_queue.has():
            in_video = self.video_queue.get()
        frame = in_video.getCvFrame()
        frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]))

        latency_ms = (dai.Clock.now() - in_video.getTimestamp()).total_seconds() * 1000
        
        # YOLOv8 inference
        inf_start_time = time.time()
        results = self.model.predict(frame, verbose=False, device="cuda")[0]
        inf_end_time = time.time()

        inference_time_ms = (inf_end_time - inf_start_time) * 1000
        self.get_logger().info(f"Camera to ROS latency: {latency_ms:.2f} ms, Inference time: {inference_time_ms:.2f} ms, Total: {latency_ms + inference_time_ms:.2f} ms")

        # Class name mapping from the model
        allowed_names = ['red_buoy', 'green_buoy', 'yellow_buoy', 'black_buoy']
        target_classes = [i for i, name in self.model.names.items() if name in allowed_names]

        # Filter detections
        filtered_indices = [i for i, cls in enumerate(results.boxes.cls.tolist()) if int(cls) in target_classes]
        results.boxes = results.boxes[filtered_indices]

        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
            
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist()) 
            conf = float(box.conf[0])
            class_id = int(box.cls[0])  

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1

            bbox = BoundingBox2D()
            bbox.center = Pose2D(position=Point2D(x=center_x, y=center_y), theta=0.0)
            bbox.size_x = width
            bbox.size_y = height

            hypothesis = ObjectHypothesis()
            hypothesis.class_id = str(class_id)
            hypothesis.score = conf

            detection = Detection2D()
            detection.header = detection_array.header
            detection.bbox = bbox
            detection.results.append(ObjectHypothesisWithPose(hypothesis=hypothesis))
            detection_array.detections.append(detection)

        # Publish bounding boxes
        self.bbox_publisher.publish(detection_array)

        # Annotate and publish
        annotated_frame = results.plot()
        image_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.image_pub.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DaiNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

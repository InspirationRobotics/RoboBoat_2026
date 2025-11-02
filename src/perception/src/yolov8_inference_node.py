import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesis, ObjectHypothesisWithPose, BoundingBox2D, Point2D, Pose2D
from src.read_yaml import extract_configuration

class YoloV8InferenceNode(Node):
    def __init__(self):
        super().__init__('yolov8_inference_node')
        
        # Subscribe to raw RGB camera frames
        self.subscription = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',
            self.image_callback,
            10
        )

        # Publisher for annotated frames
        self.publisher = self.create_publisher(
            Image,
            '/oak/rgb/image_annotated',
            10
        )

        # Publisher for Detection2DArray
        self.bbox_publisher = self.create_publisher(
            Detection2DArray,
            '/oak/rgb/bounding_boxes',
            10
        )

        # Convert between ROS Image messages and OpenCV
        self.bridge = CvBridge()

        # Load YOLOv8 model
        self.get_logger().info('Loading YOLOv8n model...')
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        # Load YOLOv8 model
        model_path = config_file['yolo']['cuda_path']       
        self.model = YOLO(model_path) 
        self.model.to('cuda')

    def image_callback(self, msg):
        try:
            # Convert ROS image (rgb8) to OpenCV BGR format
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            # Run inference
            results = self.model(bgr_image)[0]

            # Draw predictions on the frame
            annotated_frame = results.plot()

            # Create Detection2DArray message
            detection_array = Detection2DArray()
            detection_array.header = msg.header

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
                detection.header = msg.header
                detection.bbox = bbox
                detection.results.append(ObjectHypothesisWithPose(hypothesis=hypothesis))

                detection_array.detections.append(detection)

            self.bbox_publisher.publish(detection_array)

            # Convert back to ROS image and publish
            output_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            output_msg.header = msg.header
            self.publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloV8InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

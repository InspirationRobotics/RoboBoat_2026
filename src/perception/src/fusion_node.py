import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
# from sensor_msgs.msg import PointCloud
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import cv2
import depthai as dai
from ultralytics import YOLO
from std_msgs.msg import Float32MultiArray

from ament_index_python import get_package_share_directory
import os
import json
from pathlib import Path
import blobconverter
from src.read_yaml import extract_configuration, load_extrinsic_matrix, load_camera_calibration

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = config_file['general']['camera_extrinsic_calibration']
        extrinsic_yaml = os.path.join(config_folder, extrinsic_yaml)
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)
        self.T_cam_to_lidar = np.linalg.inv(self.T_lidar_to_cam.T)

        camera_yaml = config_file['general']['camera_intrinsic_calibration']
        camera_yaml = os.path.join(config_folder, camera_yaml)
        self.K, self.dist_coeffs = load_camera_calibration(camera_yaml)

        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.K))
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))

        self.img_topic_name = config_file['camera']['image_topic']

        self.show_lidar_projections = config_file['fusion']['show_lidar_projections']
        self.lidar_projections_size = config_file['fusion']['lidar_projections_size']
        self.use_oak_pipeline = config_file['fusion']['use_oak_pipeline']
        self.show_fusion_result_opencv = config_file['fusion']['show_fusion_result_opencv']
        self.yolo_on_cam = config_file['fusion']['yolo_on_cam']
        self.record_video = config_file['fusion']['record_video']
        self.video_file_name = config_file['fusion']['video_file_name']

        if self.yolo_on_cam:
            self.yolo_config = config_file['yolo']['dai_json_path']
            self.yolo_model = config_file['yolo']['dai_blob_path']
        else:
            yolo_model_path = config_file['yolo']['cuda_path']
            self.yolo_model = YOLO(yolo_model_path)
            try:
                self.yolo_model.to(device='cuda')
            except:
                print('cuda not avaliable, use cpu')
                self.yolo_model.to(device='cpu')

        # oak-d LR #################################################################
        self.img_size = [config_file['camera']['image_size']['height'], config_file['camera']['image_size']['width']]
        if self.use_oak_pipeline:
            pipeline = self.get_oak_pipeline_with_nn() if self.yolo_on_cam else self.get_oak_pipeline()
            self.device = dai.Device(pipeline)
        else:
            self.oak_d_cam_subs = self.create_subscription(
                Image,
                self.img_topic_name,
                self.cam_subs_callback,
                qos_profile_sensor_data
                )
        # ###############################################################################################################################################
        
        if self.record_video:
            print("recording video")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.recording_out = cv2.VideoWriter(self.video_file_name, fourcc, 20.0, (self.img_size[1],  self.img_size[0]))

        self.lidar_subs_ = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.lidar_subs_callback,
            qos_profile_sensor_data
        )
        self.lidar_subs_  # prevent unused variable warning
        self.depth_matrix = np.array([])
        self.frame = None
    
        self.fusion_img_pubs_ = self.create_publisher(Image, 'camera/fused_img', 10)
        self.closest_rgbouys_pubs = self.create_publisher(Float32MultiArray, '/centroids', 10)
        self.bridge = CvBridge()
        
    def cam_subs_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # self.img_size = self.frame.shape[0:2]
            # print("setted img_size: ", self.img_size)
        except CvBridgeError as e:
            print(e)

    def lidar_subs_callback(self, msg):
        # Get camera frame ########################################################################
        frame = None
        if self.use_oak_pipeline:  
            oak_q = self.device.getOutputQueue(name='rgb', maxSize=1, blocking=False)
            in_q = oak_q.tryGet()
            if self.yolo_on_cam:
                q_nn = self.device.getOutputQueue(name='nn', maxSize=4, blocking=False)
                in_nn = q_nn.get()
            if in_q is not None:
                frame = in_q.getCvFrame()
                frame = cv2.resize(frame, (768, 480))
                # print("got frame")
        else: # use ROS subscriber for oak cameras
            if self.frame is not None:
                frame = self.frame.copy()

        # Compute Depth Matrix #########################################################################
        lidar_points = np.array(list(read_points(msg, skip_nans=True))).T  # 4xn matrix, (x,y,z,i)
        xs, ys, ps = lidar2pixel(lidar_points, self.T_lidar_to_cam, self.K)

        filtered_x, filtered_y, filtered_p = filter_points(xs, ys, ps, self.img_size)

        self.depth_matrix = points_to_img(filtered_x, filtered_y, filtered_p, self.img_size)
        centroid_array = Float32MultiArray()
        centroid_data = []        
        closest_red = 40.0
        closest_green = 40.0
        xydepth = Float32MultiArray()
        xydepth = []        
        closest_red_xydepth = [-1.0, -1.0, -1.0]
        closest_green_xydepth = [-1.0, -1.0, -1.0]

        # Visualization ####################################################################################
        if frame is not None:
            if self.yolo_on_cam:
                if in_nn is not None:
                    detections = in_nn.detections
                    for detection in detections:
                        [x1,y1,x2,y2] = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                        cv2.putText(frame, self.labels[detection.label], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        object_depth = np.min(self.depth_matrix[y1:y2, x1:x2])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                        cv2.putText(frame, str(round(object_depth, 2)) + "m", (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
            else:
                detections_xyxyn, classes = self.yolo_predict(frame)
                for i, detection in enumerate(detections_xyxyn):
                    cls = None
                    if classes[i] == 16:
                        cls = 'Red'
                    elif classes[i] == 11:
                        cls = 'Green'
                    
                    x1 = int(detection[0] * self.img_size[1])
                    y1 = int(detection[1] * self.img_size[0])
                    x2 = int(detection[2] * self.img_size[1])
                    y2 = int(detection[3] * self.img_size[0])
                    object_depth = np.min(self.depth_matrix[y1:y2, x1:x2])
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                    cv2.putText(frame, str(round(object_depth, 2)) + "m", (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
                    if cls == 'Green':
                        closest_green = min(closest_green, object_depth)
                        if closest_green_xydepth[2] < 0 or closest_green_xydepth[2] > object_depth:
                            closest_green_xydepth = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, object_depth]
                    elif cls == 'Red':
                        closest_red = min(closest_red, object_depth)
                        if closest_red_xydepth[2] < 0 or closest_green_xydepth[2] > object_depth:
                            closest_red_xydepth = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, object_depth]
                            
                    # Return all detections
                    # xydepth.extend(p=[(x1 + x2) / 2.0, (y1 + y2) / 2.0, object_depth, cls])
                    
            
            # Draw circles for the lidar points
            max_dist_thresh = 10  # the max distance used for color coding in visualization window.
            if self.show_lidar_projections:
                for i in range(len(filtered_p)):
                    color_intensity = int((filtered_p[i] / max_dist_thresh * 255).clip(0, 255))
                    cv2.circle(frame, (filtered_x[i], filtered_y[i]), 1, (0,color_intensity, 255 - color_intensity), -1)

            if self.show_fusion_result_opencv:
                cv2.imshow('YOLO detection: ', frame)
                cv2.waitKey(1)

            if self.record_video:
                self.recording_out.write(frame)
                cv2.waitKey(1)

            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.fusion_img_pubs_.publish(img_msg)
            # get red & green closest bouys
            centroid_data.extend([closest_red, closest_green])
            # centroid_array.data = centroid_data

            # x, y, depth for closest bouys
            # print(closest_red_xydepth)
            # print(closest_green_xydepth)
            xydepth.extend(closest_red_xydepth)
            xydepth.extend(closest_green_xydepth)
            centroid_array.data = xydepth


            
            self.closest_rgbouys_pubs.publish(centroid_array)   


    def get_oak_pipeline(self):
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        # cam_rgb.setPreviewSize(1448, 568)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        cam_rgb.setInterleaved(False)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.video.link(xout_rgb.input)

        print("Set pipeline for OAK camera")

        return pipeline
    
    def get_oak_pipeline_with_nn(self):
        # parse config
        configPath = Path(self.yolo_config)
        if not configPath.exists():
            raise ValueError("Path {} does not exist!".format(configPath))

        with configPath.open() as f:
            config = json.load(f)
        nnConfig = config.get("nn_config", {})

        # parse input shape
        if "input_size" in nnConfig:
            W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

        # extract metadata
        metadata = nnConfig.get("NN_specific_metadata", {})
        classes = metadata.get("classes", {})
        coordinates = metadata.get("coordinates", {})
        anchors = metadata.get("anchors", {})
        anchorMasks = metadata.get("anchor_masks", {})
        iouThreshold = metadata.get("iou_threshold", {})
        confidenceThreshold = metadata.get("confidence_threshold", {})

        print(metadata)

        # parse labels
        nnMappings = config.get("mappings", {})
        self.labels = nnMappings.get("labels", {})

        # get model path
        nnPath = self.yolo_model
        if not Path(nnPath).exists():
            print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
            nnPath = str(blobconverter.from_zoo(self.yolo_model, shaves = 6, zoo_type = "depthai", use_cache=True))
        # sync outputs
        syncNN = True

        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        nnOut = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        nnOut.setStreamName("nn")

        # Properties
        camRgb.setPreviewSize(W, H)
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(40)

        # Network specific settings
        detectionNetwork.setConfidenceThreshold(confidenceThreshold)
        detectionNetwork.setNumClasses(classes)
        detectionNetwork.setCoordinateSize(coordinates)
        detectionNetwork.setAnchors(anchors)
        detectionNetwork.setAnchorMasks(anchorMasks)
        detectionNetwork.setIouThreshold(iouThreshold)
        detectionNetwork.setBlobPath(nnPath)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)

        # Linking
        camRgb.preview.link(detectionNetwork.input)
        # camRgb.video.link(xoutRgb.input)
        detectionNetwork.passthrough.link(xoutRgb.input)
        detectionNetwork.out.link(nnOut.input)

        print("Set pipeline for OAK camera with nn")
        nnConfig_size = [H, W]
        if self.img_size != nnConfig_size:
            print(f"Image size {self.img_size} but using config with {nnConfig_size}")
        self.img_size = nnConfig_size

        return pipeline
    
    def yolo_predict(self, img):
        results = self.yolo_model.predict(source=img, save=False, save_txt=False)

        # bounding box params https://docs.ultralytics.com/modes/predict/#boxes
        box = results[0].boxes.cpu()

        xyxyn = box.xyxyn.numpy().reshape((-1,))
        detections_xyxyn = list(zip(*[iter(xyxyn)] * 4))
        classes = box.cls.numpy().reshape((-1,))
        # confidence = box.conf.numpy()

        return detections_xyxyn, classes


# Helper functions #########################################################################################
def lidar2pixel(lidar_points, lidar2cam, K):
    # lidar_points assumed to be 4xn np array, each col should be (x, y, z, i)
    # R and T are (3, 3) and (3, )-like np arrays
    # K is (3, 3) camera calibration matrix
    xyz_lidar = lidar_points[0:3, :]
    # i = lidar_points[3, :]
    i = np.linalg.norm(xyz_lidar, axis=0)
    xyz_lidar = np.vstack((xyz_lidar, np.ones((len(i), ))))

    xyz_cam = lidar2cam @ xyz_lidar
    z_cam = xyz_cam[2, :]

    # Filter out the points on the back of the camera (points with negative z_cam values)
    index = z_cam >= 0
    xyz_cam = xyz_cam[:, index]
    z_cam = xyz_cam[2, :]
    i = i[index]

    xy_pixel = (K @ (xyz_cam[0:3, :] / z_cam)).astype(int)

    return xy_pixel[0, :], xy_pixel[1, :], i  # return x_array, y_array, intensity_array

def filter_points(xs, ys, ps, img_size):
    # Filter out the points that are not captured inside the camera
    lb = np.array([0, 0])
    ub = np.array(img_size).reshape((-1, ))
    points = np.stack((ys, xs), axis=1)

    inidx = np.all(np.logical_and(lb <= points, points < ub), axis=1)
    # print("xs shape: ", xs.shape)
    filtered_xy = points[inidx]
    filtered_p = ps[inidx]
    # print("filtered p shape: ", filtered_p.shape)
    # print("filtered x shape: ", filtered_xy[:, 1].shape)
    # print("filtered y shape: ", filtered_xy[:, 0].shape)

    return filtered_xy[:, 1], filtered_xy[:, 0], filtered_p  # return filtered_x, filtered_y, filtered_p

def points_to_img(xs, ys, ps, size):
    """
    Save the points (ps) into a matrix at their given locations,
        locations where there's no given points will be saved as np.inf
    :param xs: list of x (col) locations for the points
    :param ys: list of y (row) locations for the points
    :param ps: values for the points
    :param size: (row, col) size of the matrix
    :return : result np matrix
    """
    coords = np.stack((ys, xs))
    abs_coords = np.ravel_multi_index(coords, size)
    img = np.bincount(abs_coords, weights=ps, minlength=size[0]*size[1])
    img = img.reshape(size)
    img[img == 0.0] = np.inf
    return img

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

## The code below is "ported" from 
# https://github.com/ros/common_msgs/tree/noetic-devel/sensor_msgs/src/sensor_msgs
import sys
from collections import namedtuple
import ctypes
import math
import struct
# from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.

    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt

# ###############################################################################################################

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
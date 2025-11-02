#!/usr/bin/env python3

# Code Adapted from:
# https://github.com/CDonosoK/ros2_camera_lidar_fusion

import os
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml
import struct
import math

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer, Cache
from tf2_ros import TransformBroadcaster

from src.read_yaml import extract_configuration, load_extrinsic_matrix, load_camera_calibration
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped, Point, Vector3, Pose
from scipy.spatial.transform import Rotation as R
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox3D, BoundingBox3DArray, Detection3D, Detection3DArray
from src.fusion_node import lidar2pixel
from livox_ros_driver2.msg import CustomMsg

def pointcloud2_to_xyz_array_fast(cloud_msg: CustomMsg, skip_rate: int = 1) -> np.ndarray:
    if cloud_msg.point_num == 0:
        return np.zeros((0, 3), dtype=np.float32)

    data = np.array([[p.x, p.y, p.z] for p in cloud_msg.points], dtype=np.float32)

    # Mask out rows with any NaNs
    nan_mask = np.isnan(data).any(axis=1)

    data = data[~nan_mask]  # remove rows with NaNs

    # Apply downsampling if needed
    if skip_rate > 1:
        data = data[::skip_rate]

    return data

def reproject_to_lidar(u, v, depth, K, T_lidar_from_cam):
    try:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        if depth <= 0 or np.isnan(depth):
            return None

        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth

        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])  # homogeneous
        if np.any(np.isnan(point_cam)):
            print("nan encountered")

        point_lidar = T_lidar_from_cam @ point_cam
        if np.any(np.isnan(point_lidar)):
            return None
        return point_lidar[:3]
    except (ValueError, TypeError, RuntimeWarning):
        return None


class LidarCameraProjectionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_projection_node')
        
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = config_file['general']['camera_extrinsic_calibration']
        extrinsic_yaml = os.path.join(config_folder, extrinsic_yaml)
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)
        self.T_cam_to_lidar = np.linalg.inv(self.T_lidar_to_cam.T)
        self.img_size = [config_file['camera']['image_size']['height'], config_file['camera']['image_size']['width']]
        
        camera_yaml = config_file['general']['camera_intrinsic_calibration']
        camera_yaml = os.path.join(config_folder, camera_yaml)
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))

        lidar_topic = config_file['lidar']['lidar_topic']
        image_topic = '/dai_node/annotated_image'
        self.get_logger().info(f"Subscribing to lidar topic: {lidar_topic}")

        self.lidar_sub = Subscriber(self, CustomMsg, lidar_topic)
        # visualization False turns off camera 
        self.bounding_boxes = []  # will hold latest list of boxes
        self.visualize = config_file['fusion']['visualize']

        self.get_logger().info(f"Subscribing to bounding box topic: /oak/rgb/bounding_boxes")
        self.boundingbox_sub = Subscriber(self, Detection2DArray, '/oak/rgb/bounding_boxes')
        self.ts = ApproximateTimeSynchronizer(
            [self.boundingbox_sub, self.lidar_sub],
            queue_size=10,
            slop=0.07
            , allow_headerless=True
        )
        self.ts.registerCallback(self.sync_callback)

        self.latest_image = None
        if self.visualize:
            self.image_sub = self.create_subscription(
                Image,
                image_topic,
                self.image_callback,
                10
            )
            projected_topic = config_file['camera']['projected_topic']
            self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.pub_centr = self.create_publisher(Float32MultiArray, '/centroids', 1)
        self.br_tf = TransformBroadcaster(self)
        self.bridge = CvBridge()
        self.skip_rate = 1

        self.class_names = {11: "Green Buoy", 16: "Red Buoy", 23: "Yellow Buoy"}
        
        self.pub_3d = config_file['fusion']['publish_3d']
        if self.pub_3d:
            self.pub_bbox3d = self.create_publisher(Detection3DArray, '/bbox3d', 1)
            

    def image_callback(self, msg: Image):
        self.latest_image = msg

    def sync_callback(self, boundingbox_msg: Detection2DArray, lidar_msg: CustomMsg):
        if self.visualize:
            if self.latest_image is None:
                self.get_logger().warn("No image available for visualization.")
                return
            image_msg = self.latest_image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        xyz_lidar = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)
        xs, ys, ps = lidar2pixel(xyz_lidar.T, self.T_lidar_to_cam, self.camera_matrix)
        
        lb = np.array([0, 0])
        ub = np.array([self.img_size[1], self.img_size[0]]) # [H, W] to [W, H]
        
        points = np.stack((xs, ys), axis=1)
        
        filtered_idx = np.all(np.logical_and(lb <= points, points < ub), axis=1)
        filtered_points = points[filtered_idx]
        filtered_depths = ps[filtered_idx]

        h, w = self.img_size[:2]
        z_vals = ps
        z_norm = cv2.normalize(z_vals, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        z_norm = 255 - z_norm
        
        z_norm_uint8 = z_norm.astype(np.uint8)
        colors = cv2.applyColorMap(z_norm_uint8, cv2.COLORMAP_JET)
        filtered_colors = colors[filtered_idx]

        if self.visualize:
            for (pt, color) in zip(filtered_points, filtered_colors):
                u, v = pt
                u, v = int(u), int(v)
                b, g, r = color.flatten().tolist()
                cv2.circle(cv_image, (u, v), 2, (b, g, r), -1)       
   
        ys, xs = filtered_points[:, 1], filtered_points[:, 0]
        coords = np.stack((ys, xs))
        
        abs_coords = np.ravel_multi_index(coords, self.img_size)
        depth_lookup = np.bincount(abs_coords, weights=filtered_depths, minlength=self.img_size[0]*self.img_size[1])
        depth_lookup = depth_lookup.reshape(self.img_size)
        depth_lookup[depth_lookup == 0.0] = np.inf

        centroid_array = Float32MultiArray()
        centroid_data = []
        closest_red = -1
        closest_green = -1

        xydepth = Float32MultiArray()
        xydepth = []        
        closest_red_xydepth = [-1.0, -1.0, -1.0]
        closest_green_xydepth = [-1.0, -1.0, -1.0]

        if self.pub_3d:
            bbox3ds = Detection3DArray(header=boundingbox_msg.header, detections=[])
            bbox3ds.header.frame_id = "camera_init"

        for det in boundingbox_msg.detections:
            x1 = int(det.bbox.center.position.x - det.bbox.size_x / 2)
            x2 = int(det.bbox.center.position.x + det.bbox.size_x / 2)
            y1 = int(det.bbox.center.position.y - det.bbox.size_y / 2)
            y2 = int(det.bbox.center.position.y + det.bbox.size_y / 2)
            conf = det.results[0].hypothesis.score
            cls = int(det.results[0].hypothesis.class_id)
            
            if cls not in self.class_names.keys():
                return# continue
            
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w-1), min(y2, h-1)

            if len(filtered_depths) > 0:
                object_depth = np.min(depth_lookup[y1:y2, x1:x2])
                class_name = self.class_names[cls]
                label = f"{class_name} ({object_depth:.2f}) m"
                
                if self.pub_3d:
                    object_cam_pose = np.array([det.bbox.center.position.x, det.bbox.center.position.y, object_depth + 0.1])
                    object_lid_pose = reproject_to_lidar(object_cam_pose[0], object_cam_pose[1], object_cam_pose[2], self.camera_matrix, self.T_lidar_to_cam)
                    
                    if object_lid_pose is None:
                        continue
                    
                    # xyz -> yzx
                    object_pose = Pose(position=Point(x=object_lid_pose[1], y=object_lid_pose[2], z=object_lid_pose[0]))
                    object_size = Vector3(x=0.2, y=0.2, z=0.2)

                    bbox3d = BoundingBox3D(center=object_pose, size=object_size)
                    det3d = Detection3D(results=det.results, bbox=bbox3d)
                    bbox3ds.detections.append(det3d)

                if class_name == 'Green Buoy':
                    if closest_green == -1:
                        closest_green = object_depth
                    else:
                        closest_green = min(closest_green, object_depth)
                    if closest_green_xydepth[2] < 0 or closest_green_xydepth[2] > object_depth:
                        closest_green_xydepth = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, object_depth]
                elif class_name == 'Red Buoy':
                    if closest_red == -1:
                        closest_red = object_depth
                    else:
                        closest_red = min(closest_red, object_depth)
                    if closest_red_xydepth[2] < 0 or closest_green_xydepth[2] > object_depth:
                        closest_red_xydepth = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, object_depth]
                    
            else:
                label = "N/A"
        
        centroid_data.extend([closest_red, closest_green])

        # x, y, depth for closest bouys
        xydepth.extend(closest_red_xydepth)
        xydepth.extend(closest_green_xydepth)
        centroid_array.data = xydepth
        self.pub_centr.publish(centroid_array)
        
        if self.visualize:
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)

        if self.pub_3d:
            self.pub_bbox3d.publish(bbox3ds)            


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
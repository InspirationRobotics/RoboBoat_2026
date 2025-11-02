#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import numpy as np
import os
import sys
from filterpy.kalman import KalmanFilter
sys.path.insert(1, '../perception/')
from src.read_yaml import extract_configuration
import math
import argparse

class LandmarkMapper(Node):
    def __init__(self, simulate=False):
        super().__init__('landmark_mapper')
        self.simulate = simulate

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return

        self.odom_sub = self.create_subscription(
            Odometry, '/Odometry', self.odom_callback, 10)

        self.detections_sub = self.create_subscription(
            Detection3DArray, '/bbox3d', self.detections_callback, 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/landmark_markers', 10)
        self.map_pub = self.create_publisher(String, '/landmark_map', 10)

        self.map = {}  # key: class_id, value: list of KalmanFilters
        self.current_pose = np.eye(3)  # 2D SE(2) transformation matrix

        self.gt_map = {16: [
            [2.272991797174637, -0.7540841219225431], 
            [4.3550308808588706, -0.9208583673453902],
            [6.810385185167577, -0.8329025117515143], 
            ], 
            11: [
            [2.4550308808588706, 0.9794984520856926],
            [4.077538133981755, 1.0394984520856926], 
            [6.770867890677844, 0.6694984520856926], 
            ]
            }
        self.timer = self.create_timer(10.0, self.publish_and_save_map)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = np.arctan2(siny_cosp, cosy_cosp)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        self.current_pose = np.array([
            [cos_theta, -sin_theta, x],
            [sin_theta,  cos_theta, y],
            [0,          0,         1]
        ])
        
    def evaluate_map_error(self):
        total_error = 0.0
        total_count = 0

        for class_id, gt_points in self.gt_map.items():
            if class_id not in self.map:
                self.get_logger().warn(f"No detections for class {class_id}")
                continue

            est_points = [kf.x[:2] for kf in self.map[class_id]]
            gt_points = np.array(gt_points)

            # Optional: sort for consistent ordering (assumes close alignment)
            est_points = sorted(est_points, key=lambda p: p[0])
            gt_points = sorted(gt_points, key=lambda p: p[0])

            min_len = min(len(est_points), len(gt_points))
            errors = []
            for i in range(min_len):
                err = np.linalg.norm(np.array(est_points[i]) - np.array(gt_points[i]))
                errors.append(err)
                total_error += err
                total_count += 1

            avg_error = np.mean(errors) if errors else float('nan')
            self.get_logger().info(f"Class {class_id} avg error: {avg_error:.3f} m")

        overall_avg = total_error / total_count if total_count > 0 else float('nan')
        self.get_logger().info(f"Overall map RMSE: {overall_avg:.3f} m")
        return overall_avg


    def detections_callback(self, msg):
        X_THRESHOLD = 2 # meters
        Y_THRESHOLD = 3  # meters

        for det in msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            class_id = int(det.results[0].hypothesis.class_id)

            if class_id != 11 and class_id != 16:
                continue

            point_lidar = np.array([cx, cy, 1.0])
            global_pos = self.current_pose @ point_lidar
            global_pos = global_pos[:2]

            if class_id not in self.map:
                kf = self.initialize_kalman_filter(global_pos)
                self.map[class_id] = [kf]
            else:
                matched = False
                for kf in self.map[class_id]:
                    pred = kf.x[:2]
                    # Only match if y is close (aligned along x-axis)
                    # if np.linalg.norm(pred - global_pos) < 1.0:
                    if abs(pred[0] - global_pos[0]) < 0.8 or (abs(pred[0] - global_pos[0]) < X_THRESHOLD and abs(pred[1] - global_pos[1]) < Y_THRESHOLD):
                        kf.predict()
                        kf.update(global_pos)
                        matched = True
                        break
                if not matched:
                    # No existing buoy in this y-strip → create new
                    self.map[class_id].append(self.initialize_kalman_filter(global_pos))

        self.publish_markers()
        # self.evaluate_map_error()


    def initialize_kalman_filter(self, initial_pos):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.zeros(4)
        kf.x[:2] = initial_pos
        kf.F = np.eye(4)
        kf.F[:2, 2:] = np.eye(2)
        kf.H = np.eye(2, 4)
        kf.P *= 10.0
        kf.R *= 0.7
        kf.Q = np.eye(4) * 0.15
        return kf

    def publish_markers(self):
        marker_array = MarkerArray()
        marker_id = 1

        for class_id, filters in self.map.items():
            if class_id != 16 and class_id != 11:
                continue
            for kf in filters:
                pos = kf.x[:2]
                # pos = kf
                marker = Marker()
                marker.header.frame_id = "camera_init"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"landmark_{class_id}"
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(pos[0])
                marker.pose.position.y = float(pos[1])
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                if class_id == 16:  # Red buoy
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                elif class_id == 11:  # Green buoy
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                else:  # Default gray
                    marker.color.r = 0.5
                    marker.color.g = 0.5
                    marker.color.b = 0.5
                    
                marker.color.a = 1.0
                marker_array.markers.append(marker)
                marker_id += 1

        marker = Marker()
        marker.header.frame_id = "camera_init"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "self_pose"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        yaw_rad = math.pi / 2
        marker.pose.position.x = float(self.current_pose[0, 2])
        marker.pose.position.y = float(self.current_pose[1, 2])
        marker.pose.orientation.z = math.sin(yaw_rad / 2)  # ~0.7071
        marker.pose.orientation.w = math.cos(yaw_rad / 2)  # ~0.7071

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.mesh_resource = "package://perception/boat.dae" # TODO: Set up dae file and remove hard-coded path 
        marker.mesh_use_embedded_materials = True

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def publish_and_save_map(self):
        if self.simulate:
            simple_map = self.gt_map
            self.get_logger().info("Simulate=True: publishing GT map")
        else:
            simple_map = {cid: [kf.x[:2].tolist() for kf in kfs] for cid, kfs in self.map.items()}
            self.get_logger().info("Publishing estimated map")

        self.map_pub.publish(String(data=str(simple_map)))
        np.save('/tmp/landmark_map.npy', simple_map)

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description="Landmark Mapper Node")
    parser.add_argument('--simulate', action='store_true', help="Use ground truth map instead of detections")
    parsed_args, _ = parser.parse_known_args()

    node = LandmarkMapper(simulate=parsed_args.simulate)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

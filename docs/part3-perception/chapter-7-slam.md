---
title: "Chapter 7: Visual SLAM with Isaac ROS"
sidebar_position: 7
---

# Chapter 7: Visual SLAM with Isaac ROS

## 7.1 Introduction to VSLAM

### What is Visual SLAM?

Visual SLAM (Simultaneous Localization and Mapping) is the process of creating a map of an environment while simultaneously keeping track of the agent's location within that environment using visual sensors.

### Types of Visual SLAM

1. **Monocular SLAM**: Uses a single camera
2. **Stereo SLAM**: Uses two cameras for depth perception
3. **RGB-D SLAM**: Uses RGB-D cameras (like RealSense)
4. **Visual-Inertial SLAM**: Combines cameras with IMU sensors

### Key Components

- **Feature Extraction**: Detect and describe visual features
- **Feature Matching**: Track features across frames
- **Pose Estimation**: Determine camera motion
- **Bundle Adjustment**: Optimize map and poses
- **Loop Closure**: Detect when robot returns to known location

## 7.2 Hardware-Accelerated SLAM with NVIDIA Isaac ROS

### Isaac ROS SLAM Overview

NVIDIA Isaac ROS provides GPU-accelerated implementations of popular SLAM algorithms optimized for NVIDIA GPUs and Jetson platforms.

### Installing Isaac ROS SLAM

```bash
# Install Isaac ROS SLAM packages
sudo apt update
sudo apt install ros-humble-isaac-ros-visual-slam-ros2
sudo apt install ros-humble-isaac-ros-visual-slam-ros2-ws

# Install dependencies
pip install torch torchvision
```

### Isaac ROS Visual SLAM Configuration

```yaml
# config/isaac_slam_config.yaml
isaac_ros_visual_slam:
  ros__parameters:
    # Camera parameters
    camera_namespace: "/camera"
    image_topic: "color/image_raw"
    depth_image_topic: "depth/image_raw"
    camera_info_topic: "camera_info"
    
    # SLAM parameters
    enable_depth_processing: true
    enable_loop_closure: true
    enable_bundle_adjustment: true
    
    # Feature parameters
    max_features: 1000
    feature_quality_threshold: 0.01
    min_distance_between_features: 15.0
    
    # Tracking parameters
    max_tracking_distance: 50.0
    min_tracking_confidence: 0.3
    
    # Mapping parameters
    map_resolution: 0.05
    max_map_range: 100.0
    
    # GPU acceleration
    use_gpu: true
    gpu_device_id: 0
    
    # Output topics
    map_frame: "map"
    odom_frame: "odom"
    base_frame: "base_link"
    
    # Visualization
    publish_pointcloud_map: true
    publish_occupancy_grid: true
    publish_keyframe_poses: true
```

### Isaac ROS SLAM Node

```python
# isaac_ros_slam_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
import cv2
import numpy as np
import torch
from isaac_ros_visual_slam import VisualSLAM

class IsaacROSSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_ros_slam')
        
        # Initialize Isaac ROS Visual SLAM
        self.slam = VisualSLAM(
            config_file="config/isaac_slam_config.yaml",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        
        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped, '/slam/pose', 10)
        
        self.map_pub = self.create_publisher(
            OccupancyGrid, '/slam/map', 10)
        
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, '/slam/pointcloud', 10)
        
        # Camera info
        self.camera_info = None
        self.last_image = None
        self.last_depth = None
        
        # Timer for processing
        self.timer = self.create_timer(0.1, self.process_slam)
        
        self.get_logger().info('Isaac ROS SLAM node initialized')
    
    def camera_info_callback(self, msg):
        """Handle camera info"""
        self.camera_info = msg
    
    def image_callback(self, msg):
        """Handle RGB image"""
        self.last_image = msg
    
    def depth_callback(self, msg):
        """Handle depth image"""
        self.last_depth = msg
    
    def process_slam(self):
        """Process SLAM with latest images"""
        
        if self.last_image is None or self.last_depth is None or self.camera_info is None:
            return
        
        try:
            # Convert ROS messages to numpy arrays
            image = self.bridge.imgmsg_to_cv2(self.last_image, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(self.last_depth, "16UC1")
            
            # Process with SLAM
            pose, map_data, pointcloud = self.slam.process_frame(image, depth, self.camera_info)
            
            # Publish pose
            if pose is not None:
                self.publish_pose(pose)
            
            # Publish map
            if map_data is not None:
                self.publish_map(map_data)
            
            # Publish point cloud
            if pointcloud is not None:
                self.publish_pointcloud(pointcloud)
            
        except Exception as e:
            self.get_logger().error(f'SLAM processing error: {e}')
    
    def publish_pose(self, pose):
        """Publish robot pose"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        
        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.position.z = pose[2]
        
        pose_msg.pose.orientation.x = pose[3]
        pose_msg.pose.orientation.y = pose[4]
        pose_msg.pose.orientation.z = pose[5]
        pose_msg.pose.orientation.w = pose[6]
        
        self.pose_pub.publish(pose_msg)
    
    def publish_map(self, map_data):
        """Publish occupancy grid map"""
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = "map"
        
        map_msg.info.resolution = 0.05  # 5cm resolution
        map_msg.info.width = map_data.shape[1]
        map_msg.info.height = map_data.shape[0]
        map_msg.info.origin.position.x = -map_data.shape[1] * 0.05 / 2
        map_msg.info.origin.position.y = -map_data.shape[0] * 0.05 / 2
        
        # Convert map data
        map_data_flat = map_data.flatten().astype(np.int8)
        map_msg.data = map_data_flat.tolist()
        
        self.map_pub.publish(map_msg)
    
    def publish_pointcloud(self, pointcloud):
        """Publish point cloud map"""
        # Convert pointcloud to ROS message
        pc_msg = PointCloud2()
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        pc_msg.header.frame_id = "map"
        pc_msg.height = 1
        pc_msg.width = len(pointcloud)
        
        # Serialize points
        pc_msg.data = self.serialize_pointcloud(pointcloud)
        
        self.pointcloud_pub.publish(pc_msg)

def main(args=None):
    rclpy.init(args=args)
    
    slam_node = IsaacROSSLAMNode()
    
    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 7.3 Mapping & Localization for Humanoids

### Humanoid-Specific SLAM Challenges

1. **Dynamic Motion**: Humanoid robots have complex body movements
2. **Variable Camera Height**: Camera height changes during locomotion
3. **Occlusions**: Robot body parts can occlude the view
4. **Motion Blur**: Walking motion can cause image blur

### Adaptive SLAM for Humanoids

```python
# humanoid_slam.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

class HumanoidSLAM(Node):
    def __init__(self):
        super().__init__('humanoid_slam')
        
        # SLAM state
        self.current_pose = np.eye(4)
        self.keyframes = []
        self.map_points = []
        
        # Motion model
        self.last_pose = np.eye(4)
        self.velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped, '/humanoid/slam/pose', 10)
        
        # Feature detector
        self.detector = cv2.ORB_create(nfeatures=1000)
        
        # IMU integration
        self.imu_orientation = np.eye(3)
        self.imu_bias = np.zeros(3)
        
        # Motion compensation
        self.body_joints = {}
        self.camera_offset = np.array([0.1, 0, 0.5])  # Camera offset from body center
        
        self.get_logger().info('Humanoid SLAM initialized')
    
    def imu_callback(self, msg):
        """Handle IMU data for orientation"""
        
        # Extract quaternion
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        
        # Convert to rotation matrix
        rotation = Rotation([qw, qx, qy, qz])
        self.imu_orientation = rotation.as_matrix()
        
        # Extract angular velocity
        wx, wy, wz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        self.velocity[3:6] = np.array([wx, wy, wz])
    
    def odom_callback(self, msg):
        """Handle odometry data"""
        
        # Extract linear velocity
        vx, vy, vz = msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z
        self.velocity[0:3] = np.array([vx, vy, vz])
    
    def image_callback(self, msg):
        """Process image for SLAM"""
        
        # Convert to OpenCV format
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        
        if len(self.keyframes) == 0:
            # First frame - create keyframe
            self.create_keyframe(image, keypoints, descriptors, msg.header.stamp)
        else:
            # Track features
            self.track_features(image, keypoints, descriptors, msg.header.stamp)
    
    def create_keyframe(self, image, keypoints, descriptors, timestamp):
        """Create a new keyframe"""
        
        keyframe = {
            'image': image.copy(),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': self.current_pose.copy(),
            'timestamp': timestamp,
            'camera_pose': self.get_camera_pose()
        }
        
        self.keyframes.append(keyframe)
        self.last_pose = self.current_pose.copy()
    
    def get_camera_pose(self):
        """Get camera pose relative to body"""
        
        # Account for body orientation and camera offset
        body_pose = self.current_pose.copy()
        
        # Apply camera offset
        camera_pose = body_pose.copy()
        camera_pose[:3, 3] += body_pose[:3, :3] @ self.camera_offset
        
        return camera_pose
    
    def track_features(self, image, keypoints, descriptors, timestamp):
        """Track features and estimate motion"""
        
        if len(self.keyframes) == 0:
            return
        
        # Match with previous keyframe
        prev_keyframe = self.keyframes[-1]
        
        # Feature matching
        matches = self.match_features(prev_keyframe['descriptors'], descriptors)
        
        if len(matches) < 10:
            # Not enough matches - create new keyframe
            self.create_keyframe(image, keypoints, descriptors, timestamp)
            return
        
        # Estimate motion
        motion = self.estimate_motion(matches, prev_keyframe, keypoints)
        
        if motion is not None:
            # Update pose
            self.current_pose = motion @ self.current_pose
            
            # Check if new keyframe is needed
            if self.should_create_keyframe(motion):
                self.create_keyframe(image, keypoints, descriptors, timestamp)
        
        # Publish pose
        self.publish_pose()
    
    def match_features(self, desc1, desc2):
        """Match features between frames"""
        
        if desc1 is None or desc2 is None:
            return []
        
        # BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Filter good matches
        good_matches = []
        for match in matches:
            if match.distance < 50:  # Threshold
                good_matches.append(match)
        
        return good_matches
    
    def estimate_motion(self, matches, prev_keyframe, current_keypoints):
        """Estimate camera motion from feature matches"""
        
        # Extract matched points
        prev_pts = np.float32([prev_keyframe['keypoints'][m.queryIdx].pt for m in matches])
        curr_pts = np.float32([current_keypoints[m.trainIdx].pt for m in matches])
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            prev_pts, curr_pts, 
            camera_matrix=np.eye(3),  # Should use actual camera matrix
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None:
            return None
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, prev_pts, curr_pts)
        
        # Create transformation matrix
        motion = np.eye(4)
        motion[:3, :3] = R
        motion[:3, 3] = t.flatten()
        
        # Apply motion compensation for humanoid motion
        motion = self.compensate_body_motion(motion)
        
        return motion
    
    def compensate_body_motion(self, motion):
        """Compensate for body motion during walking"""
        
        # Use IMU data to correct orientation
        imu_correction = np.eye(4)
        imu_correction[:3, :3] = self.imu_orientation
        
        # Combine motion estimates
        compensated = imu_correction @ motion
        
        return compensated
    
    def should_create_keyframe(self, motion):
        """Determine if new keyframe is needed"""
        
        # Calculate motion magnitude
        translation = np.linalg.norm(motion[:3, 3])
        rotation_angle = np.arccos(np.clip((np.trace(motion[:3, :3]) - 1) / 2, -1, 1))
        
        # Create keyframe if motion is significant
        if translation > 0.2 or rotation_angle > 0.1:
            return True
        
        return False
    
    def publish_pose(self):
        """Publish current pose"""
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        
        # Extract position
        pose_msg.pose.position.x = self.current_pose[0, 3]
        pose_msg.pose.position.y = self.current_pose[1, 3]
        pose_msg.pose.position.z = self.current_pose[2, 3]
        
        # Extract orientation
        rotation = Rotation.from_matrix(self.current_pose[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]
        
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    
    slam_node = HumanoidSLAM()
    
    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 7.4 Path Planning with Nav2

### Nav2 Integration

Nav2 is the ROS 2 navigation stack that provides path planning and control capabilities.

### Nav2 Configuration for Humanoid

```yaml
# config/nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: True
    min_particles: 500
    max_particles: 2000
    update_factor_d: 0.7
    laser_z_hit: 0.95
    laser_z_short: 0.1
    laser_z_max: 0.05
    laser_z_rand: 0.05
    laser_sigma_hit: 0.2
    laser_model_type: "beam"
    odom_model_type: "diff"
    odom_alpha1: 0.1
    odom_alpha2: 0.1
    odom_alpha3: 0.1
    odom_alpha4: 0.1
    odom_alpha5: 0.1

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_goal_updated_controller_bt_node
    - nav2_is_battery_charging_condition_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugins: ["general_goal_checker"]
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::ProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    general_goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25

    # DWB parameters
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      debug_trajectory_details: False
      min_vel_x: 0.0
      min_vel_y: 0.0
      max_vel_x: 0.5
      max_vel_y: 0.0
      max_vel_theta: 1.0
      min_speed_xy: 0.0
      max_speed_xy: 0.5
      min_speed_theta: 0.0
      acc_lim_x: 2.5
      acc_lim_y: 0.0
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_y: 0.0
      decel_lim_theta: -3.2
      vx_samples: 20
      vy_samples: 1
      vtheta_samples: 20
      sim_time: 1.7
      linear_granularity: 0.05
      angular_granularity: 0.025
      transform_tolerance: 0.2
      xy_goal_tolerance: 0.25
      trans_stopped_velocity: 0.25
      short_circuit_trajectory_evaluation: True

      critics:
        - plugin: "dwb_critics::ObstacleFootprintCritic"
          enabled: True
          cost_scaling: 0.1
        - plugin: "dwb_critics::PathAlignCritic"
          enabled: True
          cost_scaling: 32.0
          max_angle: 0.2
        - plugin: "dwb_critics::PathDistCritic"
          enabled: True
          cost_scaling: 32.0
        - plugin: "dwb_critics::GoalAlignCritic"
          enabled: True
          cost_scaling: 24.0
          max_angle: 0.2
        - plugin: "dwb_critics::PathAngleCritic"
          enabled: True
          cost_scaling: 12.0
          max_angle: 0.2
        - plugin: "dwb_critics::GoalDistCritic"
          enabled: True
          cost_scaling: 20.0
        - plugin: "dwb_critors::TwirlingCritic"
          enabled: True
          cost_scaling: 12.0

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: False
      allow_unknown: True

behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    robot_base_frame: base_link
    global_frame: odom
    transform_tolerance: 0.1
    use_sim_time: True

local_costmap:
  global_frame: odom
  robot_base_frame: base_link
  use_sim_time: True
  rolling_window: true
  width: 3
  height: 3
  resolution: 0.05
  origin_x: -1.5
  origin_y: -1.5
  track_unknown_space: false
  plugins: ["voxel_layer", "inflation_layer"]
  inflation_layer:
    plugin: "nav2_costmap_2d::InflationLayer"
    cost_scaling_factor: 3.0
    inflation_radius: 0.55
  voxel_layer:
    plugin: "nav2_costmap_2d::VoxelLayer"
    enabled: True
    footprint_clearing_enabled: True
    max_obstacle_height: 2.0
    origin_z: 0.0
    z_resolution: 0.2
    z_voxels: 10
    unknown_threshold: 15
    mark_threshold: 0
    combination_method: 1
    track_unknown_space: True
    obstacle_layer:
      plugin: "nav2_costmap_2d::ObstacleLayer"
      enabled: True
      max_obstacle_height: 2.0
      origin_z: 0.0
      z_resolution: 0.2
      z_voxels: 10
      unknown_threshold: 15
      mark_threshold: 0
      combination_method: 1
      track_unknown_space: True
      obstacle_range: 2.5
      raytrace_range: 3.0
      inflation_radius: 0.55
      cost_scaling_factor: 3.0
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_range: 3.0
        obstacle_range: 2.5

global_costmap:
  global_frame: map
  robot_base_frame: base_link
  use_sim_time: True
  robot_radius: 0.22
  resolution: 0.05
  track_unknown_space: false
  plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
  static_layer:
    plugin: "nav2_costmap_2d::StaticLayer"
    map_subscribe_transient_local: True
  obstacle_layer:
    plugin: "nav2_costmap_2d::ObstacleLayer"
    enabled: True
    max_obstacle_height: 2.0
    origin_z: 0.0
    z_resolution: 0.2
    z_voxels: 10
    unknown_threshold: 15
    mark_threshold: 0
    combination_method: 1
    track_unknown_space: True
    obstacle_range: 2.5
    raytrace_range: 3.0
    inflation_radius: 0.55
    cost_scaling_factor: 3.0
    observation_sources: scan
    scan:
      topic: /scan
      max_obstacle_height: 2.0
      clearing: True
      marking: True
      data_type: "LaserScan"
      raytrace_range: 3.0
      obstacle_range: 2.5
  inflation_layer:
    plugin: "nav2_costmap_2d::InflationLayer"
    cost_scaling_factor: 3.0
    inflation_radius: 0.55
```

### Humanoid Path Planning Controller

```python
# humanoid_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import JointState
import numpy as np
import math

class HumanoidPathController(Node):
    def __init__(self):
        super().__init__('humanoid_path_controller')
        
        # Path planning state
        self.current_path = None
        self.current_waypoint_index = 0
        self.current_pose = None
        
        # Walking parameters
        self.step_length = 0.3
        self.step_height = 0.1
        self.walking_speed = 0.5
        
        # Subscribers
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        
        self.joint_pub = self.create_publisher(
            JointState, '/joint_commands', 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Walking state
        self.walking_phase = 0.0
        self.is_walking = False
        
        self.get_logger().info('Humanoid path controller initialized')
    
    def path_callback(self, msg):
        """Handle new path"""
        self.current_path = msg
        self.current_waypoint_index = 0
        self.is_walking = True
        self.get_logger().info(f'Received path with {len(msg.poses)} waypoints')
    
    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.current_pose = msg.pose.pose
    
    def control_loop(self):
        """Main control loop"""
        
        if not self.is_walking or self.current_path is None:
            return
        
        if self.current_waypoint_index >= len(self.current_path.poses):
            # Path completed
            self.stop_walking()
            return
        
        # Get current waypoint
        target_waypoint = self.current_path.poses[self.current_waypoint_index]
        
        # Calculate distance to waypoint
        if self.current_pose:
            distance = self.calculate_distance(self.current_pose, target_waypoint.pose)
            
            if distance < 0.2:  # Reached waypoint
                self.current_waypoint_index += 1
                self.get_logger().info(f'Reached waypoint {self.current_waypoint_index}')
            else:
                # Generate walking commands
                cmd = self.generate_walking_command(self.current_pose, target_waypoint.pose)
                self.cmd_vel_pub.publish(cmd)
                
                # Generate joint commands for walking
                joint_cmd = self.generate_walking_joints()
                self.joint_pub.publish(joint_cmd)
    
    def calculate_distance(self, current_pose, target_pose):
        """Calculate distance between poses"""
        
        dx = target_pose.position.x - current_pose.position.x
        dy = target_pose.position.y - current_pose.position.y
        dz = target_pose.position.z - current_pose.position.z
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def generate_walking_command(self, current_pose, target_pose):
        """Generate velocity command for walking"""
        
        # Calculate direction to target
        dx = target_pose.position.x - current_pose.position.x
        dy = target_pose.position.y - current_pose.position.y
        
        # Calculate desired heading
        desired_heading = math.atan2(dy, dx)
        
        # Current heading from quaternion
        current_heading = self.get_heading_from_quaternion(current_pose.orientation)
        
        # Calculate heading error
        heading_error = self.normalize_angle(desired_heading - current_heading)
        
        # Generate velocity command
        cmd = Twist()
        
        # Linear velocity (proportional to distance)
        distance = self.calculate_distance(current_pose, target_pose)
        cmd.linear.x = min(self.walking_speed, distance * 0.5)
        
        # Angular velocity (proportional to heading error)
        cmd.angular.z = np.clip(heading_error * 2.0, -1.0, 1.0)
        
        return cmd
    
    def get_heading_from_quaternion(self, quaternion):
        """Extract heading angle from quaternion"""
        
        # Quaternion to Euler angles
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        
        return math.atan2(siny_cosp, cosy_cosp)
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def generate_walking_joints(self):
        """Generate joint commands for walking"""
        
        # Update walking phase
        self.walking_phase += 0.2  # Walking frequency
        
        # Create joint state message
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        
        joint_cmd.name = [
            'left_hip_pitch', 'left_knee', 'left_ankle',
            'right_hip_pitch', 'right_knee', 'right_ankle',
            'left_shoulder_pitch', 'left_elbow',
            'right_shoulder_pitch', 'right_elbow'
        ]
        
        # Generate walking pattern
        positions = []
        
        # Leg joints
        for i, name in enumerate(joint_cmd.name):
            if 'hip' in name:
                if 'left' in name:
                    angle = 0.3 * math.sin(self.walking_phase)
                else:
                    angle = 0.3 * math.sin(self.walking_phase + math.pi)
                positions.append(angle)
            elif 'knee' in name:
                if 'left' in name:
                    angle = -0.6 * abs(math.sin(self.walking_phase))
                else:
                    angle = -0.6 * abs(math.sin(self.walking_phase + math.pi))
                positions.append(angle)
            elif 'ankle' in name:
                angle = 0.1 * math.sin(self.walking_phase * 2)
                positions.append(angle)
            elif 'shoulder' in name:
                if 'left' in name:
                    angle = 0.2 * math.sin(self.walking_phase + math.pi)
                else:
                    angle = 0.2 * math.sin(self.walking_phase)
                positions.append(angle)
            elif 'elbow' in name:
                angle = 0.3 * abs(math.sin(self.walking_phase))
                positions.append(angle)
        
        joint_cmd.position = positions
        
        return joint_cmd
    
    def stop_walking(self):
        """Stop walking motion"""
        
        self.is_walking = False
        
        # Send zero velocity
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        
        # Send standing joint positions
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.name = [
            'left_hip_pitch', 'left_knee', 'left_ankle',
            'right_hip_pitch', 'right_knee', 'right_ankle',
            'left_shoulder_pitch', 'left_elbow',
            'right_shoulder_pitch', 'right_elbow'
        ]
        joint_cmd.position = [0.0] * len(joint_cmd.name)
        self.joint_pub.publish(joint_cmd)
        
        self.get_logger().info('Walking completed')

def main(args=None):
    rclpy.init(args=args)
    
    controller = HumanoidPathController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Launch File for Complete SLAM System

```python
# launch/humanoid_slam.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import FindPackageShare
import os

def generate_launch_description():
    
    # Get package paths
    pkg_share = FindPackageShare(package='humanoid_robot').find('humanoid_robot')
    nav2_bringup_dir = FindPackageShare(package='nav2_bringup').find('nav2_bringup')
    
    # SLAM node
    slam_node = Node(
        package='humanoid_robot',
        executable='humanoid_slam',
        name='humanoid_slam',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    
    # Path controller
    controller_node = Node(
        package='humanoid_robot',
        executable='humanoid_path_controller',
        name='humanoid_path_controller',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    
    # Include Nav2 bringup
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_bringup_dir, '/launch', '/navigation_launch.py']),
        launch_arguments={
            'use_sim_time': 'True',
            'params_file': os.path.join(pkg_share, 'config', 'nav2_params.yaml'),
            'autostart': 'True'
        }.items()
    )
    
    return LaunchDescription([
        slam_node,
        controller_node,
        nav2_bringup
    ])
```

---

**Next Chapter**: In Chapter 8, we'll explore Vision-Language-Action (VLA) models that bridge LLMs with robotics for natural language control.

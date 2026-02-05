---
title: "Chapter 6: Robot Vision with RealSense & OpenCV"
sidebar_position: 6
---

# Chapter 6: Robot Vision with RealSense & OpenCV

## 6.1 RGB-D Data Processing

### Introduction to RGB-D Cameras

RGB-D cameras provide both color (RGB) and depth (D) information, enabling robots to perceive the 3D structure of their environment. The Intel RealSense series is the most popular RGB-D camera for robotics.

### RealSense Camera Setup

```python
# realsense_setup.py
import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.infrared, width, height, rs.format.y8, fps)
        
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Create alignment object
        self.align = rs.align(rs.stream.color)
        
        print(f"Camera initialized with depth scale: {self.depth_scale}")
    
    def get_frames(self):
        """Get aligned RGB and depth frames"""
        frames = self.pipeline.wait_for_frames()
        
        # Align depth to color frame
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        infrared_frame = aligned_frames.get_infrared_frame()
        
        if not aligned_depth_frame or not color_frame:
            return None, None, None
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        infrared_image = np.asanyarray(infrared_frame.get_data())
        
        return color_image, depth_image, infrared_image
    
    def get_point_cloud(self, color_image, depth_image):
        """Generate point cloud from RGB-D data"""
        
        # Get intrinsics
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        
        # Create point cloud
        pc = rs.pointcloud()
        points = pc.calculate(depth_image)
        
        # Map color to points
        pc.map_to(color_image)
        
        # Export to numpy
        vertices = np.asanyarray(points.get_vertices())
        colors = np.asanyarray(points.get_texture_coordinates())
        
        return vertices, colors
    
    def stop(self):
        """Stop camera streaming"""
        self.pipeline.stop()

# Usage example
camera = RealSenseCamera()

while True:
    color, depth, infrared = camera.get_frames()
    
    if color is not None:
        # Display images
        cv2.imshow('RGB', color)
        cv2.imshow('Depth', depth)
        cv2.imshow('Infrared', infrared)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.stop()
cv2.destroyAllWindows()
```

### Depth Image Processing

```python
# depth_processing.py
import numpy as np
import cv2
from scipy import ndimage

class DepthProcessor:
    def __init__(self, depth_scale=0.001):
        self.depth_scale = depth_scale
    
    def depth_to_meters(self, depth_image):
        """Convert depth values to meters"""
        return depth_image.astype(np.float32) * self.depth_scale
    
    def apply_filters(self, depth_image):
        """Apply noise reduction filters to depth data"""
        
        # Bilateral filter for edge-preserving smoothing
        filtered = cv2.bilateralFilter(depth_image, 9, 75, 75)
        
        # Median filter for salt-and-pepper noise
        filtered = cv2.medianBlur(filtered, 5)
        
        # Fill small holes
        kernel = np.ones((3,3), np.uint8)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        
        return filtered
    
    def segment_depth(self, depth_image, min_depth=0.5, max_depth=5.0):
        """Segment depth image into depth ranges"""
        
        # Convert to meters
        depth_meters = self.depth_to_meters(depth_image)
        
        # Create masks for different depth ranges
        near_mask = (depth_meters >= min_depth) & (depth_meters < 1.0)
        mid_mask = (depth_meters >= 1.0) & (depth_meters < 3.0)
        far_mask = (depth_meters >= 3.0) & (depth_meters <= max_depth)
        
        return near_mask, mid_mask, far_mask
    
    def detect_obstacles(self, depth_image, ground_threshold=0.1):
        """Detect obstacles in depth image"""
        
        # Apply filters
        filtered_depth = self.apply_filters(depth_image)
        
        # Convert to meters
        depth_meters = self.depth_to_meters(filtered_depth)
        
        # Estimate ground plane (simplified)
        ground_level = np.median(depth_meters[depth_meters > 0])
        
        # Find obstacles (points above ground)
        obstacle_mask = (depth_meters > 0) & (np.abs(depth_meters - ground_level) > ground_threshold)
        
        # Find contours of obstacles
        contours, _ = cv2.findContours(obstacle_mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small objects
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Get average depth of obstacle
                obstacle_depth = np.mean(depth_meters[y:y+h, x:x+w])
                
                obstacles.append({
                    'bbox': (x, y, w, h),
                    'depth': obstacle_depth,
                    'area': cv2.contourArea(contour)
                })
        
        return obstacles
    
    def create_occupancy_grid(self, depth_image, grid_size=0.1, max_range=5.0):
        """Create 2D occupancy grid from depth data"""
        
        # Convert to meters
        depth_meters = self.depth_to_meters(depth_image)
        
        # Grid dimensions
        grid_width = int(max_range * 2 / grid_size)
        grid_height = int(max_range * 2 / grid_size)
        
        # Initialize grid
        occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.int8)
        
        # Camera intrinsics (example values - should be calibrated)
        fx, fy = 525.0, 525.0  # Focal lengths
        cx, cy = depth_image.shape[1] / 2, depth_image.shape[0] / 2  # Principal points
        
        # Project depth to 2D grid
        for v in range(0, depth_image.shape[0], 10):  # Sample every 10 pixels
            for u in range(0, depth_image.shape[1], 10):
                depth = depth_meters[v, u]
                
                if 0.1 < depth < max_range:  # Valid depth range
                    # Convert to 3D coordinates
                    x = (u - cx) * depth / fx
                    z = depth
                    y = (v - cy) * depth / fy
                    
                    # Convert to grid coordinates
                    grid_x = int((x + max_range) / grid_size)
                    grid_y = int((y + max_range) / grid_size)
                    
                    # Update occupancy grid
                    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                        occupancy_grid[grid_y, grid_x] = 1
        
        return occupancy_grid

# Usage example
processor = DepthProcessor()

# Process depth frame
color, depth, _ = camera.get_frames()
if depth is not None:
    # Apply filters
    filtered_depth = processor.apply_filters(depth)
    
    # Detect obstacles
    obstacles = processor.detect_obstacles(filtered_depth)
    
    # Create occupancy grid
    occupancy_grid = processor.create_occupancy_grid(depth)
    
    # Visualize results
    cv2.imshow('Filtered Depth', filtered_depth)
    cv2.imshow('Occupancy Grid', occupancy_grid * 255)
```

## 6.2 Object Detection & Segmentation

### YOLO Object Detection

```python
# yolo_detection.py
import torch
import cv2
import numpy as np
from torchvision import transforms

class YOLODetector:
    def __init__(self, model_path='yolov5s.pt', confidence=0.5, iou_threshold=0.4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = confidence
        self.model.iou = iou_threshold
        self.model.to(self.device)
        
        # Class names for COCO dataset
        self.class_names = self.model.names
        
    def detect(self, image):
        """Detect objects in image"""
        
        # Run inference
        results = self.model(image)
        
        # Parse results
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_name = self.class_names[class_id]
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class_id': class_id,
                'class_name': class_name
            })
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw detection results on image"""
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image

# Usage example
detector = YOLODetector()

while True:
    color, depth, _ = camera.get_frames()
    if color is not None:
        # Detect objects
        detections = detector.detect(color)
        
        # Draw results
        result_image = detector.draw_detections(color.copy(), detections)
        
        cv2.imshow('Object Detection', result_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

### Semantic Segmentation

```python
# semantic_segmentation.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

class DeepLabSegmentation:
    def __init__(self, num_classes=21):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained DeepLabV3
        self.model = models.segmentation.deeplabv3_resnet101(
            pretrained=True, num_classes=num_classes
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Class names (PASCAL VOC)
        self.class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    
    def segment(self, image):
        """Perform semantic segmentation"""
        
        # Preprocess image
        original_size = image.shape[:2]
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)['out']
        
        # Get segmentation map
        segmentation = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
        # Resize to original size
        segmentation = cv2.resize(segmentation, (original_size[1], original_size[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        return segmentation
    
    def create_colored_segmentation(self, segmentation):
        """Create colored visualization of segmentation"""
        
        # Create color map
        colors = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
            (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
            (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
            (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
            (0, 64, 128)
        ]
        
        # Create colored image
        colored_seg = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        
        for class_id in range(len(colors)):
            mask = segmentation == class_id
            colored_seg[mask] = colors[class_id]
        
        return colored_seg
    
    def get_class_masks(self, segmentation, target_classes):
        """Get binary masks for specific classes"""
        
        masks = {}
        for class_name in target_classes:
            if class_name in self.class_names:
                class_id = self.class_names.index(class_name)
                masks[class_name] = (segmentation == class_id).astype(np.uint8) * 255
        
        return masks

# Usage example
segmentator = DeepLabSegmentation()

while True:
    color, depth, _ = camera.get_frames()
    if color is not None:
        # Perform segmentation
        segmentation = segmentator.segment(color)
        
        # Create colored visualization
        colored_seg = segmentator.create_colored_segmentation(segmentation)
        
        # Get specific class masks
        target_classes = ['person', 'chair', 'bottle']
        masks = segmentator.get_class_masks(segmentation, target_classes)
        
        # Visualize
        cv2.imshow('Original', color)
        cv2.imshow('Segmentation', colored_seg)
        
        # Overlay masks on original
        overlay = color.copy()
        for class_name, mask in masks.items():
            color_mask = np.zeros_like(color)
            color_mask[mask > 0] = (0, 255, 0)
            overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
        
        cv2.imshow('Overlay', overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

## 6.3 Visual Odometry & SLAM Basics

### Visual Odometry

```python
# visual_odometry.py
import cv2
import numpy as np
from collections import deque

class VisualOdometry:
    def __init__(self, focal_length=525.0, principal_point=(320.0, 240.0)):
        self.focal_length = focal_length
        self.cx, self.cy = principal_point
        
        # Feature detector
        self.detector = cv2.ORB_create(nfeatures=2000)
        
        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Camera matrix
        self.K = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]],
            [0, 0, 1]
        ])
        
        # Trajectory
        self.trajectory = deque(maxlen=1000)
        self.current_pose = np.eye(4)
        
        # Previous frame data
        self.prev_keypoints = None
        self.prev_descriptors = None
    
    def process_frame(self, image, depth_image=None):
        """Process frame and estimate motion"""
        
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        
        if self.prev_keypoints is None:
            # First frame
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.current_pose
        
        # Match features
        matches = self.matcher.match(self.prev_descriptors, descriptors)
        
        if len(matches) < 10:
            return self.current_pose
        
        # Extract matched points
        prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches])
        
        # Estimate motion
        if depth_image is not None:
            # Use depth for scale
            motion = self.estimate_motion_3d(prev_pts, curr_pts, depth_image)
        else:
            # Essential matrix decomposition
            motion = self.estimate_motion_2d(prev_pts, curr_pts)
        
        if motion is not None:
            # Update pose
            self.current_pose = motion @ self.current_pose
            self.trajectory.append(self.current_pose[:3, 3].copy())
        
        # Update previous frame
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return self.current_pose
    
    def estimate_motion_2d(self, prev_pts, curr_pts):
        """Estimate motion using essential matrix"""
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(prev_pts, curr_pts, self.K, 
                                      method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            return None
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, prev_pts, curr_pts, self.K)
        
        # Create transformation matrix
        motion = np.eye(4)
        motion[:3, :3] = R
        motion[:3, 3] = t.flatten()
        
        return motion
    
    def estimate_motion_3d(self, prev_pts, curr_pts, depth_image):
        """Estimate motion using 3D points"""
        
        # Get 3D points from depth
        prev_3d = []
        curr_3d = []
        
        for (x1, y1), (x2, y2) in zip(prev_pts, curr_pts):
            # Get depth values
            d1 = depth_image[int(y1), int(x1)]
            d2 = depth_image[int(y2), int(x2)]
            
            if d1 > 0 and d2 > 0:
                # Convert to 3D
                X1 = (x1 - self.cx) * d1 / self.focal_length
                Y1 = (y1 - self.cy) * d1 / self.focal_length
                Z1 = d1
                
                X2 = (x2 - self.cx) * d2 / self.focal_length
                Y2 = (y2 - self.cy) * d2 / self.focal_length
                Z2 = d2
                
                prev_3d.append([X1, Y1, Z1])
                curr_3d.append([X2, Y2, Z2])
        
        if len(prev_3d) < 5:
            return None
        
        prev_3d = np.array(prev_3d)
        curr_3d = np.array(curr_3d)
        
        # Estimate transformation using Kabsch algorithm
        # Compute centroids
        centroid_prev = np.mean(prev_3d, axis=0)
        centroid_curr = np.mean(curr_3d, axis=0)
        
        # Center the points
        prev_centered = prev_3d - centroid_prev
        curr_centered = curr_3d - centroid_curr
        
        # Compute rotation using SVD
        H = prev_centered.T @ curr_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = centroid_curr - R @ centroid_prev
        
        # Create transformation matrix
        motion = np.eye(4)
        motion[:3, :3] = R
        motion[:3, 3] = t
        
        return motion
    
    def draw_trajectory(self, image):
        """Draw trajectory on image"""
        
        if len(self.trajectory) < 2:
            return image
        
        # Create trajectory image
        traj_img = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Normalize trajectory to image coordinates
        trajectory = np.array(self.trajectory)
        
        if len(trajectory) > 0:
            # Center and scale
            centered = trajectory - trajectory[0]
            scale = 50 / (np.max(np.abs(centered)) + 1e-6)
            scaled = centered * scale + 150
            
            # Draw trajectory
            for i in range(1, len(scaled)):
                cv2.line(traj_img, 
                        (int(scaled[i-1, 0]), int(scaled[i-1, 1])),
                        (int(scaled[i, 0]), int(scaled[i, 1])),
                        (0, 255, 0), 2)
            
            # Draw current position
            cv2.circle(traj_img, 
                      (int(scaled[-1, 0]), int(scaled[-1, 1])),
                      5, (0, 0, 255), -1)
        
        return traj_img

# Usage example
vo = VisualOdometry()

while True:
    color, depth, _ = camera.get_frames()
    if color is not None:
        # Estimate motion
        pose = vo.process_frame(color, depth)
        
        # Draw trajectory
        traj_img = vo.draw_trajectory(color)
        
        # Display
        cv2.imshow('Camera', color)
        cv2.imshow('Trajectory', traj_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

## 6.4 Integrating Cameras with ROS 2

### ROS 2 Camera Node

```python
# ros2_camera_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
import cv2
import numpy as np
import pyrealsense2 as rs
from cv_bridge import CvBridge

class RealSenseROS2Node(Node):
    def __init__(self):
        super().__init__('realsense_camera')
        
        # CvBridge for converting between ROS and OpenCV
        self.bridge = CvBridge()
        
        # Publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/camera/points', 10)
        
        # Camera parameters
        self.width = 640
        self.height = 480
        self.fps = 30
        
        # Initialize RealSense camera
        self.setup_camera()
        
        # Camera info
        self.setup_camera_info()
        
        # Timer for publishing
        self.timer = self.create_timer(1.0/self.fps, self.publish_frames)
        
        self.get_logger().info('RealSense ROS 2 node started')
    
    def setup_camera(self):
        """Initialize RealSense camera"""
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Create alignment
        self.align = rs.align(rs.stream.color)
        
        # Get intrinsics
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        self.intrinsics = color_profile.get_intrinsics()
    
    def setup_camera_info(self):
        """Setup camera info message"""
        
        self.camera_info = CameraInfo()
        self.camera_info.header.frame_id = "camera_link"
        self.camera_info.width = self.width
        self.camera_info.height = self.height
        
        # Distortion model
        self.camera_info.distortion_model = "plumb_bob"
        self.camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Camera matrix (K)
        self.camera_info.k = [
            self.intrinsics.fx, 0.0, self.intrinsics.ppx,
            0.0, self.intrinsics.fy, self.intrinsics.ppy,
            0.0, 0.0, 1.0
        ]
        
        # Rectification matrix
        self.camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        
        # Projection matrix (P)
        self.camera_info.p = [
            self.intrinsics.fx, 0.0, self.intrinsics.ppx, 0.0,
            0.0, self.intrinsics.fy, self.intrinsics.ppy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
    
    def publish_frames(self):
        """Publish camera frames"""
        
        # Get frames
        frames = self.pipeline.wait_for_frames()
        
        # Align frames
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            return
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Create header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera_link"
        
        # Publish RGB image
        rgb_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        rgb_msg.header = header
        self.rgb_pub.publish(rgb_msg)
        
        # Publish depth image
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
        depth_msg.header = header
        self.depth_pub.publish(depth_msg)
        
        # Publish camera info
        self.camera_info.header = header
        self.camera_info_pub.publish(self.camera_info)
        
        # Publish point cloud (optional)
        pointcloud_msg = self.create_pointcloud_msg(color_image, depth_image, header)
        self.pointcloud_pub.publish(pointcloud_msg)
    
    def create_pointcloud_msg(self, color_image, depth_image, header):
        """Create point cloud message"""
        
        points = []
        
        # Sample points (reduce density)
        step = 4
        for v in range(0, self.height, step):
            for u in range(0, self.width, step):
                depth = depth_image[v, u] * self.depth_scale
                
                if 0.1 < depth < 10.0:  # Valid depth range
                    # Convert to 3D
                    x = (u - self.intrinsics.ppx) * depth / self.intrinsics.fx
                    y = (v - self.intrinsics.ppy) * depth / self.intrinsics.fy
                    z = depth
                    
                    point = Point32()
                    point.x = x
                    point.y = y
                    point.z = z
                    points.append(point)
        
        # Create point cloud message
        pointcloud = PointCloud2()
        pointcloud.header = header
        pointcloud.height = 1
        pointcloud.width = len(points)
        
        # Serialize points
        pointcloud.data = self.serialize_points(points)
        
        return pointcloud
    
    def serialize_points(self, points):
        """Serialize points to binary data"""
        import struct
        
        data = bytearray()
        for point in points:
            data.extend(struct.pack('fff', point.x, point.y, point.z))
        
        return bytes(data)
    
    def destroy(self):
        """Clean up resources"""
        self.pipeline.stop()

def main(args=None):
    rclpy.init(args=args)
    
    camera_node = RealSenseROS2Node()
    
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        camera_node.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Launch File

```python
# launch/camera.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    # Declare launch arguments
    width_arg = DeclareLaunchArgument(
        'width',
        default_value='640',
        description='Camera image width'
    )
    
    height_arg = DeclareLaunchArgument(
        'height',
        default_value='480',
        description='Camera image height'
    )
    
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30',
        description='Camera frame rate'
    )
    
    # Camera node
    camera_node = Node(
        package='humanoid_robot',
        executable='realsense_camera',
        name='realsense_camera',
        parameters=[
            {'width': LaunchConfiguration('width')},
            {'height': LaunchConfiguration('height')},
            {'fps': LaunchConfiguration('fps')}
        ],
        output='screen'
    )
    
    # Image viewer node (optional)
    viewer_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='image_viewer',
        arguments=['/camera/color/image_raw']
    )
    
    return LaunchDescription([
        width_arg,
        height_arg,
        fps_arg,
        camera_node,
        viewer_node
    ])
```

---

**Next Chapter**: In Chapter 7, we'll explore Visual SLAM with NVIDIA Isaac ROS for mapping and localization of humanoid robots.

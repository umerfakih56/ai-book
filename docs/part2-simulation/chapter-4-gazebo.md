---
title: "Chapter 4: Physics Simulation with Gazebo"
sidebar_position: 4
---

# Chapter 4: Physics Simulation with Gazebo

## 4.1 Setting Up Gazebo Simulation Environment

### Introduction to Gazebo

Gazebo is a 3D simulation environment for robotics that provides realistic physics, sensor simulation, and rendering capabilities. It's the most widely used simulator in the ROS ecosystem.

### Installing Gazebo

#### Ubuntu 22.04 (Recommended)

```bash
# Install Gazebo Fortress (ROS 2 compatible)
sudo apt update
sudo apt install gazebo-fortress libgazebo-fortress-dev

# Install ROS 2 Gazebo packages
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-gazebo-ros2-control
sudo apt install ros-humble-ros2-control
sudo apt install ros-humble-ros2-controllers
```

#### Verify Installation

```bash
# Check Gazebo version
gazebo --version

# Test Gazebo
gazebo
```

### Gazebo World Files

World files define the simulation environment:

```xml
<!-- worlds/humanoid_env.world -->
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="humanoid_environment">
    
    <!-- Physics Engine Settings -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      
      <!-- Gravity -->
      <gravity>0 0 -9.8066</gravity>
      
      <!-- ODE Solver -->
      <ode>
        <solver>
          <type>quick</type>
          <iters>50</iters>
          <sor>1.3</sor>
          <use_dynamic_moi_rescaling>0</use_dynamic_moi_rescaling>
        </solver>
        <constraints>
          <cfm>0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    
    <!-- Ground Plane -->
    <include>
      <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Ground Plane</uri>
    </include>
    
    <!-- Sun -->
    <include>
      <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Sun</uri>
    </include>
    
    <!-- Indoor Environment -->
    <model name="indoor_room">
      <static>true</static>
      
      <!-- Floor -->
      <link name="floor">
        <pose>0 0 0 0 0 0</pose>
        <collision name="floor_collision">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <surface>
            <contact>
              <ode>
                <max_vel>100</max_vel>
                <min_depth>0.001</min_depth>
              </ode>
            </contact>
            <friction>
              <ode>
                <mu>0.9</mu>
                <mu2>0.9</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="floor_visual">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
      
      <!-- Walls -->
      <link name="wall_north">
        <pose>0 5 2.5 0 0 0</pose>
        <collision name="wall_north_collision">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall_north_visual">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.9 1</ambient>
            <diffuse>0.8 0.8 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      
      <link name="wall_south">
        <pose>0 -5 2.5 0 0 0</pose>
        <collision name="wall_south_collision">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall_south_visual">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.9 1</ambient>
            <diffuse>0.8 0.8 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      
      <link name="wall_east">
        <pose>5 0 2.5 0 0 0</pose>
        <collision name="wall_east_collision">
          <geometry>
            <box>
              <size>0.2 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall_east_visual">
          <geometry>
            <box>
              <size>0.2 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.9 1</ambient>
            <diffuse>0.8 0.8 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      
      <link name="wall_west">
        <pose>-5 0 2.5 0 0 0</pose>
        <collision name="wall_west_collision">
          <geometry>
            <box>
              <size>0.2 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall_west_visual">
          <geometry>
            <box>
              <size>0.2 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.9 1</ambient>
            <diffuse>0.8 0.8 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      
      <!-- Obstacles for navigation -->
      <include>
        <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Table</uri>
        <pose>2 2 0 0 0 0</pose>
      </include>
      
      <include>
        <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Chair</uri>
        <pose>-2 -2 0 0 0 1.57</pose>
      </include>
    </model>
    
    <!-- Lighting -->
    <light type="point" name="point_light">
      <pose>0 0 3 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>20</range>
        <constant>0.5</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <cast_shadows>true</cast_shadows>
    </light>
    
  </world>
</sdf>
```

### Launching Gazebo with ROS 2

```python
# launch/humanoid_gazebo_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    
    # Get package share directory
    pkg_share = FindPackageShare(package='humanoid_robot').find('humanoid_robot')
    
    # World file path
    world_file_path = os.path.join(pkg_share, 'worlds', 'humanoid_env.world')
    
    # Gazebo launch
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', world_file_path],
        output='screen'
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': open(os.path.join(pkg_share, 'urdf', 'humanoid.urdf')).read(),
            'use_sim_time': True
        }]
    )
    
    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'humanoid_robot',
            '-file', os.path.join(pkg_share, 'urdf', 'humanoid.urdf'),
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )
    
    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': True}]
    )
    
    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        joint_state_publisher
    ])
```

## 4.2 Simulating Physics, Gravity, and Collisions

### Physics Engine Configuration

Gazebo uses ODE (Open Dynamics Engine) by default, but also supports Bullet, Simbody, and DART:

```xml
<!-- Physics configuration for humanoid simulation -->
<physics name="humanoid_physics" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  
  <!-- Gravity settings for Earth -->
  <gravity>0 0 -9.8066</gravity>
  
  <!-- ODE specific settings -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>
      <sor>1.3</sor>
      <use_dynamic_moi_rescaling>1</use_dynamic_moi_rescaling>
    </solver>
    
    <constraints>
      <cfm>0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
  
  <!-- Contact properties -->
  <surface>
    <contact>
      <collide_without_contact>false</collide_without_contact>
      <collide_without_contact_dist>0.001</collide_without_contact_dist>
      <max_vel>100</max_vel>
      <min_depth>0.001</min_depth>
    </contact>
    
    <friction>
      <ode>
        <mu>0.9</mu>
        <mu2>0.9</mu2>
        <fdir1>0 0 1</fdir1>
        <slip1>0.0</slip1>
        <slip2>0.0</slip2>
      </ode>
    </friction>
    
    <bounce>
      <restitution_coefficient>0.2</restitution_coefficient>
      <threshold>100000</threshold>
    </bounce>
  </surface>
</physics>
```

### Collision Detection

Gazebo provides multiple collision detection algorithms:

```xml
<!-- Collision geometry for humanoid foot -->
<collision name="left_foot_collision">
  <pose>0 0 -0.05 0 0 0</pose>
  <geometry>
    <box>
      <size>0.25 0.15 0.08</size>
    </box>
  </geometry>
  
  <!-- Contact properties -->
  <surface>
    <contact>
      <ode>
        <max_vel>0.01</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
    
    <friction>
      <ode>
        <mu>1.2</mu>  <!-- High friction for foot-ground contact -->
        <mu2>1.2</mu2>
        <slip1>0.0</slip1>
        <slip2>0.0</slip2>
        <fdir1>0 0 1</fdir1>
      </ode>
    </friction>
  </surface>
</collision>
```

### Physics Performance Tuning

```python
# Python script to tune physics parameters
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetPhysicsProperties
from gazebo_msgs.msg import ODEPhysics

class PhysicsTuner(Node):
    def __init__(self):
        super().__init__('physics_tuner')
        self.client = self.create_client(SetPhysicsProperties, '/gazebo/set_physics_properties')
        
    def set_physics_properties(self):
        req = SetPhysicsProperties.Request()
        
        # Time step settings
        req.time_step.sec = 0
        req.time_step.nanosec = 1000000  # 1ms
        
        # Max step size
        req.max_update_rate = 1000.0
        
        # Gravity
        req.gravity.x = 0.0
        req.gravity.y = 0.0
        req.gravity.z = -9.8066
        
        # ODE physics configuration
        ode_config = ODEPhysics()
        ode_config.max_step_size = 0.001
        ode_config.min_step_size = 0.0001
        ode_config.iters = 50
        ode_config.sor = 1.3
        ode_config.use_dynamic_moi_rescaling = True
        
        req.ode_config = ode_config
        
        # Call the service
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info('Physics properties updated successfully')
        else:
            self.get_logger().error('Failed to update physics properties')

def main():
    rclpy.init()
    tuner = PhysicsTuner()
    tuner.set_physics_properties()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4.3 Sensor Simulation (LiDAR, IMU, Cameras)

### Camera Sensor

```xml
<!-- RGB Camera for humanoid head -->
<sensor name="head_camera" type="camera">
  <pose>0.1 0 0.1 0 0 0</pose>
  
  <camera>
    <!-- Image properties -->
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    
    <!-- Clipping planes -->
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
    
    <!-- Distortion -->
    <distortion>
      <k1>0.0</k1>
      <k2>0.0</k2>
      <k3>0.0</k3>
      <p1>0.0</p1>
      <p2>0.0</p2>
    </distortion>
  </camera>
  
  <!-- Sensor properties -->
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
  
  <!-- ROS 2 topics -->
  <plugin name="camera_plugin" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>/humanoid</namespace>
      <remapping>~/image_raw:=/camera/image_raw</remapping>
      <remapping>~/camera_info:=/camera/camera_info</remapping>
    </ros>
    <camera_name>head_camera</camera_name>
    <frame_name>head_camera_optical_frame</frame_name>
    <hack_baseline>0.07</hack_baseline>
  </plugin>
</sensor>
```

### Depth Camera

```xml
<!-- Depth Camera for 3D perception -->
<sensor name="depth_camera" type="depth">
  <pose>0.1 0 0.1 0 0 0</pose>
  
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
  
  <!-- Depth camera plugin -->
  <plugin name="depth_camera_plugin" filename="libgazebo_ros_depth_camera.so">
    <ros>
      <namespace>/humanoid</namespace>
      <remapping>~/depth/image_raw:=/depth/image_raw</remapping>
      <remapping>~/depth/camera_info:=/depth/camera_info</remapping>
      <remapping>~/depth/points:=/depth/points</remapping>
    </ros>
    <camera_name>depth_camera</camera_name>
    <frame_name>depth_camera_optical_frame</frame_name>
  </plugin>
</sensor>
```

### LiDAR Sensor

```xml
<!-- 2D LiDAR for navigation -->
<sensor name="laser_scanner" type="ray">
  <pose>0 0 0.5 0 0 0</pose>
  
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>  <!-- -180 degrees -->
        <max_angle>3.14159</max_angle>   <!-- 180 degrees -->
      </horizontal>
    </scan>
    
    <range>
      <min>0.1</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
  
  <!-- LiDAR plugin -->
  <plugin name="laser_plugin" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <namespace>/humanoid</namespace>
      <remapping>~/out:=/scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
    <frame_name>laser_frame</frame_name>
  </plugin>
</sensor>
```

### IMU Sensor

```xml
<!-- IMU for orientation and acceleration -->
<sensor name="imu_sensor" type="imu">
  <pose>0 0 0 0 0 0</pose>
  
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <visualize>true</visualize>
  
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-3</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>8e-5</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-3</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>8e-5</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-3</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>8e-5</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  
  <!-- IMU plugin -->
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <ros>
      <namespace>/humanoid</namespace>
      <remapping>~/imu:=/imu/data</remapping>
    </ros>
    <body_name>imu_link</body_name>
    <frame_name>imu_frame</frame_name>
    <topic_name>/imu/data</topic_name>
    <service_name>/imu/calibrate</service_name>
    <angular_velocity_covariance>0.0001 0 0 0 0.0001 0 0 0 0.0001</angular_velocity_covariance>
    <linear_acceleration_covariance>0.01 0 0 0 0.01 0 0 0 0.01</linear_acceleration_covariance>
    <orientation_covariance>0.01 0 0 0 0.01 0 0 0 0.01</orientation_covariance>
  </plugin>
</sensor>
```

## 4.4 Integrating ROS 2 with Gazebo

### Gazebo ROS 2 Bridge

The Gazebo-ROS 2 bridge enables communication between Gazebo and ROS 2:

```python
# gazebo_ros_bridge.py
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates, LinkStates
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState

class GazeboBridge(Node):
    def __init__(self):
        super().__init__('gazebo_bridge')
        
        # Subscribers from Gazebo
        self.model_states_sub = self.create_subscription(
            ModelStates, '/gazebo/model_states', self.model_states_callback, 10)
        
        self.link_states_sub = self.create_subscription(
            LinkStates, '/gazebo/link_states', self.link_states_callback, 10)
        
        # Publishers for robot control
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        
        self.joint_cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10)
        
        # Robot state
        self.robot_pose = None
        self.robot_twist = None
        
    def model_states_callback(self, msg):
        """Handle model states from Gazebo"""
        try:
            robot_index = msg.name.index('humanoid_robot')
            self.robot_pose = msg.pose[robot_index]
            self.robot_twist = msg.twist[robot_index]
            
            # Log robot position
            self.get_logger().info(
                f'Robot position: x={self.robot_pose.position.x:.2f}, '
                f'y={self.robot_pose.position.y:.2f}, '
                f'z={self.robot_pose.position.z:.2f}')
        except ValueError:
            pass
    
    def link_states_callback(self, msg):
        """Handle link states for joint monitoring"""
        try:
            # Find joint positions from link states
            joint_states = JointState()
            joint_states.header.stamp = self.get_clock().now().to_msg()
            
            # Process joint states (simplified)
            for i, name in enumerate(msg.name):
                if 'joint' in name:
                    joint_states.name.append(name)
                    # Extract joint angle from orientation
                    # This is simplified - actual implementation would be more complex
                    joint_states.position.append(0.0)
            
            # Publish joint states
            self.joint_cmd_pub.publish(joint_states)
        except Exception as e:
            self.get_logger().error(f'Error processing link states: {e}')
    
    def send_velocity_command(self, linear_x, angular_z):
        """Send velocity command to robot"""
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd)

def main():
    rclpy.init()
    bridge = GazeboBridge()
    
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Joint Control with ROS 2

```python
# joint_controller.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetJointProperties, GetJointProperties
import math

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')
        
        # Joint state publisher
        self.joint_pub = self.create_publisher(
            JointState, '/joint_states', 10)
        
        # Service clients for Gazebo joint control
        self.set_joint_client = self.create_client(
            SetJointProperties, '/gazebo/set_joint_properties')
        
        self.get_joint_client = self.create_client(
            GetJointProperties, '/gazebo/get_joint_properties')
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Joint names for humanoid
        self.joint_names = [
            'left_hip_pitch', 'left_knee', 'left_ankle',
            'right_hip_pitch', 'right_knee', 'right_ankle',
            'left_shoulder_pitch', 'left_elbow',
            'right_shoulder_pitch', 'right_elbow'
        ]
        
        self.time = 0.0
        
    def control_loop(self):
        """Main control loop for joint movements"""
        self.time += 0.1
        
        # Create joint state message
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_names
        
        # Generate walking pattern
        positions = []
        for i, name in enumerate(self.joint_names):
            if 'hip' in name:
                # Hip movement for walking
                angle = 0.3 * math.sin(self.time + (0 if 'left' in name else math.pi))
                positions.append(angle)
            elif 'knee' in name:
                # Knee movement
                angle = -0.6 * abs(math.sin(self.time + (0 if 'left' in name else math.pi)))
                positions.append(angle)
            elif 'ankle' in name:
                # Ankle stabilization
                angle = 0.1 * math.sin(self.time * 2)
                positions.append(angle)
            elif 'shoulder' in name:
                # Arm swing
                angle = 0.2 * math.sin(self.time + (math.pi if 'left' in name else 0))
                positions.append(angle)
            elif 'elbow' in name:
                # Elbow bend
                angle = 0.3 * abs(math.sin(self.time + (math.pi if 'left' in name else 0)))
                positions.append(angle)
        
        joint_state.position = positions
        
        # Publish joint states
        self.joint_pub.publish(joint_state)
        
        # Also send to Gazebo for simulation
        self.send_to_gazebo(joint_state)
    
    def send_to_gazebo(self, joint_state):
        """Send joint commands to Gazebo"""
        for i, name in enumerate(joint_state.name):
            req = SetJointProperties.Request()
            req.joint_name = name
            req.ode_joint.position = joint_state.position[i]
            
            future = self.set_joint_client.call_async(req)
            # In production, you'd want to handle the future properly

def main():
    rclpy.init()
    controller = JointController()
    
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

### Launch File for Complete System

```python
# launch/humanoid_sim_complete.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    
    # Get package paths
    pkg_share = FindPackageShare(package='humanoid_robot').find('humanoid_robot')
    gazebo_ros_pkg_share = FindPackageShare(package='gazebo_ros').find('gazebo_ros')
    
    # World file
    world_file_path = os.path.join(pkg_share, 'worlds', 'humanoid_env.world')
    
    # Start Gazebo
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', 
             '-s', 'libgazebo_ros_init.so', world_file_path],
        output='screen',
        additional_env={'GAZEBO_MODEL_PATH': os.path.join(pkg_share, 'models')}
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': open(os.path.join(pkg_share, 'urdf', 'humanoid.urdf')).read(),
            'use_sim_time': True
        }]
    )
    
    # Spawn robot
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'humanoid_robot',
            '-file', os.path.join(pkg_share, 'urdf', 'humanoid.urdf'),
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )
    
    # Joint controller
    joint_controller = Node(
        package='humanoid_robot',
        executable='joint_controller',
        name='joint_controller',
        parameters=[{'use_sim_time': True}]
    )
    
    # Gazebo bridge
    gazebo_bridge = Node(
        package='humanoid_robot',
        executable='gazebo_bridge',
        name='gazebo_bridge',
        parameters=[{'use_sim_time': True}]
    )
    
    # RViz for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'humanoid_sim.rviz')],
        parameters=[{'use_sim_time': True}]
    )
    
    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        joint_controller,
        gazebo_bridge,
        rviz
    ])
```

---

**Next Chapter**: In Chapter 5, we'll explore NVIDIA Isaac Sim for high-fidelity simulation and photorealistic environments.

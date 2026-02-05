---
title: "Chapter 2: Fundamentals of Robot Operating System (ROS 2)"
sidebar_position: 2
---

# Chapter 2: Fundamentals of Robot Operating System (ROS 2)

## 2.1 ROS 2 Architecture & Core Concepts

### What is ROS 2?

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms.

### Key Improvements over ROS 1

- **Real-time Capabilities**: Support for real-time systems
- **Security**: Built-in security features for production deployments
- **Multi-Platform**: Windows, macOS, and Linux support
- **Quality of Service (QoS)**: Fine-grained control over data delivery
- **Better Networking**: DDS-based communication middleware

### Architecture Overview

ROS 2 follows a distributed architecture where different components communicate through a publish-subscribe model:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Node A    │    │   Node B    │    │   Node C    │
│  (Sensor)   │◄──►│ (Processor) │◄──►│ (Actuator)  │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌─────────────┐
                    │   DDS/RTPS  │
                    │  Middleware │
                    └─────────────┘
```

## 2.2 Nodes, Topics, Services, and Actions

### Nodes

Nodes are the fundamental building blocks of ROS 2. Each node is a process that performs computation. In the context of humanoid robotics:

```python
# Example: Sensor node for a humanoid robot
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointSensorNode(Node):
    def __init__(self):
        super().__init__('joint_sensor_node')
        self.publisher_ = self.create_publisher(
            JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_states)
    
    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['left_hip', 'left_knee', 'left_ankle']
        msg.position = [0.5, -0.8, 0.2]
        self.publisher_.publish(msg)
```

### Topics

Topics are named buses over which nodes exchange messages. For humanoid robots, common topics include:

- `/joint_states`: Current positions of all joints
- `/cmd_vel`: Velocity commands for locomotion
- `/camera/image_raw`: Camera sensor data
- `/imu/data`: Inertial measurement unit data
- `/tf`: Transform tree for coordinate frames

### Services

Services provide request-response communication:

```python
# Example: Service to reset humanoid posture
from example_interfaces.srv import AddTwoInts

class PostureService(Node):
    def __init__(self):
        super().__init__('posture_service')
        self.srv = self.create_service(
            AddTwoInts, 'reset_posture', self.reset_posture_callback)
    
    def reset_posture_callback(self, request, response):
        # Logic to reset humanoid to standing position
        response.sum = request.a + request.b
        return response
```

### Actions

Actions are for long-running tasks that provide feedback:

```python
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped

class WalkAction(Node):
    def __init__(self):
        super().__init__('walk_action_server')
        self.action_server = ActionServer(
            self, WalkToGoal, 'walk_to_pose', self.execute_callback)
    
    async def execute_callback(self, goal_handle):
        feedback_msg = WalkToGoal.Feedback()
        
        # Implement walking logic with feedback
        for i in range(100):
            feedback_msg.progress = i / 100.0
            goal_handle.publish_feedback(feedback_msg)
            await asyncio.sleep(0.1)
        
        goal_handle.succeed()
        result = WalkToGoal.Result()
        result.success = True
        return result
```

## 2.3 Writing Your First ROS 2 Package in Python

### Creating a Package

```bash
# Navigate to your ROS 2 workspace
cd ~/ros2_ws/src

# Create a new package
ros2 pkg create --build-type ament_python humanoid_robot --dependencies rclpy std_msgs
```

### Package Structure

```
humanoid_robot/
├── package.xml
├── setup.py
├── resource/
│   └── humanoid_robot
├── humanoid_robot/
│   ├── __init__.py
│   └── humanoid_robot_node.py
└── test/
    ├── test_copyright.py
    ├── test_flake8.py
    └── test_pep257.py
```

### Complete Example: Humanoid Control Node

```python
#!/usr/bin/env python3
# humanoid_robot/humanoid_robot/humanoid_robot_node.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState
import math

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10)
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Robot state
        self.joint_positions = {}
        self.walking_phase = 0.0
        
        self.get_logger().info('Humanoid Controller Node Started')
    
    def joint_state_callback(self, msg):
        """Update joint positions from feedback"""
        for i, name in enumerate(msg.name):
            self.joint_positions[name] = msg.position[i]
    
    def control_loop(self):
        """Main control loop for humanoid robot"""
        # Simple walking pattern
        self.walking_phase += 0.1
        
        # Generate walking commands
        cmd = Twist()
        cmd.linear.x = 0.5 * math.sin(self.walking_phase)
        cmd.angular.z = 0.1 * math.cos(self.walking_phase * 2)
        
        self.cmd_vel_pub.publish(cmd)
        
        # Generate joint commands for walking
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.name = ['left_hip', 'left_knee', 'right_hip', 'right_knee']
        
        # Walking gait pattern
        t = self.walking_phase
        joint_cmd.position = [
            0.3 * math.sin(t),           # left_hip
            -0.6 * abs(math.sin(t)),     # left_knee
            0.3 * math.sin(t + math.pi), # right_hip
            -0.6 * abs(math.sin(t + math.pi))  # right_knee
        ]
        
        self.joint_cmd_pub.publish(joint_cmd)

def main(args=None):
    rclpy.init(args=args)
    humanoid_controller = HumanoidController()
    
    try:
        rclpy.spin(humanoid_controller)
    except KeyboardInterrupt:
        pass
    finally:
        humanoid_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Building and Running

```bash
# Build the package
cd ~/ros2_ws
colcon build --packages-select humanoid_robot

# Source the workspace
source install/setup.bash

# Run the node
ros2 run humanoid_robot humanoid_robot_node
```

## 2.4 Launch Files and Parameter Management

### Launch Files

Launch files allow you to start multiple nodes and configure them:

```python
# humanoid_robot/launch/humanoid_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    # Sensor node
    sensor_node = Node(
        package='humanoid_robot',
        executable='sensor_node',
        name='sensor_node',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    # Controller node
    controller_node = Node(
        package='humanoid_robot',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'walking_speed': 0.5,
            'step_height': 0.1
        }]
    )
    
    return LaunchDescription([
        use_sim_time,
        sensor_node,
        controller_node
    ])
```

### Parameter Management

Parameters allow you to configure node behavior without changing code:

```python
# Using parameters in a node
class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        
        # Declare parameters with defaults
        self.declare_parameter('walking_speed', 0.5)
        self.declare_parameter('step_height', 0.1)
        self.declare_parameter('arm_swing', True)
        
        # Get parameter values
        self.walking_speed = self.get_parameter('walking_speed').value
        self.step_height = self.get_parameter('step_height').value
        self.arm_swing = self.get_parameter('arm_swing').value
        
        # Parameter callback for dynamic updates
        self.add_on_set_parameters_callback(self.parameters_callback)
    
    def parameters_callback(self, params):
        """Handle parameter updates"""
        for param in params:
            if param.name == 'walking_speed':
                self.walking_speed = param.value
                self.get_logger().info(f'Updated walking speed: {self.walking_speed}')
        
        return SetParametersResult(successful=True)
```

### YAML Parameter Files

```yaml
# config/humanoid_params.yaml
humanoid_controller:
  ros__parameters:
    walking_speed: 0.8
    step_height: 0.15
    arm_swing: true
    balance_gain: 1.2
    posture_height: 1.7

sensor_node:
  ros__parameters:
    update_rate: 100.0
    noise_filter: true
    calibration_offset: 0.01
```

## 2.5 ROS 2 Middleware for Humanoid Control

### Quality of Service (QoS)

QoS policies control how data is exchanged between nodes:

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# Different QoS profiles for different use cases
# For sensor data - high reliability, keep last
sensor_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=10
)

# For control commands - best effort, keep last
control_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=1
)

# Apply QoS to publisher/subscriber
self.joint_pub = self.create_publisher(
    JointState, '/joint_commands', qos_profile=control_qos)
```

### Real-Time Considerations

For humanoid robot control, real-time performance is critical:

```python
# Real-time capable node setup
class RealTimeController(Node):
    def __init__(self):
        super().__init__('realtime_controller')
        
        # Set real-time scheduler priority
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(80))
        
        # High-frequency timer for control loop
        self.timer = self.create_timer(
            0.001,  # 1kHz control loop
            self.rt_control_loop,
            callback_group=ReentrantCallbackGroup()
        )
        
        # Pre-allocate messages to avoid allocation during runtime
        self.joint_cmd = JointState()
        self.joint_cmd.name = ['hip', 'knee', 'ankle']
        self.joint_cmd.position = [0.0, 0.0, 0.0]
    
    def rt_control_loop(self):
        """Real-time control loop - avoid allocations here"""
        # Update joint commands
        self.joint_cmd.header.stamp = self.get_clock().now().to_msg()
        
        # Control logic (pre-computed where possible)
        for i in range(len(self.joint_cmd.position)):
            self.joint_cmd.position[i] = self.compute_joint_command(i)
        
        self.joint_cmd_pub.publish(self.joint_cmd)
```

### Multi-Robot Coordination

ROS 2 supports multiple robots in the same system:

```python
# Namespace-based multi-robot system
class MultiRobotManager(Node):
    def __init__(self):
        super().__init__('multi_robot_manager')
        
        # Create controllers for multiple humanoids
        self.robots = {}
        for robot_id in ['robot1', 'robot2', 'robot3']:
            # Namespaced nodes
            controller = HumanoidController(namespace=robot_id)
            self.robots[robot_id] = controller
    
    def coordinate_robots(self):
        """Coordinate multiple humanoid robots"""
        # Synchronized walking patterns
        for i, (robot_id, controller) in enumerate(self.robots.items()):
            phase_offset = i * 2 * math.pi / len(self.robots)
            controller.set_walking_phase(phase_offset)
```

---

**Next Chapter**: In Chapter 3, we'll explore how to model humanoid robots using URDF and SDF formats, creating accurate digital representations for simulation.

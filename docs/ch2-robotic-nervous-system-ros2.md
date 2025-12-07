---
id: ch2-robotic-nervous-system-ros2
title: "Chapter 2 – Module 1: The Robotic Nervous System (ROS 2)"
---

## Spec-Kit Plus Block

- **Learning Objectives**
  - Model a humanoid robot as a distributed ROS 2 computation graph.
  - Design topics, services, and actions for perception, control, and supervision.
  - Implement publishers, subscribers, and timers in Python using `rclpy`.
  - Inspect, debug, and profile ROS 2 graphs with CLI tools.

- **Required Skills**
  - Completion of Chapter 1.
  - Python programming (functions, classes, virtual environments).
  - Basic Linux shell usage.

- **System Components**
  - ROS 2 core (DDS, discovery, QoS layer).
  - Humanoid sensor nodes: cameras, IMUs, joint encoders, microphones.
  - Control nodes: whole-body controller, balance controller, teleoperation interface.
  - Supervisory and monitoring nodes: safety monitor, logger, visualization.

- **Inputs & Outputs**
  - Inputs: sensor topics (camera, IMU, joints, audio), high-level commands.
  - Outputs: joint, base, and gripper command topics; diagnostic topics; logs.
  - Internal: services/actions for configuration and long-running behaviors.

- **Tools & Frameworks**
  - ROS 2 Humble (or later), `rclpy` for Python nodes.
  - `ros2` CLI: `ros2 node`, `ros2 topic`, `ros2 service`, `ros2 action`.
  - `colcon` build system for workspaces.
  - Optional: `rqt_graph`, `rqt_console`.

- **Performance Constraints**
  - 100 Hz joint state updates; 200–1000 Hz for low-level loops proxied into ROS 2.
  - Deterministic, non-blocking callbacks; bounded callback execution time.
  - Robust behavior under transient network failures and node restarts.

---

## 2.1 ROS 2 as the Robotic Nervous System

In biological organisms, the **nervous system** connects sensors, processing centers, and muscles. In humanoid robots, **ROS 2** plays an analogous role:

- It connects distributed processes (nodes) across CPUs and machines.
- It transports sensor streams and control commands as typed messages.
- It coordinates high-level tasks (actions) and configuration (services, parameters).

For Physical AI, this logical nervous system must be:

- **Modular** – new capabilities are provided by new nodes, not monolithic binaries.
- **Real-time aware** – data must arrive in time for control loops to act.
- **Resilient** – individual nodes can fail and restart without collapsing the system.

You will design humanoid behaviors by **shaping the ROS 2 graph**: choosing nodes, topics, their directions, and their QoS (Quality of Service) policies.

### 2.1.1 Diagram Description: Humanoid ROS 2 Graph

Imagine a diagram with three columns:

- **Left column – Perception nodes**
  - `head_camera_driver` → publishes `/head_camera/image_raw`.
  - `imu_torso` → publishes `/torso_imu`.
  - `joint_state_publisher` → publishes `/joint_states`.

- **Middle column – Estimation and cognition**
  - `state_estimator` subscribes to `/torso_imu` and `/joint_states`, publishes `/base_pose`.
  - `vla_policy` subscribes to `/head_camera/image_raw` and a language command topic `/task_text`, publishes `/motion_goals`.

- **Right column – Control and actuation**
  - `whole_body_controller` subscribes to `/motion_goals` and `/base_pose`, publishes `/joint_trajectory`.
  - `low_level_driver` subscribes to `/joint_trajectory` and sends CAN/EtherCAT commands to motors.

Arrows representing topics flow left to right. The union of nodes and topics is the **ROS 2 computation graph** of the humanoid.

---

## 2.2 Core ROS 2 Abstractions

### 2.2.1 Nodes

Nodes are long-lived processes that perform computation and interact with each other. In ROS 2, a typical humanoid might run dozens of nodes across several machines:

- Sensor nodes attached to hardware.
- Estimation and control nodes on a real-time controller.
- Perception and learning nodes on a GPU workstation.

In `rclpy`, every executable node is usually created by subclassing `Node`:

```python
import rclpy
from rclpy.node import Node


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__('example_node')
        self.get_logger().info('ExampleNode started')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExampleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2.2.2 Topics

Topics are **named, typed streams** of messages. They are used for continuous data such as:

- `/joint_states` (`sensor_msgs/JointState`).
- `/cmd_vel` (`geometry_msgs/Twist`).
- `/head_camera/image_raw` (`sensor_msgs/Image`).

Nodes publish and subscribe to topics without needing to know about each other directly. The underlying DDS layer handles discovery and transport.

### 2.2.3 Services and Actions

- **Services** implement synchronous request–response interactions.
  - Example: `/reset_world` service in simulation.
  - Example: `/set_gait_parameters` service for dynamic reconfiguration.

- **Actions** support long-running goals with feedback.
  - Example: `/walk_to_pose` for a humanoid locomotion action.
  - Example: `/reach_object` for manipulation tasks.

Actions are essential for Physical AI: they represent temporally extended behaviors that must be monitored and possibly preempted.

### 2.2.4 Parameters

ROS 2 parameters are configurable values stored in nodes. They allow you to change behavior without recompiling:

- Gains for balance controllers.
- Thresholds for perception confidence.
- Paths to model files.

Parameters can be declared, read, and set via `rclpy` or the CLI.

---

## 2.3 Building the Humanoid ROS 2 Workspace

You will maintain a persistent ROS 2 workspace for this textbook.

```bash
# 1. Create the workspace
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws

# 2. Source ROS 2 (adapt path to your distro)
source /opt/ros/humble/setup.bash

# 3. Create a Python package for this module
cd src
ros2 pkg create --build-type ament_python humanoid_nervous_system

# 4. Build and source the overlay
cd ..
colcon build
source install/setup.bash
```

You will add multiple nodes to `humanoid_nervous_system` over time.

---

## 2.4 Implementing a Sensor Publisher Node

We begin with a simulated torso IMU that publishes a 1D body pitch angle on `/body_pitch`.

```python
# humanoid_nervous_system/humanoid_nervous_system/imu_publisher.py
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class SimulatedImuPublisher(Node):
    def __init__(self) -> None:
        super().__init__('simulated_imu_publisher')
        self.publisher_ = self.create_publisher(Float64, '/body_pitch', 10)
        self.timer_period = 0.01  # 100 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.t: float = 0.0
        self.get_logger().info('SimulatedImuPublisher started.')

    def timer_callback(self) -> None:
        # Simulate a gentle oscillation around upright
        angle = 0.15 * math.sin(0.5 * self.t)
        msg = Float64()
        msg.data = angle
        self.publisher_.publish(msg)
        self.t += self.timer_period


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = SimulatedImuPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2.4.1 Entry Point Registration

In `setup.py` (generated by `ros2 pkg create`), add an entry for this node:

```python
entry_points={
    'console_scripts': [
        'simulated_imu_publisher = humanoid_nervous_system.imu_publisher:main',
    ],
},
```

Rebuild and run:

```bash
cd ~/physical_ai_ws
colcon build
source install/setup.bash

ros2 run humanoid_nervous_system simulated_imu_publisher
```

In another terminal:

```bash
source ~/physical_ai_ws/install/setup.bash
ros2 topic echo /body_pitch
```

You should see a stream of floating-point angles.

---

## 2.5 Implementing a Balance Controller Node

Next, we implement a PD controller that subscribes to `/body_pitch` and publishes a virtual ankle torque on `/ankle_torque`.

```python
# humanoid_nervous_system/humanoid_nervous_system/balance_controller.py
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class BalanceController(Node):
    def __init__(self) -> None:
        super().__init__('balance_controller')

        self.declare_parameter('kp', 20.0)
        self.declare_parameter('kd', 2.0)

        self.kp = float(self.get_parameter('kp').value)
        self.kd = float(self.get_parameter('kd').value)

        self.prev_error = 0.0
        self.dt = 0.01

        self.subscription = self.create_subscription(
            Float64,
            '/body_pitch',
            self.body_pitch_callback,
            10,
        )
        self.publisher_ = self.create_publisher(Float64, '/ankle_torque', 10)

    def body_pitch_callback(self, msg: Float64) -> None:
        angle = msg.data
        error = -angle  # we want angle -> 0
        d_error = (error - self.prev_error) / self.dt
        self.prev_error = error

        torque = self.kp * error + self.kd * d_error

        cmd = Float64()
        cmd.data = torque
        self.publisher_.publish(cmd)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = BalanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Add an entry point:

```python
entry_points={
    'console_scripts': [
        'simulated_imu_publisher = humanoid_nervous_system.imu_publisher:main',
        'balance_controller = humanoid_nervous_system.balance_controller:main',
    ],
},
```

You can now run both nodes and observe the full **sensor → controller** pipeline:

```bash
ros2 run humanoid_nervous_system simulated_imu_publisher
```

In another terminal:

```bash
ros2 run humanoid_nervous_system balance_controller
```

In a third terminal:

```bash
ros2 topic echo /ankle_torque
```

---

## 2.6 Inspecting and Debugging the ROS 2 Graph

With the nodes running, inspect the graph:

```bash
ros2 node list
ros2 topic list
ros2 topic info /body_pitch
ros2 topic info /ankle_torque
```

You should see `simulated_imu_publisher` and `balance_controller` listed as nodes, and the two topics with publisher/subscriber counts.

### 2.6.1 Diagram Description: Sensor–Controller Subgraph

Picture a small graph with two boxes:

- Left: `simulated_imu_publisher` with an arrow labeled `/body_pitch` pointing right.
- Right: `balance_controller` with an arrow labeled `/ankle_torque` pointing out of it.

This subgraph is the minimal nervous system for 1D balance.

### 2.6.2 Recording Data with rosbag

Use rosbag to record the interaction for analysis:

```bash
ros2 bag record /body_pitch /ankle_torque -o balance_test
```

After some time, stop recording with `Ctrl+C` and inspect the bag with external tools or Python scripts.

---

## 2.7 QoS and Reliability in Humanoid Systems

Quality of Service (QoS) policies influence latency, reliability, and persistence. For humanoid robots:

- **Best effort vs reliable**
  - Joint commands: often `reliable` (loss is dangerous).
  - High-rate camera streams: can be `best_effort`.

- **History and depth**
  - Sensors: `keep_last` with small depth to limit memory.
  - Parameters or configuration states: may need `transient_local` durability.

Example of specifying QoS in `rclpy`:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

qos = QoSProfile(depth=10)
qos.reliability = ReliabilityPolicy.RELIABLE

self.publisher_ = self.create_publisher(Float64, '/ankle_torque', qos)
```

Choosing QoS is an engineering trade-off between robustness, latency, and network load.

---

## 2.8 From Toy Example to Humanoid Integration

The same design patterns scale to full humanoids:

- Replace the simulated IMU with a real IMU driver node from hardware vendors.
- Replace the PD controller with a whole-body controller computing commands for dozens of joints.
- Replace single float messages with `sensor_msgs/Imu`, `sensor_msgs/JointState`, and trajectory messages.

The essential structure remains:

> Sensors → Estimator → Controller → Actuators

Each block is one or more ROS 2 nodes, with clear inputs and outputs.

---

## 2.9 Summary

In this chapter you:

- Interpreted ROS 2 as the **robotic nervous system** for a humanoid.
- Reviewed ROS 2’s core abstractions: nodes, topics, services, actions, parameters.
- Created a ROS 2 workspace and a Python package dedicated to the humanoid nervous system.
- Implemented a simulated IMU publisher and a balance controller node in Python.
- Ran both nodes and inspected the resulting ROS 2 graph with CLI tools.
- Considered QoS design for reliable and timely communication.

These patterns are fundamental. Every subsequent chapter will re-use and extend them for perception, simulation, AI policies, and full humanoid control.

---

## 2.10 Exercises

1. **Graph Sketching**  
   Draw a ROS 2 computation graph for a humanoid standing balance scenario with at least six nodes: three sensors, one estimator, one controller, and one low-level driver node. Label all topics.

2. **Parameter Experiments**  
   Modify the `BalanceController` node to read `kp` and `kd` parameters from the command line. Run several experiments with different gains and describe the effects on the torque command sequence.

3. **QoS Variants**  
   Create two versions of the IMU publisher: one with reliable QoS and one with best-effort QoS. Intentionally introduce packet loss (e.g., using network tools or throttling) and document the observed behavior.

4. **Service Design**  
   Design a ROS 2 service interface to enable or disable the balance controller remotely. Specify the service name, request fields, and response fields, and explain how it integrates with the existing nodes.

5. **Action Interface Proposal**  
   Propose an action interface for a `WalkToPose` behavior. Describe the goal, feedback, and result messages, and how this action would be used by higher-level planning nodes.

6. **Monitoring Node**  
   Implement a monitoring node that subscribes to `/body_pitch` and `/ankle_torque`, checks for values exceeding predefined safety thresholds, and logs warnings.

7. **Bag File Analysis**  
   Record `/body_pitch` and `/ankle_torque` with `ros2 bag record`. Write a small Python script (not necessarily a ROS node) that reads the bag, reconstructs the time series, and plots angle vs. torque.

8. **Namespace Design**  
   Suppose you have two humanoids in the same lab. Propose a namespace scheme (e.g., `/humanoid1`, `/humanoid2`) for topics, and describe how you would configure nodes to avoid conflicts.

9. **Robust Shutdown**  
   Extend the `BalanceController` node to perform an orderly shutdown: when it receives a termination signal, it publishes a final zero-torque command and logs a structured shutdown message.

10. **Failure Mode Analysis**  
   Enumerate at least five ways the ROS 2 nervous system of a humanoid can fail (e.g., node crash, topic misconfiguration, QoS mismatch). For each, propose a detection and mitigation strategy.

---

## 2.11 Mini Project – Humanoid Balance Nervous System Prototype

**Goal:** Build a minimal but realistic ROS 2 nervous system for 1D humanoid balance using simulation.

**Project Tasks:**

1. **System Specification**  
   Write a one-page specification of your balance nervous system:
   - Nodes and their responsibilities.
   - Topics, message types, and directions.
   - Desired control frequency and latency targets.

2. **Node Implementation**  
   Implement at least three nodes in the `humanoid_nervous_system` package:
   - `simulated_imu_publisher` (or equivalent sensor node).
   - `balance_controller` implementing PD control.
   - `safety_monitor` that observes topics and logs anomalies.

3. **Simulation Integration (Conceptual or Actual)**  
   Either:
   - Connect your nodes to a simple inverted pendulum model in Gazebo (if available), or
   - Build a standalone Python script that publishes synthetic `/body_pitch` data consistent with a pendulum model and subscribes to `/ankle_torque` to close the loop conceptually.

4. **Experiment Protocol**  
   Design an experiment protocol:
   - Step responses by initializing the body with different initial angles.
   - Disturbances by injecting noisy or delayed sensor data.
   - Logging strategy using rosbag.

5. **Evaluation and Report**  
   Analyze the captured data and document:
   - Stability and settling time for different gains.
   - Sensitivity to delays or message loss.
   - Lessons learned for scaling to full humanoid balance and walking.

Deliver as a small ROS 2 package plus a markdown report. This project forms the conceptual and infrastructural foundation for later chapters on full humanoid control and AI-augmented policies.


---
id: ch1-introduction-to-physical-ai
title: "Chapter 1 – Introduction to Physical AI"
slug: /ch1-introduction-to-physical-ai
---

## Spec-Kit Plus Block

- **Learning Objectives**
  - Define Physical AI and distinguish it from purely digital AI.
  - Explain the layered architecture of a modern intelligent robot.
  - Understand the role of ROS 2, Gazebo, Unity, and NVIDIA Isaac in the stack.
  - Trace an end-to-end loop from perception to action on a humanoid robot.

- **Required Skills**
  - Basic Python.
  - Basic linear algebra and probability.
  - Familiarity with Linux command line.

- **System Components**
  - Humanoid base platform (simulated or real).
  - Onboard computer (x86 or Jetson-class ARM).
  - Sensors: cameras, IMU, joint encoders, microphones.
  - Actuators: servo motors, grippers.
  - Middleware: ROS 2.

- **Inputs & Outputs**
  - Inputs: images, audio, joint states, IMU, user commands.
  - Outputs: joint commands, base velocity commands, speech, logs.

- **Tools & Frameworks**
  - ROS 2 Humble or later, `rclpy`.
  - Gazebo, Unity (high-level in this chapter).
  - Python 3.10+.

- **Performance Constraints**
  - End-to-end perception–action latency under 200 ms for many tasks.
  - 50–200 Hz control loops for low-level joints.

## 1. Overview

This chapter introduces the concept of **Physical AI** and positions humanoid robotics as a concrete, high-impact embodiment of AI in the real world. You will see how sensors, actuators, control, and high-level AI reasoning come together through ROS 2 and modern simulation tools.

### 1.1 A Minimal Perception–Action Loop in ROS 2

```python
# docs/snippets/ch1_minimal_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPerceptionAction(Node):
    def __init__(self) -> None:
        super().__init__('minimal_perception_action')
        self.subscription = self.create_subscription(
            String,
            'perception/text',
            self.perception_callback,
            10,
        )
        self.publisher_ = self.create_publisher(String, 'action/text_command', 10)

    def perception_callback(self, msg: String) -> None:
        self.get_logger().info(f'Received perception: {msg.data}')
        action = String()
        action.data = f"ECHO_ACTION: {msg.data}"
        self.publisher_.publish(action)
        self.get_logger().info(f'Sent action: {action.data}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MinimalPerceptionAction()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Run this node with:

```bash
ros2 run your_package minimal_perception_action
```

…then publish test input:

```bash
ros2 topic pub /perception/text std_msgs/String "data: 'hello humanoid'" -1
```

### 1.2 Simulation First: A Simple Gazebo World

A typical workflow for Physical AI experiments starts in simulation:

1. **Launch Gazebo with an empty world**.
2. **Spawn a simple robot model** (e.g., a differential drive base) as a precursor to humanoid complexity.
3. **Connect ROS 2** to stream sensor data and send velocity commands.

Example launch command stub:

```bash
ros2 launch gazebo_ros empty_world.launch.py
```

You will later replace the simple robot with a full humanoid URDF/SDF description.

## 2. Summary

This chapter defined Physical AI, introduced the basic layered architecture of an intelligent robot, and illustrated a minimal ROS 2 perception–action node and a basic simulation-first workflow. In the rest of the book, you will deepen each layer: the robotic nervous system (ROS 2), digital twin construction, AI brain integration, VLA pipelines, humanoid embodiment, conversational interfaces, hardware labs, and a capstone autonomous humanoid.

## 3. Exercises

1. **Concept Map** – Draw a diagram of the layers of a Physical AI system from sensors to high-level reasoning.
2. **Latency Budget** – Propose a latency budget for an interactive humanoid (perception to action) and justify each component.
3. **Failure Modes** – List five distinct failure modes unique to Physical AI that do not occur in purely digital AI systems.
4. **Topic Design** – Propose a minimal set of ROS 2 topics for a simple humanoid head (camera, microphone, pan-tilt joints).
5. **Safety Considerations** – Describe how you would enforce safety when executing high-level AI-generated commands on real hardware.
6. **Simulation vs Reality** – Compare three advantages and three limitations of starting in Gazebo before moving to a real humanoid.
7. **Embodiment and Learning** – Explain how embodiment changes the learning problem compared to image classification.
8. **Time Scales** – Identify and categorize at least three different time scales in a humanoid control system.
9. **Logging Strategy** – Design a logging and data collection strategy for early Physical AI experiments.
10. **Ethical Scenario** – Describe an ethical dilemma involving a humanoid assistant in a hospital environment and outline mitigation measures.

## 4. Mini Project

**Title:** Minimal Perception–Action Text Robot

**Goal:** Build a ROS 2 package that implements the `MinimalPerceptionAction` node, plus a simple command-line tool that sends different text "perceptions" and logs the resulting actions. Extend the node so that specific keywords (e.g., "forward", "backward", "stop") are translated into abstract humanoid motion commands on a separate topic (e.g., `/humanoid/motion_cmd`). Document your architecture, topics, and how this simple example would scale to richer Physical AI behaviors.

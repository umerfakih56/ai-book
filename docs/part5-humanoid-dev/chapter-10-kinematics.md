---
title: "Chapter 10: Humanoid Kinematics & Dynamics"
sidebar_position: 10
---

# Chapter 10: Humanoid Kinematics & Dynamics

## Introduction

Humanoid robots require sophisticated kinematic and dynamic models to achieve stable and efficient locomotion. This chapter covers the mathematical foundations and practical implementation of humanoid robot kinematics and dynamics.

## 10.1 Forward Kinematics

### 10.1.1 Joint Coordinate Systems

```python
import numpy as np
from scipy.spatial.transform import Rotation

class HumanoidKinematics:
    def __init__(self):
        # Joint configuration for humanoid robot
        self.joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee_pitch', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee_pitch', 'right_ankle_pitch', 'right_ankle_roll',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
            'left_elbow_pitch', 'left_wrist_pitch', 'left_wrist_roll',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow_pitch', 'right_wrist_pitch', 'right_wrist_roll',
            'neck_pitch', 'neck_yaw', 'neck_roll'
        ]
        
        # DH parameters for each joint
        self.dh_parameters = self._initialize_dh_parameters()
        
    def _initialize_dh_parameters(self):
        """Initialize Denavit-Hartenberg parameters"""
        return {
            'left_hip_yaw': {'a': 0, 'alpha': 0, 'd': 0.1, 'offset': 0},
            'left_hip_roll': {'a': 0, 'alpha': np.pi/2, 'd': 0, 'offset': 0},
            'left_hip_pitch': {'a': 0.1, 'alpha': 0, 'd': 0, 'offset': 0},
            # ... more joints
        }
    
    def forward_kinematics(self, joint_angles):
        """Calculate forward kinematics for all joints"""
        positions = {}
        orientations = {}
        
        for joint_name in self.joint_names:
            if joint_name in joint_angles:
                pos, ori = self._compute_joint_transform(joint_name, joint_angles[joint_name])
                positions[joint_name] = pos
                orientations[joint_name] = ori
                
        return positions, orientations
    
    def _compute_joint_transform(self, joint_name, angle):
        """Compute transformation matrix for a joint"""
        dh = self.dh_parameters.get(joint_name)
        if not dh:
            return np.zeros(3), np.eye(3)
        
        # DH transformation
        theta = angle + dh['offset']
        a = dh['a']
        d = dh['d']
        alpha = dh['alpha']
        
        # Transformation matrix
        T = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        
        position = T[:3, 3]
        orientation = T[:3, :3]
        
        return position, orientation
```

### 10.1.2 End-Effector Positioning

```python
class EndEffectorKinematics:
    def __init__(self, kinematics_solver):
        self.kinematics = kinematics_solver
        
    def get_foot_positions(self, joint_angles):
        """Get left and right foot positions"""
        positions, _ = self.kinematics.forward_kinematics(joint_angles)
        
        left_foot = positions.get('left_ankle_roll', np.zeros(3))
        right_foot = positions.get('right_ankle_roll', np.zeros(3))
        
        return left_foot, right_foot
    
    def get_hand_positions(self, joint_angles):
        """Get left and right hand positions"""
        positions, _ = self.kinematics.forward_kinematics(joint_angles)
        
        left_hand = positions.get('left_wrist_roll', np.zeros(3))
        right_hand = positions.get('right_wrist_roll', np.zeros(3))
        
        return left_hand, right_hand
    
    def get_center_of_mass(self, joint_angles, link_masses):
        """Calculate center of mass of the robot"""
        positions, _ = self.kinematics.forward_kinematics(joint_angles)
        
        total_mass = sum(link_masses.values())
        com = np.zeros(3)
        
        for link_name, mass in link_masses.items():
            if link_name in positions:
                com += mass * positions[link_name]
                
        return com / total_mass
```

## 10.2 Inverse Kinematics

### 10.2.1 Jacobian-Based IK Solver

```python
class InverseKinematics:
    def __init__(self, kinematics_solver):
        self.kinematics = kinematics_solver
        self.joint_limits = self._initialize_joint_limits()
        
    def _initialize_joint_limits(self):
        """Initialize joint limits for humanoid robot"""
        return {
            'left_hip_yaw': (-np.pi, np.pi),
            'left_hip_roll': (-np.pi/4, np.pi/4),
            'left_hip_pitch': (-np.pi/2, np.pi/2),
            'left_knee_pitch': (0, np.pi),
            'left_ankle_pitch': (-np.pi/2, np.pi/2),
            'left_ankle_roll': (-np.pi/4, np.pi/4),
            # ... similar for right leg and arms
        }
    
    def inverse_kinematics(self, target_position, target_orientation, 
                          initial_angles, max_iterations=100):
        """Solve inverse kinematics using Jacobian method"""
        joint_angles = initial_angles.copy()
        
        for iteration in range(max_iterations):
            # Forward kinematics
            current_pos, current_ori = self.kinematics.forward_kinematics(joint_angles)
            
            # Position error
            pos_error = target_position - current_pos
            ori_error = self._orientation_error(target_orientation, current_ori)
            
            error = np.concatenate([pos_error, ori_error])
            
            if np.linalg.norm(error) < 1e-6:
                break
            
            # Compute Jacobian
            jacobian = self._compute_jacobian(joint_angles)
            
            # Damped least squares solution
            damping = 0.01
            delta_angles = np.linalg.solve(
                jacobian.T @ jacobian + damping * np.eye(jacobian.shape[1]),
                jacobian.T @ error
            )
            
            # Update joint angles
            joint_angles += delta_angles
            
            # Apply joint limits
            joint_angles = self._apply_joint_limits(joint_angles)
            
        return joint_angles
    
    def _compute_jacobian(self, joint_angles):
        """Compute Jacobian matrix"""
        epsilon = 1e-6
        jacobian = np.zeros((6, len(self.kinematics.joint_names)))
        
        for i, joint_name in enumerate(self.kinematics.joint_names):
            # Perturb joint angle
            angles_plus = joint_angles.copy()
            angles_plus[joint_name] += epsilon
            
            # Forward kinematics
            pos_plus, ori_plus = self.kinematics.forward_kinematics(angles_plus)
            pos, ori = self.kinematics.forward_kinematics(joint_angles)
            
            # Numerical derivative
            jacobian[:3, i] = (pos_plus - pos) / epsilon
            jacobian[3:, i] = self._orientation_derivative(ori_plus, ori) / epsilon
            
        return jacobian
    
    def _orientation_error(self, target, current):
        """Calculate orientation error"""
        # Convert to rotation matrices
        R_target = Rotation.from_euler('xyz', target).as_matrix()
        R_current = Rotation.from_euler('xyz', current).as_matrix()
        
        # Orientation error
        error_matrix = 0.5 * (R_target @ R_current.T - R_current @ R_target.T)
        
        # Extract vector from skew-symmetric matrix
        error_vector = np.array([
            error_matrix[2, 1],
            error_matrix[0, 2],
            error_matrix[1, 0]
        ])
        
        return error_vector
```

## 10.3 Dynamics

### 10.3.1 Rigid Body Dynamics

```python
class HumanoidDynamics:
    def __init__(self):
        self.link_masses = self._initialize_link_masses()
        self.link_inertias = self._initialize_link_inertias()
        self.gravity = np.array([0, 0, -9.81])
        
    def _initialize_link_masses(self):
        """Initialize link masses"""
        return {
            'torso': 15.0,
            'left_thigh': 5.0,
            'left_shin': 3.0,
            'left_foot': 1.0,
            'right_thigh': 5.0,
            'right_shin': 3.0,
            'right_foot': 1.0,
            'left_upper_arm': 2.0,
            'left_forearm': 1.5,
            'right_upper_arm': 2.0,
            'right_forearm': 1.5,
            'head': 3.0
        }
    
    def _initialize_link_inertias(self):
        """Initialize link inertia tensors"""
        return {
            'torso': np.diag([0.5, 0.3, 0.2]),
            'left_thigh': np.diag([0.1, 0.1, 0.02]),
            'left_shin': np.diag([0.05, 0.05, 0.01]),
            # ... more links
        }
    
    def compute_dynamics(self, joint_angles, joint_velocities, joint_accelerations):
        """Compute joint torques using recursive Newton-Euler algorithm"""
        # Forward pass: compute velocities and accelerations
        link_velocities, link_accelerations = self._forward_pass(
            joint_angles, joint_velocities, joint_accelerations
        )
        
        # Backward pass: compute forces and torques
        joint_torques = self._backward_pass(
            link_velocities, link_accelerations
        )
        
        return joint_torques
    
    def _forward_pass(self, q, qd, qdd):
        """Forward pass of Newton-Euler algorithm"""
        velocities = {}
        accelerations = {}
        
        # Base link (assume fixed)
        velocities['base'] = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
        accelerations['base'] = np.array([0, 0, -9.81, 0, 0, 0])
        
        # Propagate through kinematic chain
        for i, joint_name in enumerate(q.keys()):
            # Transform to next link
            v_next = self._propagate_velocity(
                velocities['base'], qd[joint_name]
            )
            a_next = self._propagate_acceleration(
                accelerations['base'], qd[joint_name], qdd[joint_name]
            )
            
            velocities[joint_name] = v_next
            accelerations[joint_name] = a_next
            
        return velocities, accelerations
    
    def _backward_pass(self, velocities, accelerations):
        """Backward pass of Newton-Euler algorithm"""
        torques = {}
        
        # Initialize forces at end-effectors
        forces = {}
        for link_name in self.link_masses.keys():
            forces[link_name] = np.zeros(3)
        
        # Propagate forces backward
        for joint_name in reversed(list(velocities.keys())):
            # Compute joint torque
            if joint_name in self.link_inertias:
                I = self.link_inertias[joint_name]
                m = self.link_masses[joint_name]
                
                # Coriolis and centrifugal forces
                coriolis = self._compute_coriolis(velocities[joint_name], I)
                
                # Gravity compensation
                gravity_torque = self._compute_gravity_compensation(joint_name)
                
                # Total torque
                torques[joint_name] = coriolis + gravity_torque
                
        return torques
```

### 10.3.2 Contact Dynamics

```python
class ContactDynamics:
    def __init__(self):
        self.contact_points = ['left_foot', 'right_foot']
        self.friction_coefficient = 0.8
        self.ground_stiffness = 10000.0
        self.ground_damping = 100.0
        
    def compute_contact_forces(self, positions, velocities):
        """Compute contact forces with ground"""
        forces = {}
        
        for contact_point in self.contact_points:
            if contact_point in positions:
                pos = positions[contact_point]
                vel = velocities.get(contact_point, np.zeros(3))
                
                # Ground contact force
                if pos[2] < 0:  # Below ground
                    # Normal force
                    normal_force = -self.ground_stiffness * pos[2] - self.ground_damping * vel[2]
                    
                    # Friction force
                    tangential_vel = vel[:2]
                    if np.linalg.norm(tangential_vel) > 1e-6:
                        friction_direction = -tangential_vel / np.linalg.norm(tangential_vel)
                        friction_force = min(
                            self.friction_coefficient * normal_force,
                            np.linalg.norm(tangential_vel) * self.ground_damping
                        ) * friction_direction
                    else:
                        friction_force = np.zeros(2)
                    
                    # Total contact force
                    forces[contact_point] = np.array([
                        friction_force[0],
                        friction_force[1],
                        max(0, normal_force)
                    ])
                else:
                    forces[contact_point] = np.zeros(3)
                    
        return forces
    
    def check_stability(self, com_position, contact_forces):
        """Check if robot is statically stable"""
        # Compute center of pressure
        cop = np.zeros(2)
        total_normal_force = 0
        
        for contact_point, force in contact_forces.items():
            if force[2] > 0:  # Contact with ground
                cop += force[2] * force[:2]
                total_normal_force += force[2]
        
        if total_normal_force > 0:
            cop /= total_normal_force
        
        # Check stability margin
        stability_margin = np.linalg.norm(com_position[:2] - cop)
        
        return stability_margin < 0.1, stability_margin
```

## 10.4 Balance Control

### 10.4.1 Zero Moment Point (ZMP)

```python
class ZMPController:
    def __init__(self):
        self.zmp_preview_window = 1.0  # seconds
        self.control_frequency = 100.0  # Hz
        
    def compute_zmp(self, com_position, com_acceleration, total_mass):
        """Compute Zero Moment Point"""
        # ZMP formula
        zmp_x = com_position[0] - com_position[2] * com_acceleration[0] / 9.81
        zmp_y = com_position[1] - com_position[2] * com_acceleration[1] / 9.81
        
        return np.array([zmp_x, zmp_y])
    
    def preview_control(self, reference_zmp, current_com, horizon=1.0):
        """Model predictive control for ZMP tracking"""
        # State space model
        dt = 1.0 / self.control_frequency
        A = np.array([
            [1, dt, dt**2/2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        B = np.array([dt**3/6, dt**2/2, dt])
        
        # Preview control
        N = int(horizon * self.control_frequency)
        Q = np.eye(3 * N)
        R = 1.0
        
        # Solve optimization problem
        # This is a simplified version - in practice use QP solver
        control_sequence = self._solve_mpc(
            A, B, current_com, reference_zmp, N, Q, R
        )
        
        return control_sequence[0]  # First control input
    
    def _solve_mpc(self, A, B, x0, zmp_ref, N, Q, R):
        """Solve MPC optimization problem"""
        # Simplified MPC solution
        # In practice, use proper QP solver like cvxpy or qpOASES
        n_states = 3
        n_controls = 1
        
        # Build prediction matrices
        F = np.zeros((N * n_states, n_states))
        H = np.zeros((N * n_states, N * n_controls))
        
        x = x0.copy()
        for i in range(N):
            F[i*n_states:(i+1)*n_states, :] = np.linalg.matrix_power(A, i+1)
            
            for j in range(i+1):
                H[i*n_states:(i+1)*n_states, j*n_controls:(j+1)*n_controls] = \
                    np.linalg.matrix_power(A, i-j) @ B.reshape(-1, 1)
        
        # Cost matrices
        zmp_selection = np.array([1, 0, -9.81])  # Select ZMP from state
        C = np.kron(np.eye(N), zmp_selection)
        
        # Solve (simplified)
        P = H.T @ C.T @ Q @ C @ H + R * np.eye(N)
        q = H.T @ C.T @ Q @ (zmp_ref.flatten() - C @ F @ x0)
        
        # Solve linear system
        u_opt = np.linalg.solve(P, q)
        
        return u_opt
```

### 10.4.2 Balance Recovery

```python
class BalanceRecovery:
    def __init__(self, zmp_controller):
        self.zmp_controller = zmp_controller
        self.recovery_strategies = ['ankle_strategy', 'hip_strategy', 'step_strategy']
        
    def detect_fall(self, com_position, com_velocity, zmp_position):
        """Detect if robot is falling"""
        # Check if ZMP is outside support polygon
        support_polygon = self._get_support_polygon()
        
        outside_support = not self._point_in_polygon(zmp_position, support_polygon)
        high_velocity = np.linalg.norm(com_velocity) > 0.5
        large_tilt = abs(com_position[2]) > 0.1
        
        return outside_support or high_velocity or large_tilt
    
    def select_recovery_strategy(self, fall_state):
        """Select appropriate recovery strategy"""
        if fall_state['tilt_angle'] < 0.1:
            return 'ankle_strategy'
        elif fall_state['tilt_angle'] < 0.3:
            return 'hip_strategy'
        else:
            return 'step_strategy'
    
    def execute_ankle_strategy(self, current_angles, target_zmp):
        """Execute ankle strategy for balance recovery"""
        # Adjust ankle angles to shift ZMP
        ankle_adjustment = self._compute_ankle_adjustment(target_zmp)
        
        new_angles = current_angles.copy()
        new_angles['left_ankle_pitch'] += ankle_adjustment[0]
        new_angles['left_ankle_roll'] += ankle_adjustment[1]
        new_angles['right_ankle_pitch'] += ankle_adjustment[0]
        new_angles['right_ankle_roll'] += ankle_adjustment[1]
        
        return new_angles
    
    def execute_step_strategy(self, com_position, com_velocity):
        """Execute stepping strategy for balance recovery"""
        # Compute capture point
        capture_point = com_position[:2] + np.sqrt(com_position[2] / 9.81) * com_velocity[:2]
        
        # Plan foot placement
        foot_placement = self._plan_foot_placement(capture_point)
        
        return foot_placement
    
    def _get_support_polygon(self):
        """Get current support polygon"""
        # Simplified support polygon (feet positions)
        return np.array([
            [-0.1, -0.05],  # Left foot corners
            [-0.1, 0.05],
            [0.1, 0.05],   # Right foot corners
            [0.1, -0.05]
        ])
    
    def _point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        # Ray casting algorithm
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
                
            j = i
            
        return inside
```

## 10.5 Gait Planning

### 10.5.1 Trajectory Generation

```python
class GaitPlanner:
    def __init__(self):
        self.step_duration = 0.8  # seconds
        self.step_height = 0.05   # meters
        self.step_length = 0.3    # meters
        
    def generate_foot_trajectory(self, start_pos, end_pos, phase):
        """Generate foot trajectory using cubic spline"""
        # Parameterize trajectory
        t = phase  # 0 to 1
        
        # Cubic spline coefficients
        a0 = start_pos
        a1 = np.zeros(3)
        a2 = 3 * (end_pos - start_pos) - 2 * self.step_height * np.array([0, 0, 1])
        a3 = -2 * (end_pos - start_pos) + self.step_height * np.array([0, 0, 1])
        
        # Trajectory
        position = a0 + a1 * t + a2 * t**2 + a3 * t**3
        
        # Add height variation
        position[2] += 4 * self.step_height * t * (1 - t)
        
        return position
    
    def generate_com_trajectory(self, foot_positions, duration):
        """Generate center of mass trajectory"""
        time_points = np.linspace(0, duration, int(duration * 100))
        com_trajectory = []
        
        for t in time_points:
            # Interpolate between foot positions
            phase = (t % self.step_duration) / self.step_duration
            
            if phase < 0.5:  # Single support
                com = foot_positions[0] + 0.1 * np.array([0, 0, 1])
            else:  # Double support
                com = (foot_positions[0] + foot_positions[1]) / 2 + 0.05 * np.array([0, 0, 1])
                
            com_trajectory.append(com)
            
        return np.array(com_trajectory)
    
    def plan_walking_sequence(self, start_position, target_position, num_steps):
        """Plan complete walking sequence"""
        foot_steps = []
        
        # Generate footstep positions
        for i in range(num_steps):
            if i % 2 == 0:  # Left foot
                foot_pos = start_position + np.array([
                    (i // 2 + 1) * self.step_length,
                    0.1,  # Lateral offset
                    0
                ])
            else:  # Right foot
                foot_pos = start_position + np.array([
                    (i // 2 + 1) * self.step_length,
                    -0.1,  # Lateral offset
                    0
                ])
                
            foot_steps.append(foot_pos)
            
        return foot_steps
```

## 10.6 ROS 2 Integration

### 10.6.1 Kinematics Node

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Float64MultiArray

class HumanoidKinematicsNode(Node):
    def __init__(self):
        super().__init__('humanoid_kinematics_node')
        
        # Initialize kinematics
        self.kinematics = HumanoidKinematics()
        self.ik_solver = InverseKinematics(self.kinematics)
        
        # Publishers
        self.left_foot_pub = self.create_publisher(
            PoseStamped, 'left_foot_pose', 10
        )
        self.right_foot_pub = self.create_publisher(
            PoseStamped, 'right_foot_pose', 10
        )
        self.com_pub = self.create_publisher(
            Point, 'center_of_mass', 10
        )
        
        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10
        )
        self.target_sub = self.create_subscription(
            PoseStamped, 'target_pose', self.target_callback, 10
        )
        
        self.get_logger().info('Humanoid Kinematics Node started')
    
    def joint_callback(self, msg):
        """Process joint states"""
        # Convert joint states to dictionary
        joint_angles = {}
        for i, name in enumerate(msg.name):
            joint_angles[name] = msg.position[i]
        
        # Forward kinematics
        positions, orientations = self.kinematics.forward_kinematics(joint_angles)
        
        # Publish foot positions
        self._publish_foot_positions(positions, orientations)
        
        # Publish center of mass
        self._publish_com(positions)
    
    def target_callback(self, msg):
        """Process target pose for inverse kinematics"""
        target_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        target_orientation = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ])
        
        # Solve inverse kinematics
        current_angles = self._get_current_joint_angles()
        solution = self.ik_solver.inverse_kinematics(
            target_position, target_orientation, current_angles
        )
        
        # Publish joint commands
        self._publish_joint_commands(solution)
    
    def _publish_foot_positions(self, positions, orientations):
        """Publish foot positions"""
        # Left foot
        left_foot_msg = PoseStamped()
        left_foot_msg.header.stamp = self.get_clock().now().to_msg()
        left_foot_msg.header.frame_id = 'base_link'
        
        if 'left_ankle_roll' in positions:
            pos = positions['left_ankle_roll']
            left_foot_msg.pose.position.x = pos[0]
            left_foot_msg.pose.position.y = pos[1]
            left_foot_msg.pose.position.z = pos[2]
            
        self.left_foot_pub.publish(left_foot_msg)
        
        # Right foot
        right_foot_msg = PoseStamped()
        right_foot_msg.header.stamp = self.get_clock().now().to_msg()
        right_foot_msg.header.frame_id = 'base_link'
        
        if 'right_ankle_roll' in positions:
            pos = positions['right_ankle_roll']
            right_foot_msg.pose.position.x = pos[0]
            right_foot_msg.pose.position.y = pos[1]
            right_foot_msg.pose.position.z = pos[2]
            
        self.right_foot_pub.publish(right_foot_msg)
    
    def _publish_com(self, positions):
        """Publish center of mass"""
        com_msg = Point()
        
        # Simple COM calculation (would need link masses in practice)
        com = np.zeros(3)
        count = 0
        for pos in positions.values():
            com += pos
            count += 1
            
        if count > 0:
            com /= count
            
        com_msg.x = com[0]
        com_msg.y = com[1]
        com_msg.z = com[2]
        
        self.com_pub.publish(com_msg)
    
    def _get_current_joint_angles(self):
        """Get current joint angles (simplified)"""
        # In practice, this would read from current joint states
        return {name: 0.0 for name in self.kinematics.joint_names}
    
    def _publish_joint_commands(self, joint_angles):
        """Publish joint commands"""
        msg = Float64MultiArray()
        msg.data = [joint_angles.get(name, 0.0) for name in self.kinematics.joint_names]
        
        # Create publisher if needed
        if not hasattr(self, 'joint_cmd_pub'):
            self.joint_cmd_pub = self.create_publisher(
                Float64MultiArray, 'joint_commands', 10
            )
            
        self.joint_cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidKinematicsNode()
    
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

## Summary

This chapter covered the fundamental concepts and practical implementation of humanoid robot kinematics and dynamics:

- **Forward Kinematics**: Computing end-effector positions from joint angles
- **Inverse Kinematics**: Solving joint angles for desired positions
- **Dynamics**: Computing forces and torques for motion
- **Balance Control**: ZMP-based stability and recovery strategies
- **Gait Planning**: Generating walking trajectories
- **ROS 2 Integration**: Real-time kinematics computation

The next chapter will explore human-robot interaction systems for creating natural and intuitive interfaces with humanoid robots.
---
title: "Chapter 9: Reinforcement Learning for Robot Control"
sidebar_position: 9
---

# Chapter 9: Reinforcement Learning for Robot Control

## 9.1 Basics of RL for Robotics

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent receives rewards for good actions and penalties for bad actions, learning optimal behavior through trial and error.

### Key RL Concepts

- **Agent**: The robot or controller that makes decisions
- **Environment**: The world the robot operates in
- **State**: The current situation of the robot and environment
- **Action**: Commands the robot can execute
- **Reward**: Feedback signal for actions
- **Policy**: Strategy for selecting actions
- **Value Function**: Expected future reward from a state
- **Q-Function**: Expected future reward for state-action pairs

### RL Algorithms for Robotics

1. **Deep Q-Networks (DQN)**: Value-based learning
2. **Proximal Policy Optimization (PPO)**: Policy-based learning
3. **Soft Actor-Critic (SAC)**: Off-policy actor-critic
4. **Twin Delayed DDPG (TD3)**: Model-free control
5. **Model-Based RL**: Learn environment dynamics

### Basic RL Framework

```python
# rl_framework.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class RLAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = ReplayBuffer(max_size=1000000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        
    def select_action(self, state, training=True):
        """Select action using policy network"""
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        
        if training:
            return action.cpu().numpy()[0]
        else:
            # Deterministic action for evaluation
            _, deterministic_action = self.actor.sample(state)
            return deterministic_action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.buffer.add(state, action, reward, next_state, done)
    
    def update(self, batch_size=256):
        """Update actor and critic networks"""
        
        if len(self.buffer) < batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update critic
        critic_loss = self.update_critic(states, actions, rewards, next_states, dones)
        
        # Update actor
        actor_loss = self.update_actor(states)
        
        # Soft update target networks
        self.soft_update(self.critic.target, self.critic, self.tau)
        
        return actor_loss, critic_loss
    
    def update_critic(self, states, actions, rewards, next_states, dones):
        """Update critic network"""
        
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic.target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Get current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_actor(self, states):
        """Update actor network"""
        
        # Sample actions
        actions, log_probs = self.actor.sample(states)
        
        # Compute Q-values
        q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)
        
        # Compute actor loss
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def soft_update(self, target, source, tau):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()
        
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim)  # Mean and log_std
        )
        
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state):
        x = self.network(state)
        mean, log_std = x.chunk(2, dim=-1)
        
        # Clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        noise = torch.randn_like(mean)
        action = mean + std * noise
        
        # Squash action to [-1, 1]
        action = torch.tanh(action) * self.max_action
        
        # Compute log probability
        log_prob = torch.distributions.Normal(mean, std).log_prob(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Target networks
        self.target = type(self)(state_dim, action_dim)
        self.target.load_state_dict(self.state_dict())
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
```

## 9.2 Training in Simulation (Isaac Gym)

### Isaac Gym Integration

```python
# isaac_gym_rl.py
import isaacgym
import torch
import numpy as np
from rl_framework import RLAgent

class HumanoidIsaacGymEnv:
    def __init__(self, num_envs=512, headless=False):
        
        # Simulation parameters
        self.num_envs = num_envs
        self.headless = headless
        
        # Initialize Isaac Gym
        self.sim = isaacgym.Sim()
        
        # Create humanoid environments
        self.create_humanoid_envs()
        
        # State and action dimensions
        self.state_dim = self.get_state_dim()
        self.action_dim = self.get_action_dim()
        
        # Training parameters
        self.max_episode_length = 1000
        self.current_step = 0
        
    def create_humanoid_envs(self):
        """Create humanoid robot environments"""
        
        # Asset loading
        asset_root = "./assets"
        asset_file = "humanoid.urdf"
        
        # Create environments
        env = isaacgym.create_env(
            num_envs=self.num_envs,
            env_spacing=2.0,
            asset_root=asset_root,
            asset_filename=asset_file,
            headless=self.headless
        )
        
        # Configure environment
        self.configure_physics()
        self.configure_rewards()
        
        return env
    
    def configure_physics(self):
        """Configure physics parameters"""
        
        # Set gravity
        self.sim.set_gravity(np.array([0.0, 0.0, -9.81]))
        
        # Set time step
        self.sim.set_time_step(0.02)  # 50 Hz
        
        # Configure contact parameters
        self.sim.set_contact_parameters(
            restitution=0.1,
            friction=0.8,
            damping=0.0
        )
    
    def configure_rewards(self):
        """Configure reward functions"""
        
        self.reward_scales = {
            'velocity': 1.0,
            'torque': -0.0001,
            'orientation': -0.5,
            'height': -2.0,
            'action_rate': -0.01
        }
    
    def reset(self):
        """Reset environments"""
        
        # Reset humanoid positions
        self.reset_humanoid_poses()
        
        # Get initial states
        states = self.get_states()
        
        # Reset episode length
        self.current_step = 0
        
        return states
    
    def step(self, actions):
        """Step environments"""
        
        # Apply actions
        self.apply_actions(actions)
        
        # Step simulation
        self.sim.step()
        
        # Get new states
        states = self.get_states()
        
        # Compute rewards
        rewards = self.compute_rewards()
        
        # Check dones
        dones = self.check_dones()
        
        # Increment step
        self.current_step += 1
        
        return states, rewards, dones
    
    def apply_actions(self, actions):
        """Apply actions to humanoid joints"""
        
        # Convert actions to torques
        torques = self.actions_to_torques(actions)
        
        # Apply torques
        self.sim.apply_torques(torques)
    
    def actions_to_torques(self, actions):
        """Convert neural network actions to joint torques"""
        
        # Scale actions to torque limits
        max_torque = 50.0  # Nm
        torques = actions * max_torque
        
        return torques
    
    def get_states(self):
        """Get environment states"""
        
        states = []
        
        for i in range(self.num_envs):
            # Get joint positions
            joint_pos = self.sim.get_joint_positions(i)
            
            # Get joint velocities
            joint_vel = self.sim.get_joint_velocities(i)
            
            # Get body orientation
            orientation = self.sim.get_body_orientation(i)
            
            # Get linear velocity
            lin_vel = self.sim.get_linear_velocity(i)
            
            # Get angular velocity
            ang_vel = self.sim.get_angular_velocity(i)
            
            # Concatenate state
            state = np.concatenate([
                joint_pos,
                joint_vel,
                orientation,
                lin_vel,
                ang_vel
            ])
            
            states.append(state)
        
        return np.array(states)
    
    def compute_rewards(self):
        """Compute rewards for all environments"""
        
        rewards = np.zeros(self.num_envs)
        
        for i in range(self.num_envs):
            # Velocity reward
            lin_vel = self.sim.get_linear_velocity(i)
            velocity_reward = lin_vel[0] * self.reward_scales['velocity']
            
            # Torque penalty
            torques = self.sim.get_applied_torques(i)
            torque_penalty = np.sum(np.square(torques)) * self.reward_scales['torque']
            
            # Orientation penalty
            orientation = self.sim.get_body_orientation(i)
            upright = orientation[3]  # quaternion w component
            orientation_penalty = (1 - upright) * self.reward_scales['orientation']
            
            # Height penalty
            height = self.sim.get_body_position(i)[2]
            target_height = 1.7
            height_penalty = abs(height - target_height) * self.reward_scales['height']
            
            # Total reward
            rewards[i] = (velocity_reward + torque_penalty + 
                         orientation_penalty + height_penalty)
        
        return rewards
    
    def check_dones(self):
        """Check if episodes are done"""
        
        dones = np.zeros(self.num_envs, dtype=bool)
        
        for i in range(self.num_envs):
            # Check if robot fell
            height = self.sim.get_body_position(i)[2]
            if height < 0.5:
                dones[i] = True
            
            # Check episode length
            if self.current_step >= self.max_episode_length:
                dones[i] = True
        
        return dones
    
    def get_state_dim(self):
        """Get state dimension"""
        # 23 joints + velocities, 4 orientation, 3 linear vel, 3 angular vel
        return 23 + 23 + 4 + 3 + 3
    
    def get_action_dim(self):
        """Get action dimension"""
        # 23 controllable joints
        return 23

class HumanoidRLTrainer:
    def __init__(self, num_envs=512):
        
        # Create environment
        self.env = HumanoidIsaacGymEnv(num_envs=num_envs)
        
        # Create agent
        self.agent = RLAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )
        
        # Training parameters
        self.num_episodes = 10000
        self.save_interval = 100
        self.eval_interval = 50
        
        # Metrics
        self.episode_rewards = []
        self.eval_rewards = []
    
    def train(self):
        """Main training loop"""
        
        for episode in range(self.num_episodes):
            
            # Reset environment
            states = self.env.reset()
            
            # Episode metrics
            episode_reward = 0
            episode_length = 0
            
            # Training loop
            while True:
                
                # Select actions
                actions = np.array([
                    self.agent.select_action(state) for state in states
                ])
                
                # Step environment
                next_states, rewards, dones = self.env.step(actions)
                
                # Store transitions
                for i in range(len(states)):
                    self.agent.store_transition(
                        states[i], actions[i], rewards[i], 
                        next_states[i], dones[i]
                    )
                
                # Update agent
                actor_loss, critic_loss = self.agent.update()
                
                # Update metrics
                episode_reward += np.mean(rewards)
                episode_length += 1
                
                # Check if episode ended
                if np.any(dones):
                    break
                
                states = next_states
            
            # Log metrics
            self.episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}")
            
            # Save checkpoint
            if episode % self.save_interval == 0:
                self.save_checkpoint(episode)
            
            # Evaluate
            if episode % self.eval_interval == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                print(f"Evaluation Reward: {eval_reward:.2f}")
    
    def evaluate(self, num_episodes=10):
        """Evaluate agent performance"""
        
        total_reward = 0
        
        for _ in range(num_episodes):
            states = self.env.reset()
            episode_reward = 0
            
            while True:
                # Select deterministic actions
                actions = np.array([
                    self.agent.select_action(state, training=False) 
                    for state in states
                ])
                
                # Step environment
                next_states, rewards, dones = self.env.step(actions)
                
                episode_reward += np.mean(rewards)
                
                if np.any(dones):
                    break
                
                states = next_states
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def save_checkpoint(self, episode):
        """Save model checkpoint"""
        
        torch.save({
            'episode': episode,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'eval_rewards': self.eval_rewards
        }, f'checkpoint_episode_{episode}.pth')

def main():
    trainer = HumanoidRLTrainer(num_envs=512)
    trainer.train()

if __name__ == '__main__':
    main()
```

## 9.3 Policy Deployment on Real Hardware

### Sim-to-Real Transfer

```python
# sim2real_transfer.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import torch
import numpy as np
import time

class HumanoidRLController(Node):
    def __init__(self, model_path, sim2real=True):
        super().__init__('humanoid_rl_controller')
        
        # Load trained model
        self.agent = torch.load(model_path, map_location='cpu')
        self.agent.eval()
        
        # Sim-to-real parameters
        self.sim2real = sim2real
        if sim2real:
            self.apply_domain_randomization()
        
        # State buffer
        self.state_buffer = []
        self.buffer_size = 10
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        
        # Publishers
        self.torque_pub = self.create_publisher(
            Float64MultiArray, '/joint_torques', 10)
        
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        
        # Control timer
        self.timer = self.create_timer(0.02, self.control_loop)  # 50 Hz
        
        # Current state
        self.current_state = None
        self.last_action = None
        
        self.get_logger().info('Humanoid RL Controller initialized')
    
    def apply_domain_randomization(self):
        """Apply domain randomization for sim-to-real transfer"""
        
        # Add noise to observations
        self.observation_noise = 0.01
        
        # Scale actions for safety
        self.action_scale = 0.5
        
        # Add safety constraints
        self.max_torque = 30.0  # Reduced from 50.0
        
        # Low-pass filter for actions
        self.action_filter = 0.9
        self.filtered_action = None
    
    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        
        # Extract joint positions and velocities
        joint_positions = list(msg.position)
        joint_velocities = list(msg.velocity) if msg.velocity else [0.0] * len(msg.position)
        
        # Update state buffer
        self.state_buffer.append({
            'joint_pos': joint_positions,
            'joint_vel': joint_velocities,
            'timestamp': time.time()
        })
        
        # Keep buffer size limited
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
    
    def imu_callback(self, msg):
        """Handle IMU updates"""
        
        # Extract orientation and angular velocity
        orientation = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]
        
        angular_velocity = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]
        
        # Update latest state
        if self.state_buffer:
            self.state_buffer[-1]['orientation'] = orientation
            self.state_buffer[-1]['angular_vel'] = angular_velocity
    
    def control_loop(self):
        """Main control loop"""
        
        if not self.state_buffer:
            return
        
        # Get current state
        state = self.get_current_state()
        
        if state is None:
            return
        
        # Add observation noise if sim2real
        if self.sim2real:
            state = state + np.random.normal(0, self.observation_noise, state.shape)
        
        # Get action from policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = self.agent.actor.sample(state_tensor)
            action = action.cpu().numpy()[0]
        
        # Apply sim2real modifications
        if self.sim2real:
            action = self.postprocess_action(action)
        
        # Publish torques
        self.publish_torques(action)
        
        self.last_action = action
    
    def get_current_state(self):
        """Construct current state from sensor data"""
        
        if not self.state_buffer:
            return None
        
        latest_state = self.state_buffer[-1]
        
        # Check if all required data is available
        if 'orientation' not in latest_state or 'angular_vel' not in latest_state:
            return None
        
        # Construct state vector
        state = np.concatenate([
            latest_state['joint_pos'],
            latest_state['joint_vel'],
            latest_state['orientation'],
            latest_state['angular_vel']
        ])
        
        return state
    
    def postprocess_action(self, action):
        """Postprocess action for real robot"""
        
        # Scale actions
        action = action * self.action_scale
        
        # Clamp to torque limits
        action = np.clip(action, -self.max_torque, self.max_torque)
        
        # Apply low-pass filter
        if self.filtered_action is None:
            self.filtered_action = action
        else:
            self.filtered_action = (self.action_filter * self.filtered_action + 
                                 (1 - self.action_filter) * action)
        
        return self.filtered_action
    
    def publish_torques(self, torques):
        """Publish joint torques"""
        
        msg = Float64MultiArray()
        msg.data = torques.tolist()
        
        self.torque_pub.publish(msg)
    
    def emergency_stop(self):
        """Emergency stop - zero torques"""
        
        msg = Float64MultiArray()
        msg.data = [0.0] * 23  # Zero all torques
        
        self.torque_pub.publish(msg)
        
        self.get_logger().warn('Emergency stop activated')

class SafetyMonitor(Node):
    """Safety monitoring for RL controller"""
    
    def __init__(self):
        super().__init__('safety_monitor')
        
        # Safety thresholds
        self.max_joint_velocity = 10.0  # rad/s
        self.max_angular_velocity = 5.0  # rad/s
        self.min_height = 0.5  # meters
        self.max_orientation_error = 45.0  # degrees
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.check_joint_safety, 10)
        
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.check_imu_safety, 10)
        
        # Publishers
        self.emergency_pub = self.create_publisher(
            Bool, '/emergency_stop', 10)
        
        self.get_logger().info('Safety Monitor initialized')
    
    def check_joint_safety(self, msg):
        """Check joint velocity safety"""
        
        if msg.velocity:
            max_vel = max(abs(v) for v in msg.velocity)
            
            if max_vel > self.max_joint_velocity:
                self.trigger_emergency_stop("Joint velocity too high")
    
    def check_imu_safety(self, msg):
        """Check IMU safety"""
        
        # Check angular velocity
        ang_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        if np.linalg.norm(ang_vel) > self.max_angular_velocity:
            self.trigger_emergency_stop("Angular velocity too high")
        
        # Check orientation
        quat = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])
        
        # Convert to Euler angles
        euler = self.quaternion_to_euler(quat)
        
        # Check if robot is falling
        if abs(euler[0]) > np.radians(self.max_orientation_error) or \
           abs(euler[1]) > np.radians(self.max_orientation_error):
            self.trigger_emergency_stop("Orientation error too high")
    
    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles"""
        
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def trigger_emergency_stop(self, reason):
        """Trigger emergency stop"""
        
        self.get_logger().error(f'Emergency stop: {reason}')
        
        msg = Bool()
        msg.data = True
        self.emergency_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    # Create executor for multi-threaded operation
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    
    # Create nodes
    controller = HumanoidRLController('humanoid_model.pth', sim2real=True)
    safety_monitor = SafetyMonitor()
    
    # Add nodes to executor
    executor.add_node(controller)
    executor.add_node(safety_monitor)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        controller.emergency_stop()
        controller.destroy_node()
        safety_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 9.4 Fine-Tuning with Real-World Data

### Online Fine-Tuning

```python
# online_finetuning.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class OnlineFineTuner(Node):
    def __init__(self, pretrained_model_path):
        super().__init__('online_finetuner')
        
        # Load pretrained model
        self.agent = torch.load(pretrained_model_path, map_location='cpu')
        
        # Create separate fine-tuning network
        self.finetune_net = FineTuneNetwork(self.agent.actor).to('cpu')
        
        # Optimizer for fine-tuning
        self.optimizer = optim.Adam(self.finetune_net.parameters(), lr=1e-5)
        
        # Experience buffer for fine-tuning
        self.buffer = deque(maxlen=10000)
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.reward_sub = self.create_subscription(
            Float32, '/reward_signal', self.reward_callback, 10)
        
        # Publishers
        self.action_pub = self.create_publisher(
            Float64MultiArray, '/joint_torques', 10)
        
        # Fine-tuning parameters
        self.finetune_interval = 100  # Update every 100 steps
        self.batch_size = 32
        self.step_count = 0
        
        # Current episode data
        self.current_trajectory = []
        
        self.get_logger().info('Online Fine-Tuner initialized')
    
    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        
        # Get current state
        state = self.extract_state(msg)
        
        # Get action from policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = self.agent.actor.sample(state_tensor)
            action = action.cpu().numpy()[0]
        
        # Apply fine-tuning if available
        if self.finetune_net.training:
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            finetuned_action = self.finetune_net(state_tensor, action_tensor)
            action = finetuned_action.cpu().numpy()[0]
        
        # Publish action
        self.publish_action(action)
        
        # Store transition
        self.current_trajectory.append({
            'state': state,
            'action': action,
            'timestamp': self.get_clock().now().to_msg()
        })
        
        self.step_count += 1
    
    def reward_callback(self, msg):
        """Handle reward signal"""
        
        if self.current_trajectory:
            # Assign reward to last state-action pair
            self.current_trajectory[-1]['reward'] = msg.data
            
            # Add to buffer if trajectory is complete
            if len(self.current_trajectory) > 1:
                for i in range(len(self.current_trajectory) - 1):
                    transition = {
                        'state': self.current_trajectory[i]['state'],
                        'action': self.current_trajectory[i]['action'],
                        'reward': self.current_trajectory[i+1]['reward'],
                        'next_state': self.current_trajectory[i+1]['state'],
                        'done': False
                    }
                    self.buffer.append(transition)
        
        # Fine-tune if enough data
        if len(self.buffer) > self.batch_size and self.step_count % self.finetune_interval == 0:
            self.fine_tune()
    
    def extract_state(self, joint_state_msg):
        """Extract state from joint state message"""
        
        # Combine joint positions, velocities, and other sensor data
        state = np.concatenate([
            list(joint_state_msg.position),
            list(joint_state_msg.velocity) if joint_state_msg.velocity else [0.0] * len(joint_state_msg.position)
        ])
        
        return state
    
    def publish_action(self, action):
        """Publish action to robot"""
        
        msg = Float64MultiArray()
        msg.data = action.tolist()
        
        self.action_pub.publish(msg)
    
    def fine_tune(self):
        """Perform fine-tuning update"""
        
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = list(self.buffer)[-self.batch_size:]
        states = torch.FloatTensor([t['state'] for t in batch])
        actions = torch.FloatTensor([t['action'] for t in batch])
        rewards = torch.FloatTensor([t['reward'] for t in batch])
        next_states = torch.FloatTensor([t['next_state'] for t in batch])
        
        # Compute fine-tuning loss
        loss = self.compute_finetune_loss(states, actions, rewards, next_states)
        
        # Update fine-tuning network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.finetune_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.get_logger().info(f'Fine-tuning loss: {loss.item():.4f}')
    
    def compute_finetune_loss(self, states, actions, rewards, next_states):
        """Compute fine-tuning loss"""
        
        # Get original actions
        with torch.no_grad():
            original_actions, _ = self.agent.actor.sample(states)
        
        # Get fine-tuned actions
        finetuned_actions = self.finetune_net(states, original_actions)
        
        # Compute behavioral cloning loss
        bc_loss = nn.MSELoss()(finetuned_actions, actions)
        
        # Compute reward-based loss
        with torch.no_grad():
            next_actions, _ = self.agent.actor.sample(next_states)
            q_values = self.agent.critic(states, finetuned_actions)[0]
            next_q_values = self.agent.critic(next_states, next_actions)[0]
            targets = rewards + 0.99 * next_q_values
        
        q_loss = nn.MSELoss()(q_values, targets)
        
        # Combined loss
        total_loss = bc_loss + 0.1 * q_loss
        
        return total_loss

class FineTuneNetwork(nn.Module):
    """Network for fine-tuning pretrained policy"""
    
    def __init__(self, pretrained_actor):
        super().__init__()
        
        # Freeze pretrained network
        for param in pretrained_actor.parameters():
            param.requires_grad = False
        
        self.pretrained_actor = pretrained_actor
        
        # Fine-tuning layers
        self.finetune_layers = nn.Sequential(
            nn.Linear(pretrained_actor.action_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, pretrained_actor.action_dim),
            nn.Tanh()
        )
    
    def forward(self, state, original_action):
        """Forward pass"""
        
        # Get original action from pretrained network
        with torch.no_grad():
            _, _ = self.pretrained_actor.sample(state)
            # Extract action from pretrained network (implementation specific)
        
        # Concatenate state and original action
        combined = torch.cat([state, original_action], dim=-1)
        
        # Apply fine-tuning layers
        delta = self.finetune_layers(combined)
        
        # Add delta to original action
        finetuned_action = original_action + delta
        
        return finetuned_action

def main(args=None):
    rclpy.init(args=args)
    
    finetuner = OnlineFineTuner('humanoid_model.pth')
    
    try:
        rclpy.spin(finetuner)
    except KeyboardInterrupt:
        pass
    finally:
        # Save fine-tuned model
        torch.save(finetuner.finetune_net.state_dict(), 'finetuned_model.pth')
        finetuner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

**Next Chapter**: In Chapter 10, we'll explore Humanoid Kinematics & Dynamics for understanding and controlling humanoid robot motion.

---
title: "Chapter 5: High-Fidelity Simulation with NVIDIA Isaac Sim"
sidebar_position: 5
---

# Chapter 5: High-Fidelity Simulation with NVIDIA Isaac Sim

## 5.1 Introduction to Omniverse & Isaac Sim

### What is NVIDIA Isaac Sim?

Isaac Sim is NVIDIA's robotics simulation platform built on Omniverse, providing photorealistic rendering, physics simulation, and AI-powered tools for robotics development. It bridges the gap between simulation and reality with unprecedented realism.

### Key Features

- **Photorealistic Rendering**: RTX-powered ray tracing for realistic visuals
- **Advanced Physics**: PhysX-based simulation with accurate material properties
- **Synthetic Data Generation**: AI-ready training data at scale
- **Cloud-Native**: Distributed simulation across multiple GPUs
- **ROS 2 Integration**: Seamless connection with robot software

### System Requirements

**Minimum Requirements:**
- NVIDIA RTX GPU (RTX 3060 or better)
- 16GB RAM
- Intel i7 or AMD Ryzen 7 CPU
- 100GB SSD storage

**Recommended Setup:**
- NVIDIA RTX 4090 or A100
- 32GB+ RAM
- Intel i9 or AMD Ryzen 9 CPU
- 500GB NVMe SSD

### Installation

```bash
# Download Isaac Sim from NVIDIA Developer portal
# Extract and run the installer

# Set up environment
export ISAAC_SIM_PATH=/path/to/isaac-sim
export PYTHONPATH=$ISAAC_SIM_PATH/kit/python:$PYTHONPATH
export LD_LIBRARY_PATH=$ISAAC_SIM_PATH/kit/lib:$LD_LIBRARY_PATH

# Launch Isaac Sim
cd $ISAAC_SIM_PATH
./python.sh standalone_examples/api/omni.isaac.sim/main.py
```

## 5.2 Photorealistic Environments & Synthetic Data

### Creating Photorealistic Scenes

```python
# isaac_sim_environment.py
import carb
import omni.kit.commands
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import Usd, UsdGeom, Gf
import numpy as np

class PhotorealisticEnvironment:
    def __init__(self):
        self.world = World()
        self.assets_root = get_assets_root_path()
        self.stage = get_current_stage()
        
    def create_indoor_environment(self):
        """Create a photorealistic indoor environment"""
        
        # Load kitchen environment
        kitchen_usd = f"{self.assets_root}/Isaac/Environments/Simple_Warehouse/simple_warehouse.usd"
        add_reference_to_stage(kitchen_usd, "/simple_warehouse")
        
        # Configure lighting for realism
        self.setup_realistic_lighting()
        
        # Add materials with PBR properties
        self.add_realistic_materials()
        
        # Configure camera with real-world parameters
        self.setup_realistic_cameras()
        
    def setup_realistic_lighting(self):
        """Setup physically-based lighting"""
        
        # Dome light for global illumination
        from omni.isaac.core.utils.prims import define_prim
        from pxr import UsdLux
        
        # HDRI dome light
        dome_light = define_prim("/DomeLight", "DomeLight")
        dome_light.CreateAttribute("texture:file", Sdf.ValueTypeNames.Asset).Set(
            f"{self.assets_root}/Isaac/Environments/Indoor/studio_small_03_4k.hdr"
        )
        
        # Area lights for local illumination
        area_light1 = define_prim("/AreaLight1", "RectLight")
        area_light1.GetAttribute("intensity").Set(5000)
        area_light1.GetAttribute("width").Set(2.0)
        area_light1.GetAttribute("height").Set(2.0)
        area_light1.GetAttribute("color").Set(Gf.Vec3f(1.0, 0.95, 0.8))
        
        # Point lights for accent
        point_light1 = define_prim("/PointLight1", "PointLight")
        point_light1.GetAttribute("intensity").Set(10000)
        point_light1.GetAttribute("position").Set(Gf.Vec3f(2.0, 2.0, 3.0))
        
    def add_realistic_materials(self):
        """Add physically-based rendering materials"""
        
        from omni.isaac.core.utils.materials import OmniPBRSurface
        
        # Wood material for floor
        wood_material = OmniPBRSurface(
            prim_path="/Looks/WoodMaterial",
            props={
                "diffuse_texture": f"{self.assets_root}/Isaac/Environments/Indoor/wood_floor_diffuse.png",
                "normal_texture": f"{self.assets_root}/Isaac/Environments/Indoor/wood_floor_normal.png",
                "roughness_texture": f"{self.assets_root}/Isaac/Environments/Indoor/wood_floor_roughness.png",
                "metallic": 0.0,
                "roughness": 0.8
            }
        )
        
        # Metal material for robot parts
        metal_material = OmniPBRSurface(
            prim_path="/Looks/MetalMaterial",
            props={
                "diffuse_color": (0.7, 0.7, 0.7),
                "metallic": 1.0,
                "roughness": 0.3,
                "normal_texture": f"{self.assets_root}/Isaac/Environments/Indoor/metal_normal.png"
            }
        )
        
        # Apply materials to objects
        self.apply_material_to_prim("/simple_warehouse/floor", wood_material)
        self.apply_material_to_prim("/humanoid", metal_material)
        
    def setup_realistic_cameras(self):
        """Setup cameras with real-world parameters"""
        
        from omni.isaac.core.sensors import Camera
        
        # Main RGB camera with DSLR parameters
        rgb_camera = Camera(
            prim_path="/humanoid/head_camera",
            position=(0.1, 0, 0.5),
            frequency=30,
            resolution=(1920, 1080),
            data_types=["rgb", "depth", "semantic_segmentation"],
            clipping_range=(0.1, 100.0)
        )
        
        # Configure camera properties
        rgb_camera.set_aperture(f_stop=2.8)
        rgb_camera.set_focal_length(35)  # 35mm equivalent
        rgb_camera.set_shutter_speed(1/125)
        rgb_camera.set_iso(400)
        
        # Add camera noise for realism
        rgb_camera.add_noise_model(
            noise_type="gaussian",
            mean=0.0,
            std_dev=0.01
        )
        
        return rgb_camera
    
    def apply_material_to_prim(self, prim_path, material):
        """Apply material to a primitive"""
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim:
            material.apply(prim)

# Initialize environment
env = PhotorealisticEnvironment()
env.create_indoor_environment()
```

### Synthetic Data Generation

```python
# synthetic_data_generator.py
import omni.kit.commands
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.synthetic_utils import SyntheticDataHelper
import numpy as np
import cv2
import os

class SyntheticDataGenerator:
    def __init__(self, output_dir="./synthetic_data"):
        self.world = World()
        self.output_dir = output_dir
        self.sd_helper = SyntheticDataHelper()
        
        # Create output directories
        os.makedirs(f"{output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/semantic", exist_ok=True)
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)
        
    def generate_training_data(self, num_samples=1000):
        """Generate synthetic training data for AI models"""
        
        for i in range(num_samples):
            # Randomize environment
            self.randomize_environment()
            
            # Randomize robot pose
            self.randomize_robot_pose()
            
            # Randomize lighting
            self.randomize_lighting()
            
            # Capture data
            rgb_data = self.capture_rgb()
            depth_data = self.capture_depth()
            semantic_data = self.capture_semantic()
            
            # Generate annotations
            annotations = self.generate_annotations()
            
            # Save data
            self.save_sample(i, rgb_data, depth_data, semantic_data, annotations)
            
            if i % 100 == 0:
                print(f"Generated {i} samples...")
    
    def randomize_environment(self):
        """Randomize environment for domain randomization"""
        
        # Random object positions
        objects = ["table", "chair", "box", "bottle"]
        for obj in objects:
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            z = 0.0
            orientation = np.random.uniform(0, 2*np.pi)
            
            # Move object
            obj_prim = self.stage.GetPrimAtPath(f"/{obj}")
            if obj_prim:
                omni.kit.commands.execute(
                    "TransformPrimCommand",
                    path=obj_prim.GetPath(),
                    new_transform=Gf.Transform(
                        Gf.Rotation(Gf.Vec3d(0, 0, 1), orientation),
                        Gf.Vec3d(x, y, z)
                    )
                )
        
        # Random material properties
        self.randomize_materials()
    
    def randomize_robot_pose(self):
        """Randomize humanoid robot pose"""
        
        # Random joint positions
        joint_names = ["left_hip", "left_knee", "right_hip", "right_knee",
                      "left_shoulder", "right_shoulder", "neck"]
        
        for joint in joint_names:
            angle = np.random.uniform(-np.pi/4, np.pi/4)
            # Apply joint angle
            self.set_joint_angle(joint, angle)
    
    def randomize_lighting(self):
        """Randomize lighting conditions"""
        
        # Random dome light intensity
        dome_light = self.stage.GetPrimAtPath("/DomeLight")
        if dome_light:
            intensity = np.random.uniform(1000, 5000)
            dome_light.GetAttribute("intensity").Set(intensity)
        
        # Random area light positions and colors
        for i in range(3):
            light_path = f"/AreaLight{i+1}"
            light = self.stage.GetPrimAtPath(light_path)
            if light:
                # Random position
                x = np.random.uniform(-3, 3)
                y = np.random.uniform(-3, 3)
                z = np.random.uniform(2, 4)
                light.GetAttribute("position").Set(Gf.Vec3f(x, y, z))
                
                # Random color temperature
                temp = np.random.uniform(3000, 7000)  # Kelvin
                color = self.kelvin_to_rgb(temp)
                light.GetAttribute("color").Set(Gf.Vec3f(*color))
    
    def kelvin_to_rgb(self, kelvin):
        """Convert color temperature to RGB"""
        # Simplified color temperature conversion
        if kelvin <= 6600:
            r = 255
            g = max(0, min(255, -155.25 + 0.445 * kelvin / 100 - 99.2))
            b = max(0, min(255, -254.76 + 0.827 * kelvin / 100 + 104.96))
        else:
            r = max(0, min(255, 351.98 - 0.114 * kelvin / 100))
            g = max(0, min(255, 325.45 - 0.079 * kelvin / 100))
            b = 255
        
        return (r/255, g/255, b/255)
    
    def capture_rgb(self):
        """Capture RGB image with realistic camera effects"""
        
        # Get camera data
        rgb_data = self.sd_helper.get_rgb_data("/humanoid/head_camera")
        
        # Add realistic camera effects
        rgb_data = self.add_camera_effects(rgb_data)
        
        return rgb_data
    
    def add_camera_effects(self, image):
        """Add realistic camera effects like lens distortion, chromatic aberration"""
        
        # Lens distortion
        h, w = image.shape[:2]
        fx, fy = w/2, h/2
        cx, cy = w/2, h/2
        
        # Barrel distortion
        k1, k2 = -0.1, 0.01
        
        # Apply distortion
        distorted = np.zeros_like(image)
        for y in range(h):
            for x in range(w):
                # Normalize coordinates
                x_norm = (x - cx) / fx
                y_norm = (y - cy) / fy
                
                # Apply distortion
                r2 = x_norm**2 + y_norm**2
                distortion = 1 + k1 * r2 + k2 * r2**2
                
                x_dist = x_norm * distortion * fx + cx
                y_dist = y_norm * distortion * fy + cy
                
                # Bilinear interpolation
                if 0 <= x_dist < w-1 and 0 <= y_dist < h-1:
                    x1, y1 = int(x_dist), int(y_dist)
                    x2, y2 = min(x1+1, w-1), min(y1+1, h-1)
                    
                    dx, dy = x_dist - x1, y_dist - y1
                    
                    # Interpolate for each channel
                    for c in range(3):
                        val = (image[y1, x1, c] * (1-dx) * (1-dy) +
                               image[y1, x2, c] * dx * (1-dy) +
                               image[y2, x1, c] * (1-dx) * dy +
                               image[y2, x2, c] * dx * dy)
                        distorted[y, x, c] = val
        
        return distorted
    
    def capture_depth(self):
        """Capture depth data with realistic sensor noise"""
        
        depth_data = self.sd_helper.get_depth_data("/humanoid/head_camera")
        
        # Add sensor noise
        noise = np.random.normal(0, 0.01, depth_data.shape)
        depth_data = depth_data + noise
        
        # Clip to valid range
        depth_data = np.clip(depth_data, 0.1, 10.0)
        
        return depth_data
    
    def capture_semantic(self):
        """Capture semantic segmentation data"""
        
        semantic_data = self.sd_helper.get_semantic_data("/humanoid/head_camera")
        
        return semantic_data
    
    def generate_annotations(self):
        """Generate annotations for training"""
        
        annotations = {
            "objects": self.detect_objects(),
            "poses": self.get_robot_poses(),
            "scene_info": self.get_scene_info()
        }
        
        return annotations
    
    def save_sample(self, index, rgb, depth, semantic, annotations):
        """Save training sample"""
        
        # Save RGB image
        cv2.imwrite(f"{self.output_dir}/rgb/sample_{index:06d}.png", rgb)
        
        # Save depth (convert to 16-bit)
        depth_16bit = (depth * 1000).astype(np.uint16)
        cv2.imwrite(f"{self.output_dir}/depth/sample_{index:06d}.png", depth_16bit)
        
        # Save semantic
        cv2.imwrite(f"{self.output_dir}/semantic/sample_{index:06d}.png", semantic)
        
        # Save annotations as JSON
        import json
        with open(f"{self.output_dir}/annotations/sample_{index:06d}.json", "w") as f:
            json.dump(annotations, f, indent=2)

# Generate synthetic data
generator = SyntheticDataGenerator()
generator.generate_training_data(num_samples=1000)
```

## 5.3 Sim-to-Real Pipeline

### Domain Randomization

```python
# domain_randomization.py
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim
from pxr import Usd, UsdGeom, Gf

class DomainRandomization:
    def __init__(self):
        self.world = World()
        self.randomization_params = {
            "lighting": {"intensity_range": (500, 5000), "color_temp_range": (3000, 7000)},
            "materials": {"roughness_range": (0.1, 0.9), "metallic_range": (0.0, 1.0)},
            "physics": {"gravity_range": (-10.5, -9.5), "friction_range": (0.5, 1.0)},
            "camera": {"noise_range": (0.001, 0.05), "distortion_range": (0.0, 0.2)}
        }
    
    def randomize_all(self):
        """Apply all domain randomization"""
        self.randomize_lighting()
        self.randomize_materials()
        self.randomize_physics()
        self.randomize_camera()
    
    def randomize_lighting(self):
        """Randomize lighting conditions"""
        
        # Random dome light
        dome_light = self.stage.GetPrimAtPath("/DomeLight")
        if dome_light:
            intensity = np.random.uniform(*self.randomization_params["lighting"]["intensity_range"])
            dome_light.GetAttribute("intensity").Set(intensity)
        
        # Random area lights
        for i in range(3):
            light_path = f"/AreaLight{i+1}"
            light = self.stage.GetPrimAtPath(light_path)
            if light:
                # Random intensity
                intensity = np.random.uniform(1000, 10000)
                light.GetAttribute("intensity").Set(intensity)
                
                # Random color
                temp = np.random.uniform(*self.randomization_params["lighting"]["color_temp_range"])
                color = self.kelvin_to_rgb(temp)
                light.GetAttribute("color").Set(Gf.Vec3f(*color))
    
    def randomize_materials(self):
        """Randomize material properties"""
        
        materials = ["/Looks/WoodMaterial", "/Looks/MetalMaterial", "/Looks/PlasticMaterial"]
        
        for mat_path in materials:
            material = self.stage.GetPrimAtPath(mat_path)
            if material:
                # Random roughness
                roughness = np.random.uniform(*self.randomization_params["materials"]["roughness_range"])
                material.GetAttribute("roughness").Set(roughness)
                
                # Random metallic
                metallic = np.random.uniform(*self.randomization_params["materials"]["metallic_range"])
                material.GetAttribute("metallic").Set(metallic)
    
    def randomize_physics(self):
        """Randomize physics parameters"""
        
        # Random gravity
        gravity = np.random.uniform(*self.randomization_params["physics"]["gravity_range"])
        self.world.set_gravity(gravity)
        
        # Random friction for ground plane
        ground = self.stage.GetPrimAtPath("/ground_plane")
        if ground:
            friction = np.random.uniform(*self.randomization_params["physics"]["friction_range"])
            # Apply friction to collision material
    
    def randomize_camera(self):
        """Randomize camera parameters"""
        
        camera = self.stage.GetPrimAtPath("/humanoid/head_camera")
        if camera:
            # Random noise
            noise = np.random.uniform(*self.randomization_params["camera"]["noise_range"])
            # Apply noise to camera sensor
            
            # Random distortion
            distortion = np.random.uniform(*self.randomization_params["camera"]["distortion_range"])
            # Apply lens distortion
```

### Transfer Learning Pipeline

```python
# sim_to_real_transfer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class Sim2RealTransfer:
    def __init__(self, sim_model_path, real_dataset_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load simulation-trained model
        self.sim_model = self.load_model(sim_model_path)
        
        # Load real-world dataset
        self.real_dataset = self.load_real_dataset(real_dataset_path)
        
        # Create adaptation model
        self.adaptation_model = self.create_adaptation_model()
        
    def load_model(self, path):
        """Load pre-trained simulation model"""
        model = torch.load(path)
        model.eval()
        return model.to(self.device)
    
    def create_adaptation_model(self):
        """Create domain adaptation network"""
        
        class AdaptationNet(nn.Module):
            def __init__(self, feature_dim=512):
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, feature_dim)
                )
                
                self.domain_classifier = nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
                
                self.task_classifier = nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)  # 10 classes
                )
            
            def forward(self, x):
                features = self.feature_extractor(x)
                domain_pred = self.domain_classifier(features)
                task_pred = self.task_classifier(features)
                return domain_pred, task_pred
        
        return AdaptationNet().to(self.device)
    
    def train_adaptation(self, epochs=100):
        """Train domain adaptation"""
        
        optimizer = torch.optim.Adam(self.adaptation_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            total_loss = 0
            domain_loss = 0
            task_loss = 0
            
            for batch in self.real_dataset:
                # Get simulation features
                sim_features = self.get_sim_features(batch)
                
                # Get real features
                real_features = self.adaptation_model.feature_extractor(batch)
                
                # Domain adversarial training
                domain_pred, task_pred = self.adaptation_model(batch)
                
                # Compute losses
                d_loss = self.domain_loss(domain_pred, real_features)
                t_loss = self.task_loss(task_pred, batch.labels)
                
                # Combined loss
                loss = d_loss + t_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                domain_loss += d_loss.item()
                task_loss += t_loss.item()
            
            print(f"Epoch {epoch}: Total Loss: {total_loss:.4f}, "
                  f"Domain Loss: {domain_loss:.4f}, Task Loss: {task_loss:.4f}")
    
    def evaluate_transfer(self, test_dataset):
        """Evaluate transfer performance"""
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_dataset:
                _, task_pred = self.adaptation_model(batch)
                predicted = torch.argmax(task_pred, dim=1)
                total += batch.labels.size(0)
                correct += (predicted == batch.labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Transfer Accuracy: {accuracy:.2f}%")
        return accuracy
```

## 5.4 Training AI Models in Simulation

### Reinforcement Learning Environment

```python
# rl_environment.py
import gym
from gym import spaces
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path

class HumanoidRLEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.world = World()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)
        
        # Initialize humanoid
        self.humanoid = self.create_humanoid()
        
        # Target position
        self.target_pos = np.array([2.0, 0.0, 0.0])
        
        # Episode parameters
        self.max_steps = 1000
        self.current_step = 0
        
    def create_humanoid(self):
        """Create humanoid robot in simulation"""
        
        from omni.isaac.core.robots import Robot
        
        # Load humanoid robot
        humanoid = Robot(
            prim_path="/humanoid",
            usd_path=f"{get_assets_root_path()}/Isaac/Robots/Humanoid/humanoid.usd",
            name="humanoid"
        )
        
        # Add to world
        self.world.add_robot(humanoid)
        
        return humanoid
    
    def reset(self):
        """Reset environment"""
        
        # Reset humanoid position
        self.humanoid.set_world_pose(position=[0, 0, 1.0])
        
        # Reset joint positions
        initial_joints = np.zeros(12)
        self.humanoid.set_joint_positions(initial_joints)
        
        # Random target position
        self.target_pos = np.array([
            np.random.uniform(-3, 3),
            np.random.uniform(-3, 3),
            0.0
        ])
        
        # Reset step counter
        self.current_step = 0
        
        return self.get_observation()
    
    def step(self, action):
        """Execute action in environment"""
        
        # Apply joint commands
        self.humanoid.set_joint_positions(action)
        
        # Step simulation
        self.world.step(render=True)
        
        # Get observation
        obs = self.get_observation()
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check if done
        done = self.current_step >= self.max_steps or self.check_success()
        
        self.current_step += 1
        
        return obs, reward, done, {}
    
    def get_observation(self):
        """Get current observation"""
        
        # Get robot pose
        pose = self.humanoid.get_world_pose()
        
        # Get joint positions and velocities
        joint_pos = self.humanoid.get_joint_positions()
        joint_vel = self.humanoid.get_joint_velocities()
        
        # Get target relative position
        robot_pos = np.array(pose[0])
        target_rel = self.target_pos - robot_pos
        
        # Combine observations
        obs = np.concatenate([
            pose[0],  # position (3)
            pose[1],  # orientation (4)
            joint_pos,  # joint positions (12)
            joint_vel,  # joint velocities (12)
            target_rel,  # target relative position (3)
            [self.current_step / self.max_steps]  # progress (1)
        ])
        
        return obs.astype(np.float32)
    
    def calculate_reward(self):
        """Calculate reward based on current state"""
        
        # Distance to target
        robot_pos = np.array(self.humanoid.get_world_pose()[0])
        distance = np.linalg.norm(self.target_pos - robot_pos)
        
        # Reward components
        distance_reward = -distance * 0.1
        
        # Energy penalty
        joint_vel = self.humanoid.get_joint_velocities()
        energy_penalty = -np.sum(joint_vel**2) * 0.001
        
        # Upright bonus
        orientation = self.humanoid.get_world_pose()[1]
        upright_bonus = orientation[3]  # quaternion w component
        
        # Total reward
        reward = distance_reward + energy_penalty + upright_bonus * 0.1
        
        return reward
    
    def check_success(self):
        """Check if task is completed"""
        
        robot_pos = np.array(self.humanoid.get_world_pose()[0])
        distance = np.linalg.norm(self.target_pos - robot_pos)
        
        return distance < 0.5  # Success if within 0.5m of target
```

### Training Loop

```python
# train_humanoid_rl.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from rl_environment import HumanoidRLEnvironment

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        ).to(self.device)
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=3e-4
        )
        
        # PPO parameters
        self.clip_ratio = 0.2
        self.gamma = 0.99
        self.lam = 0.95
        self.epochs = 10
        self.batch_size = 64
    
    def get_action(self, state):
        """Get action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state)
            value = self.critic(state)
        
        # Add exploration noise
        action = action.cpu().numpy()[0]
        action += np.random.normal(0, 0.1, size=action.shape)
        action = np.clip(action, -1.0, 1.0)
        
        return action, value.item()
    
    def update(self, states, actions, rewards, dones, values):
        """Update policy using PPO"""
        
        # Calculate advantages
        advantages = self.calculate_advantages(rewards, values, dones)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(rewards).to(self.device)
        
        # PPO update
        for _ in range(self.epochs):
            # Get current policy
            current_actions = self.actor(states)
            current_values = self.critic(states)
            
            # Calculate ratio
            ratio = torch.exp(current_actions - actions)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = nn.MSELoss()(current_values.squeeze(), returns)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def calculate_advantages(self, rewards, values, dones):
        """Calculate GAE advantages"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return advantages

def train_humanoid():
    """Train humanoid robot using RL"""
    
    env = HumanoidRLEnvironment()
    agent = PPOAgent(state_dim=48, action_dim=12)
    
    # Training loop
    for episode in range(1000):
        state = env.reset()
        
        # Collect trajectory
        states, actions, rewards, dones, values = [], [], [], [], []
        
        for step in range(1000):
            action, value = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            
            state = next_state
            
            if done:
                break
        
        # Update agent
        agent.update(states, actions, rewards, dones, values)
        
        # Log progress
        if episode % 10 == 0:
            avg_reward = np.mean(rewards)
            print(f"Episode {episode}: Average Reward: {avg_reward:.2f}")
            
            # Save checkpoint
            torch.save(agent.actor.state_dict(), f"humanoid_actor_episode_{episode}.pth")
            torch.save(agent.critic.state_dict(), f"humanoid_critic_episode_{episode}.pth")

if __name__ == "__main__":
    train_humanoid()
```

---

**Next Chapter**: In Chapter 6, we'll explore robot vision systems using RealSense cameras and OpenCV for perception.

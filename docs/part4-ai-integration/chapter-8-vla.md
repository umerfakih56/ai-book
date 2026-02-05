---
title: "Chapter 8: Vision-Language-Action (VLA) Models"
sidebar_position: 8
---

# Chapter 8: Vision-Language-Action (VLA) Models

## 8.1 Bridging LLMs and Robotics

### What are VLA Models?

Vision-Language-Action (VLA) models are AI systems that can process visual information, understand natural language commands, and generate appropriate robot actions. They represent the convergence of computer vision, natural language processing, and robotics.

### Key Components

1. **Vision Encoder**: Processes camera images to understand the environment
2. **Language Model**: Interprets natural language commands
3. **Action Decoder**: Translates understanding into robot actions
4. **Multimodal Fusion**: Combines vision and language information

### Popular VLA Architectures

- **RT-1/RT-2**: Google's Robotics Transformer models
- **PaLM-E**: Google's embodied language model
- **Flamingo**: DeepMind's multimodal model
- **CLIP-RT**: Combines CLIP with robotics transformers

### VLA Model Architecture

```python
# vla_model.py
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
import numpy as np

class VisionLanguageActionModel(nn.Module):
    def __init__(self, vision_backbone='resnet50', language_model='bert-base-uncased', 
                 action_dim=12, hidden_dim=512):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Vision encoder
        if vision_backbone == 'resnet50':
            self.vision_encoder = models.resnet50(pretrained=True)
            self.vision_encoder.fc = nn.Identity()
            vision_dim = 2048
        else:
            raise ValueError(f"Unsupported vision backbone: {vision_backbone}")
        
        # Language model
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.language_model = AutoModel.from_pretrained(language_model)
        language_dim = self.language_model.config.hidden_size
        
        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Multimodal fusion
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Normalize actions to [-1, 1]
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
    def forward(self, images, text_commands):
        """
        Args:
            images: Batch of images [B, C, H, W]
            text_commands: List of text commands
        Returns:
            actions: Predicted robot actions [B, action_dim]
        """
        
        # Encode vision
        vision_features = self.encode_vision(images)  # [B, hidden_dim]
        
        # Encode language
        language_features = self.encode_language(text_commands)  # [B, seq_len, hidden_dim]
        
        # Fuse multimodal information
        fused_features = self.fuse_vision_language(vision_features, language_features)
        
        # Decode actions
        actions = self.action_decoder(fused_features)
        
        return actions
    
    def encode_vision(self, images):
        """Encode images to feature vectors"""
        
        # Extract features
        with torch.no_grad():
            vision_features = self.vision_encoder(images)  # [B, 2048]
        
        # Project to hidden dimension
        vision_features = self.vision_proj(vision_features)  # [B, hidden_dim]
        
        # Add positional encoding
        vision_features = self.pos_encoding(vision_features.unsqueeze(1)).squeeze(1)
        
        return vision_features
    
    def encode_language(self, text_commands):
        """Encode text commands to feature vectors"""
        
        # Tokenize text
        encoded_inputs = self.tokenizer(
            text_commands, 
            padding=True, 
            truncation=True, 
            return_tensors='pt',
            max_length=512
        )
        
        # Get language features
        with torch.no_grad():
            language_outputs = self.language_model(**encoded_inputs)
            language_features = language_outputs.last_hidden_state  # [B, seq_len, hidden_dim]
        
        # Project to hidden dimension
        language_features = self.language_proj(language_features)
        
        # Add positional encoding
        language_features = self.pos_encoding(language_features)
        
        return language_features
    
    def fuse_vision_language(self, vision_features, language_features):
        """Fuse vision and language features using attention"""
        
        # Vision as query, language as key and value
        batch_size = vision_features.size(0)
        
        # Expand vision features for multi-head attention
        vision_query = vision_features.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Apply attention
        fused_features, _ = self.fusion_layer(
            vision_query, 
            language_features, 
            language_features
        )  # [B, 1, hidden_dim]
        
        # Squeeze and return
        fused_features = fused_features.squeeze(1)  # [B, hidden_dim]
        
        return fused_features

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
```

## 8.2 Voice Commands with Whisper & ROS 2

### Whisper Integration

```python
# whisper_ros2.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import whisper
import numpy as np
import torch
import io
import wave

class WhisperROS2Node(Node):
    def __init__(self, model_name='base'):
        super().__init__('whisper_asr')
        
        # Initialize Whisper model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = whisper.load_model(model_name, device=self.device)
        
        # Audio parameters
        self.sample_rate = 16000
        self.audio_buffer = []
        self.is_recording = False
        
        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData, '/audio/audio', self.audio_callback, 10)
        
        # Publishers
        self.text_pub = self.create_publisher(
            String, '/voice/text', 10)
        
        # Timer for processing audio
        self.timer = self.create_timer(1.0, self.process_audio)
        
        self.get_logger().info(f'Whisper ASR node initialized with model: {model_name}')
    
    def audio_callback(self, msg):
        """Handle audio data"""
        
        if self.is_recording:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.int16)
            
            # Resample if necessary (assuming input is 16kHz)
            if self.sample_rate != 16000:
                audio_data = self.resample_audio(audio_data, msg.sample_rate, 16000)
            
            self.audio_buffer.extend(audio_data.tolist())
    
    def process_audio(self):
        """Process accumulated audio for transcription"""
        
        if len(self.audio_buffer) > 0 and self.is_recording:
            # Convert buffer to numpy array
            audio_array = np.array(self.audio_buffer, dtype=np.float32) / 32768.0
            
            # Transcribe with Whisper
            result = self.model.transcribe(audio_array)
            
            # Publish transcription
            if result['text'].strip():
                text_msg = String()
                text_msg.data = result['text']
                self.text_pub.publish(text_msg)
                
                self.get_logger().info(f'Transcribed: {result["text"]}')
            
            # Clear buffer
            self.audio_buffer = []
    
    def resample_audio(self, audio_data, orig_sr, target_sr):
        """Resample audio to target sample rate"""
        
        # Simple resampling (for production, use librosa or similar)
        ratio = target_sr / orig_sr
        new_length = int(len(audio_data) * ratio)
        
        # Linear interpolation
        indices = np.linspace(0, len(audio_data) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
        
        return resampled.astype(np.int16)
    
    def start_recording(self):
        """Start voice recording"""
        self.is_recording = True
        self.audio_buffer = []
        self.get_logger().info('Started voice recording')
    
    def stop_recording(self):
        """Stop voice recording"""
        self.is_recording = False
        self.get_logger().info('Stopped voice recording')

def main(args=None):
    rclpy.init(args=args)
    
    whisper_node = WhisperROS2Node()
    
    try:
        rclpy.spin(whisper_node)
    except KeyboardInterrupt:
        pass
    finally:
        whisper_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Voice Command Processor

```python
# voice_command_processor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import re
import numpy as np

class VoiceCommandProcessor(Node):
    def __init__(self):
        super().__init__('voice_command_processor')
        
        # Command patterns
        self.command_patterns = {
            'move_forward': r'(move|go|walk) (forward|ahead)',
            'move_backward': r'(move|go|walk) (backward|back)',
            'turn_left': r'turn left',
            'turn_right': r'turn right',
            'stop': r'stop|halt',
            'pick_up': r'pick up|grab|take',
            'put_down': r'put down|place|release',
            'wave': r'wave|hello',
            'sit': r'sit|sit down',
            'stand': r'stand|stand up',
            'walk_to': r'walk to|go to|move to',
            'look_at': r'look at|watch'
        }
        
        # Subscribers
        self.text_sub = self.create_subscription(
            String, '/voice/text', self.text_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        
        self.joint_traj_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory', 10)
        
        self.get_logger().info('Voice command processor initialized')
    
    def text_callback(self, msg):
        """Process transcribed text"""
        
        text = msg.data.lower().strip()
        self.get_logger().info(f'Processing command: {text}')
        
        # Parse command
        command, params = self.parse_command(text)
        
        if command:
            self.execute_command(command, params)
        else:
            self.get_logger().warn(f'Unknown command: {text}')
    
    def parse_command(self, text):
        """Parse text into command and parameters"""
        
        for command_name, pattern in self.command_patterns.items():
            match = re.search(pattern, text)
            if match:
                # Extract additional parameters
                params = self.extract_parameters(text, command_name)
                return command_name, params
        
        return None, None
    
    def extract_parameters(self, text, command_name):
        """Extract parameters from command text"""
        
        params = {}
        
        if command_name == 'walk_to':
            # Extract location
            location_match = re.search(r'(to|at) (\w+)', text)
            if location_match:
                params['location'] = location_match.group(2)
        
        elif command_name == 'look_at':
            # Extract object
            object_match = re.search(r'at (\w+)', text)
            if object_match:
                params['object'] = object_match.group(1)
        
        return params
    
    def execute_command(self, command, params):
        """Execute parsed command"""
        
        if command == 'move_forward':
            self.move_robot(0.5, 0.0)
        
        elif command == 'move_backward':
            self.move_robot(-0.5, 0.0)
        
        elif command == 'turn_left':
            self.move_robot(0.0, 0.5)
        
        elif command == 'turn_right':
            self.move_robot(0.0, -0.5)
        
        elif command == 'stop':
            self.move_robot(0.0, 0.0)
        
        elif command == 'pick_up':
            self.pick_up_object()
        
        elif command == 'put_down':
            self.put_down_object()
        
        elif command == 'wave':
            self.wave_hand()
        
        elif command == 'sit':
            self.sit_down()
        
        elif command == 'stand':
            self.stand_up()
        
        elif command == 'walk_to':
            self.walk_to_location(params.get('location'))
        
        elif command == 'look_at':
            self.look_at_object(params.get('object'))
    
    def move_robot(self, linear_x, angular_z):
        """Send movement command"""
        
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info(f'Moving: linear={linear_x}, angular={angular_z}')
    
    def pick_up_object(self):
        """Execute pick up motion"""
        
        trajectory = self.create_pickup_trajectory()
        self.joint_traj_pub.publish(trajectory)
        
        self.get_logger().info('Executing pick up motion')
    
    def put_down_object(self):
        """Execute put down motion"""
        
        trajectory = self.create_putdown_trajectory()
        self.joint_traj_pub.publish(trajectory)
        
        self.get_logger().info('Executing put down motion')
    
    def wave_hand(self):
        """Execute waving motion"""
        
        trajectory = self.create_wave_trajectory()
        self.joint_traj_pub.publish(trajectory)
        
        self.get_logger().info('Executing wave motion')
    
    def sit_down(self):
        """Execute sitting motion"""
        
        trajectory = self.create_sit_trajectory()
        self.joint_traj_pub.publish(trajectory)
        
        self.get_logger().info('Executing sit down motion')
    
    def stand_up(self):
        """Execute standing motion"""
        
        trajectory = self.create_stand_trajectory()
        self.joint_traj_pub.publish(trajectory)
        
        self.get_logger().info('Executing stand up motion')
    
    def walk_to_location(self, location):
        """Navigate to specific location"""
        
        # This would integrate with navigation system
        self.get_logger().info(f'Walking to location: {location}')
    
    def look_at_object(self, object_name):
        """Look at specific object"""
        
        # This would integrate with vision system
        self.get_logger().info(f'Looking at object: {object_name}')
    
    def create_pickup_trajectory(self):
        """Create joint trajectory for picking up"""
        
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = [
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow',
            'left_wrist_pitch', 'left_gripper'
        ]
        
        # Add trajectory points
        points = []
        
        # Pre-grasp position
        point1 = JointTrajectoryPoint()
        point1.positions = [0.5, 0.3, -0.8, 0.2, 0.0]
        point1.velocities = [0.0] * 5
        point1.time_from_start.sec = 1
        points.append(point1)
        
        # Grasp position
        point2 = JointTrajectoryPoint()
        point2.positions = [0.6, 0.4, -1.0, 0.3, 0.5]
        point2.velocities = [0.0] * 5
        point2.time_from_start.sec = 2
        points.append(point2)
        
        trajectory.points = points
        return trajectory
    
    def create_wave_trajectory(self):
        """Create joint trajectory for waving"""
        
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = ['right_shoulder_pitch', 'right_elbow']
        
        points = []
        
        # Wave motion (simplified)
        for i in range(3):
            point = JointTrajectoryPoint()
            point.positions = [0.3, (-1)**i * 0.8]
            point.velocities = [0.0, 0.0]
            point.time_from_start.sec = i + 1
            points.append(point)
        
        trajectory.points = points
        return trajectory

def main(args=None):
    rclpy.init(args=args)
    
    processor = VoiceCommandProcessor()
    
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 8.3 Natural Language to Action Sequences

### Task Planning with LLMs

```python
# task_planner.py
import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import json
import re

class LLMTaskPlanner(Node):
    def __init__(self, api_key=None):
        super().__init__('llm_task_planner')
        
        # OpenAI configuration
        if api_key:
            openai.api_key = api_key
        
        # Robot capabilities
        self.capabilities = {
            'navigation': ['move_forward', 'turn_left', 'turn_right', 'stop'],
            'manipulation': ['pick_up', 'put_down', 'wave'],
            'locomotion': ['sit', 'stand', 'walk'],
            'perception': ['look_at', 'find_object']
        }
        
        # Environment knowledge
        self.environment = {
            'locations': ['kitchen', 'living_room', 'bedroom', 'bathroom'],
            'objects': ['cup', 'bottle', 'book', 'phone', 'remote'],
            'furniture': ['table', 'chair', 'sofa', 'bed']
        }
        
        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/voice/text', self.command_callback, 10)
        
        # Publishers
        self.action_pub = self.create_publisher(
            String, '/planner/action_sequence', 10)
        
        self.get_logger().info('LLM Task Planner initialized')
    
    def command_callback(self, msg):
        """Process natural language command"""
        
        command = msg.data
        self.get_logger().info(f'Planning command: {command}')
        
        # Generate action sequence
        action_sequence = self.plan_actions(command)
        
        if action_sequence:
            # Publish action sequence
            action_msg = String()
            action_msg.data = json.dumps(action_sequence)
            self.action_pub.publish(action_msg)
            
            self.get_logger().info(f'Generated action sequence: {len(action_sequence)} steps')
        else:
            self.get_logger().warn('Failed to generate action sequence')
    
    def plan_actions(self, command):
        """Generate action sequence using LLM"""
        
        # Create prompt for LLM
        prompt = self.create_planning_prompt(command)
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a humanoid robot task planner. Generate step-by-step actions to accomplish tasks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Parse response
            action_sequence = self.parse_llm_response(response.choices[0].message.content)
            
            return action_sequence
            
        except Exception as e:
            self.get_logger().error(f'LLM API error: {e}')
            return None
    
    def create_planning_prompt(self, command):
        """Create planning prompt for LLM"""
        
        capabilities_str = json.dumps(self.capabilities, indent=2)
        environment_str = json.dumps(self.environment, indent=2)
        
        prompt = f"""
Given the command: "{command}"

Robot Capabilities:
{capabilities_str}

Environment:
{environment_str}

Generate a step-by-step action sequence to accomplish this command.
Each step should be a JSON object with 'action' and 'parameters' fields.
Available actions are: {list(set([action for actions in self.capabilities.values() for action in actions]))}

Example format:
[
    {{"action": "turn_left", "parameters": {{"angle": 90}}},
     {"action": "move_forward", "parameters": {{"distance": 2.0}}},
     {"action": "pick_up", "parameters": {{"object": "cup"}}}}
]

Generate only the JSON array, no additional text.
"""
        
        return prompt
    
    def parse_llm_response(self, response_text):
        """Parse LLM response into action sequence"""
        
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                action_sequence = json.loads(json_str)
                
                # Validate actions
                validated_sequence = []
                for action in action_sequence:
                    if self.validate_action(action):
                        validated_sequence.append(action)
                    else:
                        self.get_logger().warn(f'Invalid action: {action}')
                
                return validated_sequence
            
        except Exception as e:
            self.get_logger().error(f'Error parsing LLM response: {e}')
        
        return None
    
    def validate_action(self, action):
        """Validate action against robot capabilities"""
        
        if 'action' not in action:
            return False
        
        action_name = action['action']
        
        # Check if action is in capabilities
        for category, actions in self.capabilities.items():
            if action_name in actions:
                return True
        
        return False

class ActionExecutor(Node):
    """Execute action sequences"""
    
    def __init__(self):
        super().__init__('action_executor')
        
        # Subscribers
        self.action_sub = self.create_subscription(
            String, '/planner/action_sequence', self.action_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_traj_pub = self.create_publisher(JointTrajectory, '/joint_trajectory', 10)
        
        # Execution state
        self.current_sequence = None
        self.current_step = 0
        
        # Timer for execution
        self.timer = self.create_timer(0.1, self.execute_step)
        
        self.get_logger().info('Action Executor initialized')
    
    def action_callback(self, msg):
        """Handle new action sequence"""
        
        try:
            action_sequence = json.loads(msg.data)
            self.current_sequence = action_sequence
            self.current_step = 0
            
            self.get_logger().info(f'Received action sequence with {len(action_sequence)} steps')
            
        except Exception as e:
            self.get_logger().error(f'Error parsing action sequence: {e}')
    
    def execute_step(self):
        """Execute current step in action sequence"""
        
        if self.current_sequence is None:
            return
        
        if self.current_step >= len(self.current_sequence):
            self.get_logger().info('Action sequence completed')
            self.current_sequence = None
            return
        
        # Get current action
        action = self.current_sequence[self.current_step]
        
        # Execute action
        self.execute_single_action(action)
        
        # Move to next step
        self.current_step += 1
    
    def execute_single_action(self, action):
        """Execute a single action"""
        
        action_name = action.get('action', '')
        parameters = action.get('parameters', {})
        
        self.get_logger().info(f'Executing: {action_name} with params: {parameters}')
        
        if action_name == 'move_forward':
            distance = parameters.get('distance', 1.0)
            self.move_forward(distance)
        
        elif action_name == 'turn_left':
            angle = parameters.get('angle', 90)
            self.turn(angle)
        
        elif action_name == 'turn_right':
            angle = parameters.get('angle', 90)
            self.turn(-angle)
        
        elif action_name == 'pick_up':
            object_name = parameters.get('object', '')
            self.pick_up(object_name)
        
        elif action_name == 'put_down':
            object_name = parameters.get('object', '')
            self.put_down(object_name)
        
        # Add more actions as needed
    
    def move_forward(self, distance):
        """Move forward for specified distance"""
        
        cmd = Twist()
        cmd.linear.x = 0.5  # Constant speed
        
        # Calculate duration based on distance
        duration = distance / 0.5
        
        # Send command (simplified - would need timing control)
        self.cmd_vel_pub.publish(cmd)
    
    def turn(self, angle):
        """Turn by specified angle"""
        
        cmd = Twist()
        cmd.angular.z = 0.5 if angle > 0 else -0.5
        
        # Calculate duration based on angle
        duration = abs(angle) / 0.5
        
        # Send command
        self.cmd_vel_pub.publish(cmd)
    
    def pick_up(self, object_name):
        """Pick up specified object"""
        
        # Generate pickup trajectory
        trajectory = self.create_pickup_trajectory()
        self.joint_traj_pub.publish(trajectory)

def main(args=None):
    rclpy.init(args=args)
    
    # Create executor for multi-threaded spinning
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    
    # Create nodes
    planner = LLMTaskPlanner()
    executor = ActionExecutor()
    
    # Add nodes to executor
    executor.add_node(planner)
    executor.add_node(executor)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        executor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 8.4 Cognitive Planning with GPT Models

### Advanced Cognitive Planning

```python
# cognitive_planner.py
import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json
import numpy as np
from collections import deque

class CognitivePlanner(Node):
    def __init__(self, api_key=None):
        super().__init__('cognitive_planner')
        
        # OpenAI configuration
        if api_key:
            openai.api_key = api_key
        
        # Memory systems
        self.working_memory = deque(maxlen=10)  # Short-term memory
        self.long_term_memory = {}  # Knowledge base
        self.episodic_memory = []  # Experience memory
        
        # Planning state
        self.current_goal = None
        self.current_plan = None
        self.plan_step = 0
        
        # World model
        self.world_state = {
            'robot_position': [0, 0, 0],
            'robot_orientation': [0, 0, 0, 1],
            'objects': {},
            'locations': {},
            'time': 0
        }
        
        # Subscribers
        self.goal_sub = self.create_subscription(
            String, '/cognitive/goal', self.goal_callback, 10)
        
        self.perception_sub = self.create_subscription(
            String, '/perception/world_state', self.perception_callback, 10)
        
        # Publishers
        self.plan_pub = self.create_publisher(
            String, '/cognitive/plan', 10)
        
        self.action_pub = self.create_publisher(
            String, '/cognitive/action', 10)
        
        # Timer for planning
        self.timer = self.create_timer(1.0, self.planning_loop)
        
        self.get_logger().info('Cognitive Planner initialized')
    
    def goal_callback(self, msg):
        """Handle new goal"""
        
        goal = msg.data
        self.get_logger().info(f'New goal: {goal}')
        
        # Add to working memory
        self.working_memory.append({
            'type': 'goal',
            'content': goal,
            'time': self.world_state['time']
        })
        
        # Generate plan
        self.generate_plan(goal)
    
    def perception_callback(self, msg):
        """Update world state from perception"""
        
        try:
            world_update = json.loads(msg.data)
            self.world_state.update(world_update)
            self.world_state['time'] += 1
            
            # Add to episodic memory
            self.episodic_memory.append({
                'time': self.world_state['time'],
                'state': self.world_state.copy()
            })
            
        except Exception as e:
            self.get_logger().error(f'Error updating world state: {e}')
    
    def generate_plan(self, goal):
        """Generate cognitive plan using GPT"""
        
        # Create planning prompt
        prompt = self.create_cognitive_prompt(goal)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an advanced cognitive planner for a humanoid robot. Create detailed plans considering perception, memory, and reasoning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse plan
            plan = self.parse_cognitive_plan(response.choices[0].message.content)
            
            if plan:
                self.current_plan = plan
                self.plan_step = 0
                
                # Publish plan
                plan_msg = String()
                plan_msg.data = json.dumps(plan)
                self.plan_pub.publish(plan_msg)
                
                self.get_logger().info(f'Generated cognitive plan with {len(plan)} steps')
            
        except Exception as e:
            self.get_logger().error(f'Cognitive planning error: {e}')
    
    def create_cognitive_prompt(self, goal):
        """Create cognitive planning prompt"""
        
        working_memory_str = json.dumps(list(self.working_memory), indent=2)
        world_state_str = json.dumps(self.world_state, indent=2)
        
        prompt = f"""
You are a cognitive planner for a humanoid robot. Create a detailed plan to achieve the goal.

Current Goal: {goal}

Working Memory (recent events):
{working_memory_str}

Current World State:
{world_state_str}

Available Actions:
- navigate_to(location)
- pick_up(object)
- put_down(object, location)
- look_at(object)
- ask_human(question)
- wait(seconds)

Create a step-by-step plan that includes:
1. Perception requirements
2. Action sequences
3. Contingency planning
4. Memory updates

Format as JSON array:
[
    {{
        "step": 1,
        "action": "action_name",
        "parameters": {{}},
        "perception": "what to perceive",
        "expected_outcome": "what should happen",
        "contingency": "what to do if it fails"
    }}
]
"""
        
        return prompt
    
    def parse_cognitive_plan(self, response_text):
        """Parse cognitive plan from LLM response"""
        
        try:
            # Extract JSON array
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                plan = json.loads(json_str)
                
                # Validate plan
                validated_plan = []
                for step in plan:
                    if self.validate_plan_step(step):
                        validated_plan.append(step)
                
                return validated_plan
        
        except Exception as e:
            self.get_logger().error(f'Error parsing cognitive plan: {e}')
        
        return None
    
    def validate_plan_step(self, step):
        """Validate plan step"""
        
        required_fields = ['step', 'action', 'parameters', 'perception', 'expected_outcome']
        return all(field in step for field in required_fields)
    
    def planning_loop(self):
        """Main planning loop"""
        
        if self.current_plan is None:
            return
        
        if self.plan_step >= len(self.current_plan):
            self.get_logger().info('Plan completed')
            self.current_plan = None
            return
        
        # Get current step
        current_step = self.current_plan[self.plan_step]
        
        # Check if step can be executed
        if self.can_execute_step(current_step):
            self.execute_plan_step(current_step)
            self.plan_step += 1
        else:
            # Handle contingency
            self.handle_contingency(current_step)
    
    def can_execute_step(self, step):
        """Check if step can be executed"""
        
        # Check perception requirements
        perception = step.get('perception', '')
        if perception and not self.check_perception(perception):
            return False
        
        # Check world state
        action = step.get('action', '')
        parameters = step.get('parameters', {})
        
        if action == 'pick_up':
            object_name = parameters.get('object')
            if object_name not in self.world_state.get('objects', {}):
                return False
        
        return True
    
    def check_perception(self, perception_requirement):
        """Check if perception requirement is met"""
        
        # Simplified perception check
        if 'object' in perception_requirement:
            return len(self.world_state.get('objects', {})) > 0
        
        return True
    
    def execute_plan_step(self, step):
        """Execute plan step"""
        
        action = step.get('action', '')
        parameters = step.get('parameters', {})
        
        # Create action message
        action_msg = String()
        action_msg.data = json.dumps({
            'action': action,
            'parameters': parameters
        })
        
        self.action_pub.publish(action_msg)
        
        # Update working memory
        self.working_memory.append({
            'type': 'action',
            'content': f"{action}: {parameters}",
            'time': self.world_state['time']
        })
        
        self.get_logger().info(f'Executing step {step["step"]}: {action}')
    
    def handle_contingency(self, step):
        """Handle plan contingencies"""
        
        contingency = step.get('contingency', '')
        
        if contingency:
            self.get_logger().info(f'Executing contingency: {contingency}')
            
            # Generate new plan for contingency
            self.generate_plan(contingency)
        else:
            # Skip step
            self.plan_step += 1
            self.get_logger().warning('No contingency plan, skipping step')

def main(args=None):
    rclpy.init(args=args)
    
    planner = CognitivePlanner()
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

**Next Chapter**: In Chapter 9, we'll explore Reinforcement Learning for robot control, including training in simulation and deployment on real hardware.

#!/usr/bin/env python3
import yaml
import os
from pathlib import Path

def populate_chapter_content(spec_file, docs_dir):
    """Populate all chapter MDX files with rich content"""
    
    # Load the specification
    with open(spec_file, 'r') as f:
        spec = yaml.safe_load(f)
    
    base_path = Path(docs_dir)
    
    for module in spec['modules']:
        module_id = module['id']
        module_path = base_path / module_id
        
        for i, chapter in enumerate(module['chapters']):
            chapter_num = i + 1
            chapter_file = f"chapter-{chapter_num}.mdx"
            chapter_path = module_path / chapter_file
            
            # Generate content for this chapter
            content = generate_chapter_content(chapter, chapter_num, module, len(module['chapters']))
            
            # Write to file
            with open(chapter_path, 'w') as f:
                f.write(content)
    
    print(f"Populated content for all chapters in {docs_dir}")

def generate_chapter_content(chapter, chapter_num, module, total_chapters):
    """Generate rich content for a single chapter"""
    
    content = f"""---
id: {chapter['id']}
title: {chapter['title']}
sidebar_position: {chapter_num}
duration: {chapter['duration']}
difficulty: {chapter['difficulty']}
---

# {chapter['title']}

{chapter['description']}

## Learning Outcomes

"""
    # Learning Outcomes
    for outcome in chapter['learning_outcomes']:
        content += f"- {outcome}\n"
    
    content += "\n## Prerequisites\n\n"
    for prereq in chapter['prerequisites']:
        content += f"- {prereq}\n"
    
    content += "\n## Introduction\n\n"
    content += generate_introduction(chapter, module)
    
    content += "\n## Key Concepts\n\n"
    content += generate_key_concepts(chapter, module)
    
    content += "\n## Code Examples\n\n"
    content += generate_code_examples(chapter, module)
    
    content += "\n## Practical Exercises\n\n"
    content += generate_practical_exercises(chapter, module)
    
    content += "\n## Assessment\n\n"
    content += generate_assessment(chapter, module)
    
    content += "\n## Resources\n\n"
    for resource in chapter['resources']:
        content += f"- [{resource}]({resource})\n"
    
    # Add additional resources
    content += f"""
- [Official Documentation](https://docs.ros.org/)
- [Video Tutorials](https://www.youtube.com/results?search_query={chapter['title'].replace(' ', '+')})
- [Community Forums](https://answers.ros.org/)
"""
    
    content += "\n## What's Next\n\n"
    if chapter_num < total_chapters:
        next_chapter = chapter_num + 1
        content += f"Continue to [Chapter {next_chapter}](chapter-{next_chapter}.mdx) to learn more about {module['title']}.\n"
    else:
        content += f"Congratulations! You've completed {module['title']}. Move on to the next module to continue your learning journey.\n"
    
    return content

def generate_introduction(chapter, module):
    """Generate introduction content"""
    return f"""Welcome to {chapter['title']}. This chapter will guide you through the essential concepts and practical applications needed to master {chapter['description'].lower()}.

Throughout this chapter, you'll build a strong foundation that will serve you well as you progress through {module['title']}. We'll start with the fundamentals and gradually move to more advanced topics, ensuring you have plenty of hands-on practice along the way.

By the end of this chapter, you'll have the knowledge and skills to confidently work with {chapter['title'].lower()} in real-world robotic applications."""

def generate_key_concepts(chapter, module):
    """Generate key concepts section"""
    concepts = get_concepts_for_chapter(chapter['id'])
    
    content = ""
    for concept in concepts:
        content += f"### {concept['title']}\n\n"
        content += f"{concept['content']}\n\n"
    
    return content

def get_concepts_for_chapter(chapter_id):
    """Get key concepts for a specific chapter"""
    concepts_map = {
        'ch1-1': [
            {
                'title': 'ROS 2 Architecture Overview',
                'content': 'ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides a collection of tools, libraries, and conventions that simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms. The architecture is based on a distributed computing model where different processes communicate over a network using the Data Distribution Service (DDS) standard.'
            },
            {
                'title': 'DDS Communication Middleware',
                'content': 'The Data Distribution Service (DDS) is the underlying communication middleware in ROS 2. It provides a publish-subscribe model that allows different nodes to communicate asynchronously. DDS handles discovery, data serialization, and transport mechanisms, making it ideal for real-time robotic applications where reliability and performance are critical.'
            },
            {
                'title': 'ROS 2 Workspace Structure',
                'content': 'A ROS 2 workspace is a directory where you develop, build, and install ROS 2 packages. The standard structure includes a `src` directory for source code, `build` and `install` directories created during compilation, and a `log` directory for build logs. Understanding this structure is essential for organizing your robotic projects effectively.'
            }
        ],
        'ch1-2': [
            {
                'title': 'Nodes and Topics',
                'content': 'Nodes are the fundamental computational units in ROS 2. They are processes that perform computation and communicate with other nodes using topics. Topics are named channels over which nodes exchange messages. The publish-subscribe model allows for loose coupling between nodes, making the system modular and scalable.'
            },
            {
                'title': 'Services and Actions',
                'content': 'Services provide request-response communication between nodes, suitable for tasks that require immediate feedback. Actions are similar to services but designed for long-running tasks that can be monitored and canceled. Understanding when to use each communication pattern is crucial for designing efficient robotic systems.'
            },
            {
                'title': 'Message Types and Interfaces',
                'content': 'ROS 2 uses strongly-typed message definitions to ensure data consistency across different programming languages. Messages can be simple data types or complex structures with nested fields. Interface definitions (srv for services, action for actions) define the contract between communicating nodes.'
            }
        ],
        # Add more chapter concepts as needed
    }
    
    return concepts_map.get(chapter_id, [
        {
            'title': 'Core Concepts',
            'content': 'This chapter covers fundamental concepts that are essential for understanding the topic. We will explore the theoretical foundations and practical applications through detailed explanations and examples.'
        },
        {
            'title': 'Advanced Topics', 
            'content': 'Building on the fundamentals, we will dive deeper into advanced concepts and best practices. These topics will help you understand the nuances and complexities involved in real-world applications.'
        }
    ])

def generate_code_examples(chapter, module):
    """Generate code examples section"""
    examples = get_code_examples_for_chapter(chapter['id'])
    
    content = ""
    for example in examples:
        content += f"### {example['title']}\n\n"
        content += f"{example['description']}\n\n"
        content += f"```{example['language']}\n{example['code']}\n```\n\n"
        content += f"{example['explanation']}\n\n"
    
    return content

def get_code_examples_for_chapter(chapter_id):
    """Get code examples for a specific chapter"""
    examples_map = {
        'ch1-1': [
            {
                'title': 'Setting up ROS 2 Environment',
                'language': 'bash',
                'description': 'Initialize your ROS 2 workspace and set up the environment',
                'code': '''# Source ROS 2 setup
source /opt/ros/humble/setup.bash

# Create a new workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace
colcon build

# Source the workspace
source install/setup.bash''',
                'explanation': 'This sequence of commands sets up a basic ROS 2 workspace. First, we source the ROS 2 installation, then create the workspace structure, build any packages, and finally source the workspace to make it available in the current shell session.'
            },
            {
                'title': 'Checking ROS 2 Installation',
                'language': 'bash',
                'description': 'Verify that ROS 2 is properly installed and configured',
                'code': '''# Check ROS 2 version
ros2 --version

# List available packages
ros2 pkg list

# Test with a talker node
ros2 run demo_nodes_cpp talker''',
                'explanation': 'These commands help verify your ROS 2 installation. The version check confirms ROS 2 is installed, listing packages shows available functionality, and running the talker node demonstrates the communication system is working.'
            }
        ],
        'ch1-2': [
            {
                'title': 'Creating a Simple Publisher Node',
                'language': 'python',
                'description': 'Python implementation of a basic ROS 2 publisher',
                'code': '''#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.count = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.count}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    publisher = SimplePublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()''',
                'explanation': 'This Python script creates a ROS 2 node that publishes a String message to the "topic" topic every second. The node demonstrates the basic structure of a ROS 2 publisher including initialization, timer creation, and proper cleanup.'
            },
            {
                'title': 'Creating a Subscriber Node',
                'language': 'python',
                'description': 'Python implementation of a ROS 2 subscriber',
                'code': '''#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimpleSubscriber(Node):
    def __init__(self):
        super().__init__('simple_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    subscriber = SimpleSubscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()''',
                'explanation': 'This subscriber node listens for messages on the "topic" topic and logs the received data. It demonstrates the subscription pattern in ROS 2, showing how to create a subscriber and handle incoming messages asynchronously.'
            }
        ]
    }
    
    return examples_map.get(chapter_id, [
        {
            'title': 'Basic Example',
            'language': 'python',
            'description': 'A simple example to get started',
            'code': '# Basic example code\nprint("Hello, World!")',
            'explanation': 'This is a basic example that demonstrates the fundamental concepts covered in this chapter.'
        }
    ])

def generate_practical_exercises(chapter, module):
    """Generate practical exercises section"""
    exercises = get_exercises_for_chapter(chapter['id'])
    
    content = ""
    for i, exercise in enumerate(exercises, 1):
        content += f"### Exercise {i}: {exercise['title']}\n\n"
        content += f"**Objective**: {exercise['objective']}\n\n"
        content += f"**Instructions**: {exercise['instructions']}\n\n"
        content += f"**Expected Results**: {exercise['expected_results']}\n\n"
    
    return content

def get_exercises_for_chapter(chapter_id):
    """Get exercises for a specific chapter"""
    exercises_map = {
        'ch1-1': [
            {
                'title': 'ROS 2 Environment Setup',
                'objective': 'Set up a complete ROS 2 development environment',
                'instructions': '1. Install ROS 2 Humble on your system\n2. Create a new workspace directory\n3. Build the workspace using colcon\n4. Verify the installation by running demo nodes',
                'expected_results': 'You should have a working ROS 2 environment where you can run demo nodes and create new packages.'
            },
            {
                'title': 'Workspace Navigation',
                'objective': 'Master ROS 2 workspace commands and structure',
                'instructions': '1. Navigate through different workspace directories\n2. Use ros2 CLI commands to explore packages\n3. Practice sourcing different workspaces\n4. Create and build a simple package',
                'expected_results': 'You should be comfortable navigating ROS 2 workspaces and using the basic CLI tools.'
            }
        ],
        'ch1-2': [
            {
                'title': 'Publisher-Subscriber Communication',
                'objective': 'Implement basic ROS 2 communication between nodes',
                'instructions': '1. Create a publisher node that publishes custom messages\n2. Create a subscriber node that receives the messages\n3. Run both nodes simultaneously\n4. Verify message transmission using ros2 topic echo',
                'expected_results': 'You should see messages flowing from the publisher to the subscriber in real-time.'
            },
            {
                'title': 'Service Implementation',
                'objective': 'Create and use ROS 2 services for request-response communication',
                'instructions': '1. Define a custom service interface\n2. Implement a service server\n3. Create a service client\n4. Test the service communication',
                'expected_results': 'You should successfully demonstrate request-response communication between ROS 2 nodes.'
            }
        ]
    }
    
    return exercises_map.get(chapter_id, [
        {
            'title': 'Basic Exercise',
            'objective': 'Practice fundamental concepts',
            'instructions': 'Complete the basic exercises covered in this chapter to reinforce your understanding.',
            'expected_results': 'You should be able to apply the concepts learned in practical scenarios.'
        }
    ])

def generate_assessment(chapter, module):
    """Generate assessment section"""
    return f"""This chapter contributes to your overall assessment for {module['title']}. 

**Assessment Type**: Project-based evaluation

**Key Tasks**:
1. Complete all practical exercises successfully
2. Demonstrate understanding of key concepts through code implementation
3. Document your learning process and challenges encountered

**Evaluation Criteria**:
- Code quality and functionality (40%)
- Understanding of concepts (30%)
- Documentation and explanation (20%)
- Innovation and creativity (10%)

**Submission Requirements**:
- Submit your code implementations
- Include a brief explanation of your approach
- Provide screenshots or videos of working examples
- Reflect on challenges and solutions"""

if __name__ == "__main__":
    populate_chapter_content("book-spec.yaml", "docs")

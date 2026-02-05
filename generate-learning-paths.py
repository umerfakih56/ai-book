#!/usr/bin/env python3
import yaml
import os
from pathlib import Path

def generate_learning_paths(spec_file, output_dir):
    """Generate learning path guides and overview"""
    
    # Load the specification
    with open(spec_file, 'r') as f:
        spec = yaml.safe_load(f)
    
    base_path = Path(output_dir)
    base_path.mkdir(exist_ok=True)
    
    # Generate individual path guides
    for path in spec['learning_paths']:
        generate_path_guide(path, base_path, spec)
    
    # Generate overview file
    generate_paths_overview(spec, base_path / 'learning-paths.mdx')
    
    print(f"Generated learning paths in {output_dir}")

def generate_path_guide(path, base_path, spec):
    """Generate individual learning path guide"""
    content = f"""---
title: {path['title']}
description: {path['description']}
sidebar_position: 1
---

# {path['title']}

{path['description']}

## Path Overview

**Duration**: {path['duration']} hours  
**Difficulty**: {path['difficulty']}  
**Target Audience**: {path['target_audience']}

## Prerequisites

"""
    for prereq in path['prerequisites']:
        content += f"- {prereq}\n"
    
    content += "\n## Learning Outcomes\n\n"
    for outcome in path['target_outcomes']:
        content += f"- {outcome}\n"
    
    content += "\n## Curriculum\n\n"
    content += f"This path includes **{len(path['chapters'])}** chapters organized for optimal learning progression:\n\n"
    
    # Generate chapter list with progress tracking
    for i, chapter_id in enumerate(path['chapters'], 1):
        chapter_info = get_chapter_info(chapter_id)
        content += f"### {i}. {chapter_info['title']}\n"
        content += f"- **Duration**: {chapter_info['duration']} hours\n"
        content += f"- **Difficulty**: {chapter_info['difficulty']}\n"
        content += f"- **Description**: {chapter_info['description']}\n\n"
    
    content += "## Milestones\n\n"
    for milestone in path['milestones']:
        content += f"- {milestone}\n"
    
    content += "\n## Assessments\n\n"
    for assessment in path['assessments']:
        content += f"- {assessment}\n"
    
    content += "\n## Progress Tracking\n\n"
    content += "Use this checklist to track your progress:\n\n"
    for i, chapter_id in enumerate(path['chapters'], 1):
        content += f"- [ ] Chapter {i}: {get_chapter_info(chapter_id)['title']}\n"
    
    content += "\n## Certificate\n\n"
    certificate_info = spec.get('completion_certificates', {}).get(path['id'], {})
    if certificate_info:
        content += f"**Certificate**: {certificate_info.get('title', 'N/A')}\n\n"
        content += f"**Requirements**: {certificate_info.get('requirements', 'N/A')}\n\n"
        content += "**Skills Verified**:\n"
        for skill in certificate_info.get('skills_verified', []):
            content += f"- {skill}\n"
    
    # Write to file
    file_path = base_path / f"{path['id']}.mdx"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def generate_paths_overview(spec, output_path):
    """Generate overview of all learning paths"""
    content = """---
title: Learning Paths
description: Choose your learning journey through Physical AI & Humanoid Robotics
sidebar_position: 2
---

# Learning Paths

Choose the path that best fits your goals, timeline, and experience level. Each path is carefully designed to take you from your current skill level to your desired expertise.

## Path Comparison

| Path | Duration | Difficulty | Focus Area | Prerequisites |
|------|----------|------------|------------|---------------|
"""
    
    for path in spec['learning_paths']:
        content += f"| [{path['title']}]({path['id']}.mdx) | {path['duration']}h | {path['difficulty']} | {get_focus_area(path)} | {len(path['prerequisites'])} required |\n"
    
    content += "\n## Detailed Paths\n\n"
    
    for path in spec['learning_paths']:
        content += f"### {path['title']}\n\n"
        content += f"**{path['description']}**\n\n"
        content += f"- **Duration**: {path['duration']} hours\n"
        content += f"- **Difficulty**: {path['difficulty']}\n"
        content += f"- **Chapters**: {len(path['chapters'])}\n"
        content += f"- **Target**: {path['target_audience']}\n\n"
        
        # Progress indicator
        total_chapters = len(path['chapters'])
        content += "**Progress Tracker**:\n"
        content += f"[{'█' * min(5, total_chapters // 3)}{'░' * max(0, 5 - total_chapters // 3)}] {total_chapters} chapters\n\n"
        
        # Key outcomes
        content += "**Key Outcomes**:\n"
        for outcome in path['target_outcomes'][:3]:  # Show first 3 outcomes
            content += f"- {outcome}\n"
        content += "\n"
        
        content += f"[→ Start this path]({path['id']}.mdx)\n\n"
        content += "---\n\n"
    
    content += "## Recommendations\n\n"
    
    recommendations = spec.get('recommendations', {})
    for audience, paths in recommendations.items():
        content += f"### For {audience.title()}\n\n"
        for path_id in paths:
            path = next(p for p in spec['learning_paths'] if p['id'] == path_id)
            content += f"- [{path['title']}]({path['id']}.mdx)\n"
        content += "\n"
    
    content += "## Certificates\n\n"
    content += "Complete any learning path to earn a certificate verifying your skills:\n\n"
    
    certificates = spec.get('completion_certificates', {})
    for cert_id, cert_info in certificates.items():
        content += f"### {cert_info.get('title', 'N/A')}\n\n"
        content += f"**Requirements**: {cert_info.get('requirements', 'N/A')}\n\n"
        content += "**Skills Verified**:\n"
        for skill in cert_info.get('skills_verified', []):
            content += f"- {skill}\n"
        content += "\n"
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

def get_chapter_info(chapter_id):
    """Get chapter information from book-spec.yaml"""
    chapter_map = {
        'ch1-1': {'title': 'ROS 2 Architecture & Concepts', 'duration': 10, 'difficulty': 'beginner', 'description': 'Introduction to ROS 2 framework, its architecture, and core concepts for robotic development'},
        'ch1-2': {'title': 'Nodes, Topics, Services & Actions', 'duration': 12, 'difficulty': 'intermediate', 'description': 'Deep dive into ROS 2 communication patterns including publish/subscribe, request/response, and goal-based interactions'},
        'ch1-3': {'title': 'URDF & Robot Description Formats', 'duration': 10, 'difficulty': 'intermediate', 'description': 'Learn to create robot models using Unified Robot Description Format and visual representations'},
        'ch1-4': {'title': 'Python rclpy Programming', 'duration': 8, 'difficulty': 'intermediate', 'description': 'Develop complete ROS 2 applications using Python client library rclpy'},
        'ch2-1': {'title': 'Gazebo Simulation Environment', 'duration': 12, 'difficulty': 'intermediate', 'description': 'Set up and configure Gazebo for robotic simulation with ROS 2 integration'},
        'ch2-2': {'title': 'Physics Simulation & Collisions', 'duration': 10, 'difficulty': 'advanced', 'description': 'Master physics engines, collision detection, and realistic material properties'},
        'ch2-3': {'title': 'Sensor Simulation (LiDAR, Cameras, IMUs)', 'duration': 12, 'difficulty': 'intermediate', 'description': 'Simulate various robot sensors including LiDAR, cameras, and inertial measurement units'},
        'ch2-4': {'title': 'Environment and Scene Building', 'duration': 11, 'difficulty': 'intermediate', 'description': 'Create complex simulation environments and scenes for testing humanoid robots'},
        'ch3-1': {'title': 'NVIDIA Isaac Sim Platform', 'duration': 15, 'difficulty': 'advanced', 'description': 'Introduction to NVIDIA Isaac Sim for photorealistic robot simulation and AI development'},
        'ch3-2': {'title': 'Visual SLAM & Perception Pipelines', 'duration': 12, 'difficulty': 'advanced', 'description': 'Implement simultaneous localization and mapping using visual sensors and deep learning'},
        'ch3-3': {'title': 'Path Planning with Nav2', 'duration': 13, 'difficulty': 'advanced', 'description': 'Advanced navigation systems using ROS 2 Navigation 2 framework for autonomous movement'},
        'ch3-4': {'title': 'Sim-to-Real Transfer Techniques', 'duration': 10, 'difficulty': 'advanced', 'description': 'Bridge the gap between simulation and real-world robot deployment'},
        'ch4-1': {'title': 'Voice-to-Action Systems (Whisper)', 'duration': 14, 'difficulty': 'advanced', 'description': 'Implement speech recognition and command interpretation using OpenAI Whisper'},
        'ch4-2': {'title': 'LLM-based Cognitive Planning', 'duration': 13, 'difficulty': 'advanced', 'description': 'Utilize Large Language Models for robot task planning and decision making'},
        'ch4-3': {'title': 'Multi-modal Interaction Design', 'duration': 13, 'difficulty': 'advanced', 'description': 'Create systems that integrate vision, language, and action for natural robot interaction'},
        'ch4-4': {'title': 'Capstone Project - Autonomous Humanoid', 'duration': 15, 'difficulty': 'advanced', 'description': 'Complete integration project building an autonomous humanoid robot with full VLA capabilities'}
    }
    return chapter_map.get(chapter_id, {'title': 'Unknown Chapter', 'duration': 0, 'difficulty': 'unknown', 'description': 'Chapter description not available'})

def get_focus_area(path):
    """Get the main focus area for a learning path"""
    if 'complete' in path['id']:
        return 'Full Stack'
    elif 'simulation' in path['id']:
        return 'Simulation'
    elif 'software' in path['id']:
        return 'Development'
    elif 'quick' in path['id']:
        return 'Essentials'
    elif 'foundations' in path['id']:
        return 'Basics'
    else:
        return 'General'

if __name__ == "__main__":
    generate_learning_paths("learning-paths-spec.yaml", "docs/learning-paths")

#!/usr/bin/env python3
import yaml
import os
from pathlib import Path

def generate_mdx_files(spec_file, output_dir):
    """Generate MDX files from book specification YAML"""
    
    # Load the specification
    with open(spec_file, 'r') as f:
        spec = yaml.safe_load(f)
    
    # Create output directory structure
    base_path = Path(output_dir)
    base_path.mkdir(exist_ok=True)
    
    # Generate module directories and chapter files
    for module in spec['modules']:
        module_id = module['id']
        module_path = base_path / module_id
        module_path.mkdir(exist_ok=True)
        
        # Generate index.md for module
        generate_module_index(module, module_path)
        
        # Generate chapter files
        for i, chapter in enumerate(module['chapters']):
            chapter_file = f"chapter-{i+1}.mdx"
            chapter_path = module_path / chapter_file
            generate_chapter_mdx(chapter, chapter_path, i+1)
    
    # Generate main TOC
    generate_toc(spec, base_path)
    
    print(f"Generated MDX files in {output_dir}")

def generate_module_index(module, module_path):
    """Generate index.md for a module"""
    content = f"""---
title: {module['title']}
description: {module['description']}
sidebar_position: 1
---

# {module['title']}

{module['description']}

## Module Overview

**Duration**: {module['duration']} hours  
**Difficulty**: {module['difficulty']}

## Prerequisites
"""
    for prereq in module['prerequisites']:
        content += f"- {prereq}\n"
    
    content += "\n## Learning Outcomes\n"
    for outcome in module['learning_outcomes']:
        content += f"- {outcome}\n"
    
    content += "\n## Chapters\n"
    for i, chapter in enumerate(module['chapters']):
        content += f"{i+1}. [{chapter['title']}]({{'<ref \"{chapter['id']}\" />'}})\n"
    
    with open(module_path / 'index.md', 'w') as f:
        f.write(content)

def generate_chapter_mdx(chapter, chapter_path, chapter_num):
    """Generate MDX file for a chapter"""
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
    for outcome in chapter['learning_outcomes']:
        content += f"- {outcome}\n"
    
    content += "\n## Prerequisites\n\n"
    for prereq in chapter['prerequisites']:
        content += f"- {prereq}\n"
    
    content += """

## Introduction

<!-- Introduction content will be added here -->

## Key Concepts

<!-- Key concepts will be explained here -->

## Code Examples

<!-- Code examples and implementations will be added here -->

## Assessment

<!-- Assessment details will be added here -->

## What's Next

<!-- Next steps and related content -->

## Resources

"""
    for resource in chapter['resources']:
        content += f"- [{resource}]({resource})\n"
    
    with open(chapter_path, 'w') as f:
        f.write(content)

def generate_toc(spec, base_path):
    """Generate table of contents"""
    content = """# Physical AI & Humanoid Robotics - Table of Contents

## Overview

This comprehensive guide covers building intelligent humanoid robots using modern robotics and AI technologies.

## Modules

"""
    for module in spec['modules']:
        content += f"### {module['title']}\n\n"
        content += f"{module['description']}\n\n"
        content += f"**Duration**: {module['duration']} hours | **Difficulty**: {module['difficulty']}\n\n"
        
        for i, chapter in enumerate(module['chapters']):
            content += f"{i+1}. [{chapter['title']}]({module['id']}/chapter-{i+1}.mdx)\n"
            content += f"   - {chapter['description']}\n"
            content += f"   - Duration: {chapter['duration']}h | Difficulty: {chapter['difficulty']}\n\n"
        
        content += "\n"
    
    content += "## Assessments\n\n"
    for assessment in spec['assessments']:
        content += f"- **{assessment['name']}** ({assessment['type']}) - Module: {assessment['module']} - Weight: {assessment['weight']}%\n"
        content += f"  - {assessment['description']}\n\n"
    
    with open(base_path / 'toc.md', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    generate_mdx_files("book-spec.yaml", "docs")

#!/usr/bin/env python3
import yaml
import os
from pathlib import Path
import json

def generate_assessments(spec_file, output_dir):
    """Generate interactive assessment pages from specification"""
    
    # Load the specification
    with open(spec_file, 'r') as f:
        spec = yaml.safe_load(f)
    
    base_path = Path(output_dir)
    base_path.mkdir(exist_ok=True)
    
    # Generate individual assessment pages
    for assessment in spec['assessments']:
        generate_assessment_page(assessment, base_path)
    
    # Generate assessments overview
    generate_assessments_overview(spec, base_path / 'assessments.mdx')
    
    # Generate scoring system
    generate_scoring_system(spec, base_path / 'scoring-system.json')
    
    print(f"Generated assessments in {output_dir}")

def generate_assessment_page(assessment, base_path):
    """Generate individual assessment page"""
    
    content = f"""---
title: {assessment['title']}
description: {assessment['description']}
sidebar_position: 1
---

# {assessment['title']}

{assessment['description']}

## Assessment Details

- **Module**: {assessment['module']}
- **Type**: {assessment['type'].title()}
- **Duration**: {assessment['duration']} hours
- **Points**: {assessment['points']}
- **Week**: {assessment['week']}

## Learning Outcomes Assessed

"""
    for outcome in assessment['learning_outcomes']:
        content += f"- {outcome}\n"
    
    content += "\n## Prerequisites\n\n"
    for prereq in assessment['prerequisites']:
        content += f"- {prereq}\n"
    
    if assessment['type'] == 'quiz':
        content += generate_quiz_content(assessment)
    elif assessment['type'] in ['project', 'capstone']:
        content += generate_project_content(assessment)
    
    content += "\n## Resources\n\n"
    for resource in assessment['resources']:
        content += f"- [{resource['title']}]({resource['url']})\n"
    
    content += f"\n## Submission Format\n\n"
    if isinstance(assessment['submission_format'], list):
        for format_item in assessment['submission_format']:
            content += f"- {format_item}\n"
    else:
        content += assessment['submission_format']
    
    content += f"\n## Evaluation\n\n"
    content += f"- **Total Points**: {assessment['points']}\n"
    content += f"- **Passing Score**: {assessment.get('passing_score', 'N/A')}\n"
    content += f"- **Time Limit**: {assessment.get('time_limit', 'N/A')} minutes\n"
    
    # Write to file
    file_path = base_path / f"{assessment['id']}.mdx"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def generate_quiz_content(assessment):
    """Generate interactive quiz content"""
    content = "\n## Quiz Questions\n\n"
    content += f"*Total Questions: {len(assessment['questions'])}*\n\n"
    
    for i, question in enumerate(assessment['questions'], 1):
        content += f"### Question {i}\n\n"
        content += f"**Points**: {question['points']} | **Difficulty**: {question['difficulty']}\n\n"
        content += f"{question['question']}\n\n"
        
        if question['type'] == 'multiple_choice':
            content += "**Options**:\n"
            for j, option in enumerate(question['options'], 1):
                content += f"{j}. {option}\n"
            content += f"\n**Correct Answer**: {question['correct_answer']}\n\n"
            content += f"**Explanation**: {question['explanation']}\n\n"
        
        elif question['type'] == 'true_false':
            content += f"**Answer**: {question['correct_answer']}\n\n"
            content += f"**Explanation**: {question['explanation']}\n\n"
        
        elif question['type'] == 'short_answer':
            content += f"**Correct Answer**: {question['correct_answer']}\n\n"
            content += f"**Explanation**: {question['explanation']}\n\n"
        
        elif question['type'] == 'code_completion':
            content += "**Code Template**:\n```python\n"
            content += question['code_template']
            content += "\n```\n\n"
            content += f"**Answer**: `{question['correct_answer']}`\n\n"
            content += f"**Explanation**: {question['explanation']}\n\n"
        
        elif question['type'] == 'matching':
            content += "**Match the following**:\n"
            for pair in question['pairs']:
                content += f"- {pair['concept']}: {pair['description']}\n"
            content += "\n"
    
    content += "\n## Quiz Instructions\n\n"
    content += "1. Read each question carefully before answering\n"
    content += "2. Select the best answer for multiple choice questions\n"
    content += "3. Provide concise answers for short answer questions\n"
    content += "4. Complete the code as specified for programming questions\n"
    content += "5. Review your answers before submitting\n"
    
    return content

def generate_project_content(assessment):
    """Generate project-based assessment content"""
    content = "\n## Project Tasks\n\n"
    content += f"*Total Tasks: {len(assessment['tasks'])}*\n\n"
    
    for i, task in enumerate(assessment['tasks'], 1):
        content += f"### Task {i}: {task['title']}\n\n"
        content += f"**Points**: {task['points']} | **Difficulty**: {task['difficulty']}\n\n"
        content += f"{task['description']}\n\n"
        
        content += "**Deliverables**:\n"
        for deliverable in task['deliverables']:
            content += f"- {deliverable}\n"
        
        content += "\n**Evaluation Rubric**:\n"
        rubric = task['rubric']
        for level, criteria in rubric.items():
            content += f"- **{level.title()}**: {criteria}\n"
        content += "\n"
    
    content += "## Project Requirements\n\n"
    content += "1. Complete all tasks according to specifications\n"
    content += "2. Follow best practices for code quality and documentation\n"
    content += "3. Test your implementation thoroughly\n"
    content += "4. Document your design decisions and challenges\n"
    content += "5. Prepare a comprehensive demonstration\n\n"
    
    content += "## Evaluation Criteria\n\n"
    if 'evaluation_criteria' in assessment:
        criteria = assessment['evaluation_criteria']
        for criterion, weight in criteria.items():
            content += f"- **{criterion.title()}**: {weight}%\n"
    
    return content

def generate_assessments_overview(spec, output_path):
    """Generate overview of all assessments"""
    content = """---
title: Assessments
description: Complete assessment system for Physical AI & Humanoid Robotics course
sidebar_position: 3
---

# Course Assessments

This course uses a comprehensive assessment system to evaluate your progress and skills across all learning modules.

## Assessment Schedule

| Week | Assessment | Type | Points | Weight |
|------|------------|------|--------|--------|
"""
    
    schedule = spec.get('assessment_schedule', [])
    for item in schedule:
        assessment = next(a for a in spec['assessments'] if a['id'] == item['assessment'])
        content += f"| {item['week']} | [{assessment['title']}]({assessment['id']}.mdx) | {assessment['type'].title()} | {assessment['points']} | {item['weight']}% |\n"
    
    content += "\n## Assessment Types\n\n"
    
    # Group assessments by type
    quiz_assessments = [a for a in spec['assessments'] if a['type'] == 'quiz']
    project_assessments = [a for a in spec['assessments'] if a['type'] == 'project']
    capstone_assessments = [a for a in spec['assessments'] if a['type'] == 'capstone']
    
    content += "### Quizzes\n\n"
    content += "Quizzes evaluate your understanding of theoretical concepts and fundamental knowledge:\n\n"
    for quiz in quiz_assessments:
        content += f"- **[{quiz['title']}]({quiz['id']}.mdx)** (Week {quiz['week']}, {quiz['points']} points)\n"
    
    content += "\n### Projects\n\n"
    content += "Projects assess your practical skills and ability to apply concepts to real-world problems:\n\n"
    for project in project_assessments:
        content += f"- **[{project['title']}]({project['id']}.mdx)** (Week {project['week']}, {project['points']} points)\n"
    
    content += "\n### Capstone\n\n"
    content += "The capstone project evaluates your ability to integrate all course technologies:\n\n"
    for capstone in capstone_assessments:
        content += f"- **[{capstone['title']}]({capstone['id']}.mdx)** (Week {capstone['week']}, {capstone['points']} points)\n"
    
    content += "\n## Grading Scale\n\n"
    grading_scale = spec.get('grading_scale', {})
    for grade, range in grading_scale.items():
        content += f"- **{grade}**: {range}%\n"
    
    content += "\n## Submission Guidelines\n\n"
    guidelines = spec.get('submission_guidelines', {})
    content += f"**Late Policy**: {guidelines.get('late_policy', 'N/A')}\n\n"
    content += f"**Academic Integrity**: {guidelines.get('academic_integrity', 'N/A')}\n\n"
    content += f"**Collaboration**: {guidelines.get('collaboration', 'N/A')}\n\n"
    content += f"**Format Requirements**: {guidelines.get('format_requirements', 'N/A')}\n\n"
    
    content += "## Certificate Requirements\n\n"
    cert_reqs = spec.get('certificate_requirements', {})
    content += f"**Minimum Grade**: {cert_reqs.get('minimum_grade', 'N/A')}%\n\n"
    content += "**Required Assessments**:\n"
    for req in cert_reqs.get('required_assessments', []):
        assessment = next(a for a in spec['assessments'] if a['id'] == req)
        content += f"- {assessment['title']}\n"
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

def generate_scoring_system(spec, output_path):
    """Generate scoring system configuration"""
    scoring_config = {
        "assessments": {},
        "grading_scale": spec.get('grading_scale', {}),
        "certificate_requirements": spec.get('certificate_requirements', {}),
        "evaluation_weights": {}
    }
    
    # Add assessment scoring details
    for assessment in spec['assessments']:
        scoring_config["assessments"][assessment['id']] = {
            "title": assessment['title'],
            "type": assessment['type'],
            "points": assessment['points'],
            "week": assessment['week'],
            "passing_score": assessment.get('passing_score', 70),
            "time_limit": assessment.get('time_limit', None),
            "questions": len(assessment.get('questions', [])),
            "tasks": len(assessment.get('tasks', []))
        }
    
    # Add evaluation weights from schedule
    schedule = spec.get('assessment_schedule', [])
    for item in schedule:
        scoring_config["evaluation_weights"][item['assessment']] = item['weight']
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scoring_config, f, indent=2)

if __name__ == "__main__":
    generate_assessments("assessments-spec.yaml", "docs/assessments")

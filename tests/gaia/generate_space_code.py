#!/usr/bin/env python3
"""
Generate space_code.py for GAIA benchmark submission
"""

import json
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("space_code_generator")

def generate_space_code(submission_file: str = "tests/gaia/submision/gaia_submission_answers.json",
                       output_file: str = "space_code.py") -> bool:
    """Generate space_code.py from GAIA submission answers"""
    
    try:
        # Load submission answers
        with open(submission_file, 'r', encoding='utf-8') as f:
            answers = json.load(f)
        
        logger.info(f"Loaded {len(answers)} submission answers from {submission_file}")
        
        # Generate space_code.py content
        space_code_content = '''#!/usr/bin/env python3
"""
GAIA Benchmark Submission - Generated Space Code
This file contains the model answers for GAIA benchmark evaluation.
"""

# GAIA submission answers in the required format
ANSWERS = [
'''
        
        # Add each answer to the space code
        for answer in answers:
            task_id = answer.get("task_id", "")
            model_answer = answer.get("model_answer", "")
            
            # Escape quotes and newlines for Python string literals
            submitted_answer_escaped = repr(model_answer)
            
            space_code_content += f'''    {{
        "task_id": "{task_id}",
        "submitted_answer": {submitted_answer_escaped}
    }},
'''
        
        # Close the list
        space_code_content += ''']

def get_answer(task_id: str) -> dict:
    """Get answer for a specific task ID"""
    for answer in ANSWERS:
        if answer["task_id"] == task_id:
            return answer
    return {
        "task_id": task_id,
        "submitted_answer": "Task ID not found"
    }

def get_all_answers() -> list:
    """Get all answers"""
    return ANSWERS

def get_task_ids() -> list:
    """Get all task IDs"""
    return [answer["task_id"] for answer in ANSWERS]

def get_statistics() -> dict:
    """Get submission statistics"""
    total_tasks = len(ANSWERS)
    answered_tasks = sum(1 for answer in ANSWERS
                        if answer.get("submitted_answer", "").strip())
    
    return {
        "total_tasks": total_tasks,
        "answered_tasks": answered_tasks,
        "completion_rate": answered_tasks / total_tasks if total_tasks > 0 else 0.0
    }

if __name__ == "__main__":
    # Print statistics when run directly
    stats = get_statistics()
    print(f"GAIA Submission Statistics:")
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"Answered tasks: {stats['answered_tasks']}")
    print(f"Completion rate: {stats['completion_rate']:.2%}")
'''
        
        # Write space_code.py
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(space_code_content)
        
        logger.info(f"✅ Generated {output_file} with {len(answers)} answers")
        return True
        
    except FileNotFoundError:
        logger.error(f"❌ Submission file not found: {submission_file}")
        return False
    except Exception as e:
        logger.error(f"❌ Error generating space code: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate space_code.py for GAIA submission")
    parser.add_argument("--input", "-i", default="tests/gaia/submision/gaia_submission_answers.json",
                       help="Input submission answers JSON file")
    parser.add_argument("--output", "-o", default="space_code.py",
                       help="Output space_code.py file")
    
    args = parser.parse_args()
    
    success = generate_space_code(args.input, args.output)
    
    if success:
        logger.info("🎉 Space code generation completed successfully!")
        return 0
    else:
        logger.error("💥 Space code generation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
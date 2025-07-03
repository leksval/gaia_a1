#!/usr/bin/env python3
"""
Update GAIA submission answers format to match the new requirements
"""

import json
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("format_updater")

def main():
    # Input and output files
    input_file = "gaia_submission_answers.json"
    output_file = "gaia_submission_answers.json"
    
    logger.info(f"Loading answers from {input_file}")
    
    try:
        # Read current answers
        with open(input_file, "r", encoding="utf-8") as f:
            answers = json.load(f)
        
        logger.info(f"Loaded {len(answers)} answers")
        
        # Update format: change submitted_answer to model_answer and add reasoning_trace
        new_answers = []
        for item in answers:
            # Extract reasoning portion from the answer if possible
            answer = item.get("submitted_answer", "")
            reasoning = ""  # We'll use an empty string as default reasoning
            
            # Create new item with updated format
            new_item = {
                "task_id": item.get("task_id", ""),
                "model_answer": answer,
                "reasoning_trace": reasoning
            }
            
            new_answers.append(new_item)
        
        # Write updated answers
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_answers, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Updated {len(new_answers)} answers and saved to {output_file}")
        
        # Regenerate space_code.py with updated format
        logger.info("Regenerating space_code.py with updated format")
        import subprocess
        subprocess.run(["python", "tests/gaia/generate_space_code.py"])
        
        logger.info("Done! The submission format has been updated to match the new requirements.")
        return 0
    
    except Exception as e:
        logger.error(f"Error updating submission format: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Submit GAIA answers to the leaderboard using gradio_client
"""

import argparse
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gaia_submission_client")

def main():
    parser = argparse.ArgumentParser(description="Submit GAIA answers to the leaderboard using gradio_client")
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--space-name", required=True, help="Name of your duplicated space (e.g., 'my-gaia-submission')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    args = parser.parse_args()
    
    # Check if gradio_client is installed
    try:
        from gradio_client import Client
    except ImportError:
        logger.error("gradio_client package is not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio_client"])
        from gradio_client import Client
    
    # Your Hugging Face Space name
    space_name = f"{args.username}/{args.space_name}"
    
    logger.info(f"Connecting to Space: {space_name}")
    
    try:
        # Connect to your Space
        client = Client(space_name)
        
        # Check login status
        logger.info("Checking login status...")
        login_status = client.predict(
            api_name="/_check_login_status"
        )
        logger.info(f"Login status: {login_status}")
        
        # If not logged in, show instructions
        if "Not logged in" in login_status:
            logger.error("You are not logged in to HuggingFace. Please log in using one of these methods:")
            logger.error("1. Log in to HuggingFace in your browser")
            logger.error("2. Create a token at https://huggingface.co/settings/tokens")
            logger.error("3. Set the token as an environment variable: export HUGGINGFACE_TOKEN=your_token_here")
            return 1
            
        # Run the submission
        logger.info("Submitting answers to leaderboard...")
        result = client.predict(
            api_name="/run_and_submit_all"
        )
        
        # Print results
        logger.info("\nSubmission results:")
        logger.info(f"Status: {result[0]}")
        
        if args.verbose and result[1] and 'data' in result[1] and result[1]['data']:
            logger.info("\nDetailed results:")
            for row in result[1]['data']:
                logger.info(f"Question: {row[1]}")
                logger.info(f"Answer: {row[2]}")
                logger.info(f"Correct: {row[3]}")
                logger.info("---")
        
        return 0
    except Exception as e:
        logger.error(f"Error during submission: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
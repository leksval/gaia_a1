#!/usr/bin/env python3
"""
Official GAIA benchmark testing with proper submission format
"""

import json
import requests
import logging
import sys
import os
from typing import List, Dict, Any
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gaia_official_test")

class GaiaOfficialTester:
    """Official GAIA benchmark tester with proper submission format"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    def fetch_gaia_questions_from_api(self) -> List[Dict[str, Any]]:
        """Fetch GAIA questions from the official API"""
        try:
            logger.info("Fetching GAIA questions from official API...")
            gaia_api_url = "https://agents-course-unit4-scoring.hf.space"
            response = requests.get(f"{gaia_api_url}/questions", timeout=30)
            
            if response.status_code == 200:
                questions = response.json()
                logger.info(f"Fetched {len(questions)} GAIA questions from official API")
                return questions
            else:
                logger.error(f"Failed to fetch questions: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching GAIA questions: {e}")
            return []

    def load_gaia_questions(self, questions_file: str) -> List[Dict[str, Any]]:
        """Load GAIA questions from JSON file (fallback method)"""
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            logger.info(f"Loaded {len(questions)} GAIA questions from {questions_file}")
            return questions
        except FileNotFoundError:
            logger.error(f"Questions file not found: {questions_file}")
            return []
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            return []
    
    def test_single_question(self, question_data: Dict[str, Any], question_num: int = 1, total_questions: int = 1) -> Dict[str, Any]:
        """Test a single GAIA question and return result in submission format"""
        task_id = question_data.get("task_id", "unknown")
        question = question_data.get("question", "")
        files = question_data.get("files", [])
        
        logger.info(f"📝 Question {question_num}/{total_questions} - Task ID: {task_id}")
        logger.info(f"📋 Question: {question[:200]}{'...' if len(question) > 200 else ''}")
        if files:
            logger.info(f"📎 Files: {len(files)} attached")
        
        try:
            # Prepare request payload
            payload = {
                "question": question,
                "files": files
            }
            
            # Call GAIA answer endpoint
            response = requests.post(
                f"{self.base_url}/gaia-answer",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract answer and reasoning
                model_answer = result.get("answer", "")
                reasoning_trace = result.get("reasoning", "")
                sources = result.get("sources", [])
                
                # Log the answer
                logger.info(f"💡 Answer: {model_answer}")
                logger.info(f"📝 Reasoning length: {len(reasoning_trace)} characters")
                logger.info(f"📚 Sources: {len(sources)} found")
                
                # Create submission format
                submission_entry = {
                    "task_id": task_id,
                    "model_answer": model_answer,
                    "reasoning_trace": reasoning_trace
                }
                
                logger.info(f"✅ Task {task_id} completed successfully")
                return submission_entry
                
            else:
                logger.error(f"❌ Task {task_id} failed with status {response.status_code}: {response.text}")
                return {
                    "task_id": task_id,
                    "model_answer": "",
                    "reasoning_trace": f"Error: HTTP {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            logger.error(f"❌ Task {task_id} failed with exception: {e}")
            return {
                "task_id": task_id,
                "model_answer": "",
                "reasoning_trace": f"Error: {str(e)}"
            }
    
    def run_single_test(self, output_file: str = "gaia_single_submission.json") -> bool:
        """Run test with one random GAIA question"""
        logger.info("Starting single GAIA question test")
        
        # Fetch questions from API
        questions = self.fetch_gaia_questions_from_api()
        if not questions:
            logger.error("No questions fetched, aborting test")
            return False
        
        # Select one random question
        import random
        random_question = random.choice(questions)
        logger.info(f"Selected random question: {random_question.get('task_id', 'unknown')}")
        
        # Test the question
        result = self.test_single_question(random_question, 1, 1)
        
        if result["model_answer"]:
            # Save submission file
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump([result], f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Single test completed and saved to {output_file}")
                
                # Print summary
                print("\n" + "="*60)
                print("GAIA SINGLE QUESTION TEST RESULT")
                print("="*60)
                print(f"Task ID: {random_question.get('task_id', 'unknown')}")
                print(f"Question: {random_question.get('question', '')[:200]}...")
                print(f"Model Answer: {result['model_answer']}")
                print(f"Reasoning Length: {len(result['reasoning_trace'])} chars")
                print("="*60)
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to save single test result: {e}")
                return False
        else:
            logger.error("❌ Single question test failed")
            return False

    def run_official_tests(self, questions_file: str = None, output_file: str = "gaia_submission_answers.json") -> bool:
        """Run official GAIA tests and generate submission file"""
        logger.info("Starting official GAIA benchmark testing")
        
        # Load questions - try API first, then file
        if questions_file:
            questions = self.load_gaia_questions(questions_file)
        else:
            questions = self.fetch_gaia_questions_from_api()
            
        if not questions:
            logger.error("No questions loaded, aborting test")
            return False
        
        # Test each question
        submission_results = []
        success_count = 0
        
        for i, question_data in enumerate(questions, 1):
            result = self.test_single_question(question_data, i, len(questions))
            submission_results.append(result)
            
            if result["model_answer"]:  # Non-empty answer indicates success
                success_count += 1
        
        # Save submission file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(submission_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Submission file saved: {output_file}")
            logger.info(f"📊 Results: {success_count}/{len(questions)} questions answered successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save submission file: {e}")
            return False
    
    def validate_api_health(self) -> bool:
        """Validate that the API is healthy before running tests"""
        try:
            health_response = requests.get(f"{self.base_url}/health", timeout=10)
            if health_response.status_code == 200:
                logger.info("✅ API health check passed")
                return True
            else:
                logger.error(f"❌ API health check failed: {health_response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ API health check failed: {e}")
            return False

def main():
    """Main function for official GAIA testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run official GAIA benchmark tests")
    parser.add_argument("--questions", "-q",
                       help="Path to GAIA questions JSON file (optional, will use API if not provided)")
    parser.add_argument("--output", "-o", default="gaia_submission_answers.json",
                       help="Output submission file path")
    parser.add_argument("--url", "-u", default="http://localhost:8000",
                       help="Base URL for GAIA API")
    parser.add_argument("--all", action="store_true",
                       help="Test with all available questions (default: test with one random question)")
    
    args = parser.parse_args()
    
    # Create tester
    tester = GaiaOfficialTester(base_url=args.url)
    
    # Validate API health
    if not tester.validate_api_health():
        logger.error("API health check failed, aborting tests")
        return 1
    
    # Run tests based on arguments
    if args.all:
        # Run all tests
        success = tester.run_official_tests(args.questions, args.output)
    else:
        # Run single test (default)
        success = tester.run_single_test(args.output.replace('.json', '_single.json'))
    
    if success:
        logger.info("🎉 Official GAIA testing completed successfully!")
        return 0
    else:
        logger.error("💥 Official GAIA testing failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
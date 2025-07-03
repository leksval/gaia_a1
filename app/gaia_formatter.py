#!/usr/bin/env python3
"""
Enhanced GAIA Answer Formatter with Zero-Space Programming

This module handles proper formatting of GAIA benchmark answers to ensure:
1. Only required fields are included (task_id and submitted_answer)
2. The output is properly formatted JSON with no unnecessary comments
3. The answer format follows GAIA benchmark expectations for optimal scoring

Enhanced with assertion-based validation and contract programming.
"""

import json
import logging
import re
import copy
from typing import List, Dict, Any, Optional

# Set up logging and assertions
from tools.logging import get_logger
from tools.assertions import (
    require, ensure, invariant, assert_not_none, assert_type,
    contract, GaiaAssertionError
)

logger = get_logger(__name__)

# Constants
UNNECESSARY_PHRASES = [
    r"^I think ",
    r"^I believe ",
    r"^Based on my knowledge,",
    r"^In my opinion,",
    r"^As far as I know,",
    r"^According to my understanding,",
    r"^From my understanding,",
    r"^Let me provide",
    r"^Let me answer",
    r"^To answer your question,",
]

VERBOSE_ENDINGS = [
    r"\. I hope this helps\.$",
    r"\. Hope this helps\.$",
    r"\. Let me know if you need anything else\.$",
    r"\. Let me know if you have any other questions\.$",
]

CLARIFICATIONS = [
    r"\(Note:.*?\)",
    r"\(This.*?\)",
    r"\[Note:.*?\]",
]


class GaiaFormatter:
    """
    Enhanced class for formatting GAIA benchmark answers to optimize scoring.
    
    Implements precise answer formatting as recommended in GAIA_PPX.md:
    - Question type detection (number, word, list, multi-line)
    - Format-specific extraction rules
    - Confidence scoring for answer extraction
    - Validation loops for low-confidence answers
    """
    
    @staticmethod
    @contract(
        preconditions=[lambda question: isinstance(question, str) and len(question.strip()) > 0],
        postconditions=[lambda result, question: isinstance(result, str) and result in ['number', 'word', 'list', 'multi_line', 'general']]
    )
    def detect_question_type(question: str) -> str:
        """
        Detect the type of GAIA question to apply appropriate formatting with assertion validation.
        
        Args:
            question: The original question text
            
        Returns:
            Question type: 'number', 'word', 'list', 'multi_line', or 'general'
        """
        assert_type(question, str, "question", {"operation": "question_type_detection"})
        require(
            len(question.strip()) > 0,
            "Question must not be empty",
            context={"question_length": len(question)}
        )
        
        question_lower = question.lower()
        
        # Number questions
        if any(phrase in question_lower for phrase in [
            'how many', 'what is the number', 'count', 'calculate',
            'what percentage', 'how much', 'what year', 'what age'
        ]):
            return 'number'
        
        # Single word questions
        if any(phrase in question_lower for phrase in [
            'what is the name', 'who is', 'what country', 'what city',
            'what color', 'what type', 'which', 'name the'
        ]) and 'list' not in question_lower:
            return 'word'
        
        # List questions
        if any(phrase in question_lower for phrase in [
            'list', 'name all', 'what are the', 'which are',
            'enumerate', 'identify all'
        ]):
            return 'list'
        
        # Multi-line questions (explanations, descriptions)
        if any(phrase in question_lower for phrase in [
            'explain', 'describe', 'why', 'how does', 'what happens',
            'analyze', 'compare', 'discuss'
        ]):
            return 'multi_line'
        
        return 'general'
    
    @staticmethod
    @contract(
        preconditions=[
            lambda text, question_type: isinstance(text, str) and len(text.strip()) > 0,
            lambda text, question_type: isinstance(question_type, str) and question_type in ['number', 'word', 'list', 'multi_line', 'general']
        ],
        postconditions=[lambda result, text, question_type: isinstance(result, str)]
    )
    def extract_answer_by_type(text: str, question_type: str) -> str:
        """
        Extract answer based on detected question type with assertion validation.
        
        Args:
            text: Raw answer text
            question_type: Detected question type
            
        Returns:
            Formatted answer optimized for GAIA scoring
        """
        assert_type(text, str, "text", {"operation": "answer_extraction"})
        require(
            len(text.strip()) > 0,
            "Text must not be empty",
            context={"text_length": len(text)}
        )
        
        assert_type(question_type, str, "question_type", {"operation": "answer_extraction"})
        require(
            question_type in ['number', 'word', 'list', 'multi_line', 'general'],
            "Question type must be valid",
            context={"question_type": question_type, "valid_types": ['number', 'word', 'list', 'multi_line', 'general']}
        )
        
        # First, try to extract from "FINAL ANSWER:" format
        final_answer = GaiaFormatter._extract_final_answer_format(text)
        if final_answer:
            text = final_answer
        
        if question_type == 'number':
            result = GaiaFormatter._extract_number_answer(text)
        elif question_type == 'word':
            result = GaiaFormatter._extract_word_answer(text)
        elif question_type == 'list':
            result = GaiaFormatter._extract_list_answer(text)
        elif question_type == 'multi_line':
            result = GaiaFormatter._extract_multiline_answer(text)
        else:
            result = GaiaFormatter._extract_general_answer(text)
        
        ensure(
            isinstance(result, str),
            "Extracted answer must be string",
            context={"question_type": question_type, "original_length": len(text)}
        )
        
        return result
    
    @staticmethod
    def _extract_final_answer_format(text: str) -> str:
        """Extract answer from 'FINAL ANSWER:' format."""
        import re
        
        # Look for "FINAL ANSWER:" pattern (case insensitive)
        pattern = r'FINAL\s+ANSWER\s*:\s*(.+?)(?:\n|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            answer = match.group(1).strip()
            # Remove any trailing punctuation or formatting
            answer = re.sub(r'[.!?]*$', '', answer).strip()
            return answer
        
        return ""
    
    @staticmethod
    def _extract_number_answer(text: str) -> str:
        """Extract numeric answer with proper formatting."""
        import re
        
        # Look for numbers in the text
        number_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*%',  # Percentages
            r'\b(\d{4})\b',            # Years
            r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b',  # Numbers with commas
            r'\b(\d+(?:\.\d+)?)\b'     # Simple numbers
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Return the first number found, cleaned up
                number = matches[0].replace(',', '')
                # Remove trailing zeros for decimals
                if '.' in number:
                    number = number.rstrip('0').rstrip('.')
                return number
        
        # If no number found, return the original text cleaned
        return text.strip()
    
    @staticmethod
    def _extract_word_answer(text: str) -> str:
        """Extract single word/phrase answer."""
        # Remove common prefixes
        prefixes_to_remove = [
            'the answer is', 'it is', 'this is', 'that is',
            'the name is', 'the country is', 'the city is'
        ]
        
        cleaned = text.lower()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # Take the first sentence or phrase
        sentences = text.split('.')
        first_sentence = sentences[0].strip()
        
        # Remove quotes if they surround the entire answer
        if first_sentence.startswith('"') and first_sentence.endswith('"'):
            first_sentence = first_sentence[1:-1]
        
        return first_sentence
    
    @staticmethod
    def _extract_list_answer(text: str) -> str:
        """Extract list answer with proper formatting."""
        # Check if already in comma-separated format
        if ',' in text and not any(marker in text for marker in ['•', '-', '*', '\n']):
            # Clean up existing comma-separated list
            items = [item.strip() for item in text.split(',')]
            return ', '.join(items)
        
        # Extract from bullet points or numbered lists
        lines = text.split('\n')
        list_items = []
        
        for line in lines:
            line = line.strip()
            # Match various list formats
            if re.match(r'^[\d\w]*[\.\)]\s*(.+)', line):
                # Numbered list: "1. item" or "a) item"
                match = re.match(r'^[\d\w]*[\.\)]\s*(.+)', line)
                list_items.append(match.group(1).strip())
            elif re.match(r'^[-•*]\s*(.+)', line):
                # Bullet list: "- item" or "• item"
                match = re.match(r'^[-•*]\s*(.+)', line)
                list_items.append(match.group(1).strip())
        
        if list_items:
            return ', '.join(list_items)
        
        # If no clear list structure, return original
        return text.strip()
    
    @staticmethod
    def _extract_multiline_answer(text: str) -> str:
        """Extract multi-line answer with proper structure."""
        # For explanations, keep the structure but clean up
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def _extract_general_answer(text: str) -> str:
        """Extract general answer with basic cleanup."""
        # Remove common AI response patterns
        patterns_to_remove = [
            r'^(Based on|According to|From|In|The answer is:?)\s*',
            r'\s*(I hope this helps|Let me know if you need more information)\.?$'
        ]
        
        cleaned = text
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    @staticmethod
    def calculate_answer_confidence(original_text: str, extracted_answer: str, question_type: str) -> float:
        """
        Calculate confidence score for extracted answer.
        
        Args:
            original_text: Original response text
            extracted_answer: Extracted answer
            question_type: Type of question
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Higher confidence for shorter, more direct answers
        if len(extracted_answer) < 50:
            confidence += 0.2
        
        # Higher confidence if answer matches expected type
        if question_type == 'number' and re.match(r'^\d+(?:\.\d+)?$', extracted_answer.strip()):
            confidence += 0.3
        elif question_type == 'word' and len(extracted_answer.split()) <= 3:
            confidence += 0.3
        elif question_type == 'list' and ',' in extracted_answer:
            confidence += 0.2
        
        # Lower confidence for very long answers to simple questions
        if question_type in ['number', 'word'] and len(extracted_answer) > 100:
            confidence -= 0.3
        
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, confidence))
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text by handling newlines and basic cleanup.
        
        Args:
            text: The text to normalize
            
        Returns:
            Normalized text
        """
        # Replace string-encoded newlines with actual newlines
        if '\\n' in text:
            logger.debug("Replacing string-encoded newlines")
            text = text.replace('\\n', '\n')
            
        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def _process_bullet_lists(text: str) -> str:
        """
        Process bullet lists consistently by converting to comma-separated format or preserving structure.
        
        Args:
            text: The text containing potential bullet lists
            
        Returns:
            Processed text
        """
        bullet_pattern = r'^\s*[-•*]\s*(.*?)$'
        
        # First check if there are bullet points
        if not re.search(bullet_pattern, text, re.MULTILINE):
            return text
            
        # Extract bullet items
        items = re.findall(bullet_pattern, text, re.MULTILINE)
        if not items:
            return text
            
        # Clean up items
        items = [item.strip() for item in items if item.strip()]
        
        # Determine if we should preserve bullet format or convert to comma-separated
        if GaiaFormatter.should_preserve_bullet_format(items):
            # Keep structured format with standardized bullets
            lines = text.split('\n')
            formatted_lines = []
            
            for line in lines:
                # Normalize bullet points but keep the content structure
                if re.match(bullet_pattern, line):
                    # Extract the content after the bullet
                    content = re.sub(r'^\s*[-•*]\s*', '• ', line)
                    formatted_lines.append(content)
                else:
                    # Keep non-bullet lines as they are
                    formatted_lines.append(line)
            
            return '\n'.join(formatted_lines)
        else:
            # Convert to comma-separated format for consistency
            return ', '.join(items)
    
    @staticmethod
    def should_preserve_bullet_format(bullet_items: List[str]) -> bool:
        """
        Determine if bullet list structure should be preserved based on content.
        
        Args:
            bullet_items: List of strings extracted from bullet points
            
        Returns:
            Boolean indicating if bullet format should be preserved
        """
        try:
            # Validate input
            if not bullet_items or not isinstance(bullet_items, list):
                return False
                
            # Preserve bullet format if:
            # 1. Any bullet item contains multiple sentences or paragraphs
            # 2. Average length of bullet items is more than 80 characters
            
            # Check for multiple sentences in any item
            for item in bullet_items:
                if not isinstance(item, str):
                    continue  # Skip non-string items
                    
                if '. ' in item.strip() or '\n' in item:
                    return True
            
            # Check average length
            string_items = [item for item in bullet_items if isinstance(item, str)]
            if not string_items:
                return False
                
            avg_length = sum(len(item) for item in string_items) / len(string_items)
            if avg_length > 80:
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error in should_preserve_bullet_format: {str(e)}")
            return False  # Default to not preserving on error
    
    @staticmethod
    def strip_unnecessary_text(answer: str) -> str:
        """
        Strip unnecessary text from the answer.
        
        Args:
            answer: The answer text
            
        Returns:
            Cleaned answer text
        """
        # Strip unnecessary phrases at the beginning
        for phrase in UNNECESSARY_PHRASES:
            if re.match(phrase, answer, re.IGNORECASE):
                answer = re.sub(phrase, "", answer, flags=re.IGNORECASE)
                # Capitalize the first letter if needed
                if answer and answer[0].islower():
                    answer = answer[0].upper() + answer[1:]
        
        # Strip verbose endings
        for ending in VERBOSE_ENDINGS:
            answer = re.sub(ending, ".", answer, flags=re.IGNORECASE)
        
        # Remove clarifications in parentheses or brackets
        for clarification in CLARIFICATIONS:
            answer = re.sub(clarification, "", answer, flags=re.IGNORECASE)
        
        # Fix multiple spaces
        answer = re.sub(r' +', ' ', answer)
        
        return answer.strip()

    @classmethod
    def format_gaia_answer(cls, answer: str) -> str:
        """
        Format a single GAIA answer string.
        
        Args:
            answer: The answer text
            
        Returns:
            Formatted answer text
        """
        # Create a dummy answer object
        dummy_answer = {"task_id": "dummy", "submitted_answer": answer}
        
        # Strip unnecessary text
        dummy_answer["submitted_answer"] = cls.strip_unnecessary_text(dummy_answer["submitted_answer"])
        
        # Format by type
        formatted = cls.format_answer_by_type(dummy_answer)
        
        return formatted["submitted_answer"]

    @staticmethod
    def format_answer_by_type(answer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format an answer based on specific question types.
        
        Args:
            answer: The answer object
            
        Returns:
            Formatted answer object
        """
        try:
            task_id = answer.get("task_id", "")
            submitted_answer = answer.get("submitted_answer", "")
            
            # Skip empty answers
            if not submitted_answer:
                return answer
            
            # Handle string-encoded newlines CONSISTENTLY and FIRST
            if isinstance(submitted_answer, str) and '\\n' in submitted_answer:
                submitted_answer = GaiaFormatter._normalize_text(submitted_answer)
            
            # Format short answers (likely single word/phrase answers)
            if len(submitted_answer) < 30:
                # Remove periods from short answers
                submitted_answer = submitted_answer.rstrip(".")
                # Remove quotes if they surround the entire answer
                if submitted_answer.startswith('"') and submitted_answer.endswith('"'):
                    submitted_answer = submitted_answer[1:-1]
                # Remove phrases like "the answer is"
                submitted_answer = re.sub(r"^(the answer is|answer:)\s+", "", submitted_answer, flags=re.IGNORECASE)
            
            # Clean up list-type answers
            if "," in submitted_answer and len(submitted_answer) < 50:
                # Remove whitespace around commas
                submitted_answer = re.sub(r'\s*,\s*', ', ', submitted_answer)
                # Ensure no period at the end for comma-separated lists
                submitted_answer = submitted_answer.rstrip(".")
                
            # Enhanced bullet point/dash lists handling
            bullet_pattern = r'^\s*[-•*]\s*(.*?)$'
            if re.search(bullet_pattern, submitted_answer, re.MULTILINE):
                # Process bullet lists using our enhanced method
                submitted_answer = GaiaFormatter._process_bullet_lists(submitted_answer)
            
            # Format "not enough information" answers to be clearer and more concise
            if re.search(r"(not enough|insufficient|don'?t have) (information|data|context|evidence|details)", submitted_answer, re.IGNORECASE):
                # But only if the answer is apologetic and verbose
                if submitted_answer.lower().startswith("i apologize") and len(submitted_answer) > 200:
                    # Extract the core message
                    not_enough_info_match = re.search(r"(not enough|insufficient|don'?t have) (information|data|context|evidence|details).*?(?=\.|$)", submitted_answer, re.IGNORECASE | re.DOTALL)
                    if not_enough_info_match:
                        core_message = not_enough_info_match.group(0)
                        # Find a concise sentence with the core message
                        sentences = re.split(r'(?<=[.!?])\s+', submitted_answer)
                        for sentence in sentences:
                            if re.search(r"(not enough|insufficient|don'?t have) (information|data|context|evidence|details)", sentence, re.IGNORECASE):
                                submitted_answer = sentence
                                break
            
            # Update the answer
            result = copy.deepcopy(answer)
            result["submitted_answer"] = submitted_answer
            return result
            
        except Exception as e:
            logger.error(f"Error formatting answer by type: {str(e)}")
            # If any error occurs, return the original answer unmodified
            return answer

    @classmethod
    def format_gaia_answer_dict(cls, answer_dict: Dict[str, Any], question: str = None) -> Dict[str, Any]:
        """
        Format a complete GAIA answer dictionary (with answer, reasoning, sources).
        
        Args:
            answer_dict: The complete answer dictionary
            question: Optional question text for type detection
            
        Returns:
            Formatted answer dictionary
        """
        try:
            # Validate input
            if not isinstance(answer_dict, dict):
                logger.warning(f"answer_dict is not a dictionary but {type(answer_dict)}")
                try:
                    # Try to convert to dictionary if possible
                    if hasattr(answer_dict, "__dict__"):
                        answer_dict = answer_dict.__dict__
                    else:
                        # Create a simple dict with string representation
                        answer_dict = {"answer": str(answer_dict)}
                except Exception as e:
                    logger.error(f"Failed to convert answer_dict to dictionary: {str(e)}")
                    # Fall back to empty dictionary with error message
                    return {
                        "answer": "Error: Input is not a valid dictionary",
                        "reasoning": "",
                        "sources": []
                    }
            
            # Start with an empty result
            result = {}
            
            # Format the answer field with enhanced error handling
            if "answer" in answer_dict:
                if isinstance(answer_dict["answer"], str):
                    # Apply question type detection if question is provided
                    if question:
                        question_type = cls.detect_question_type(question)
                        result["answer"] = cls.extract_answer_by_type(answer_dict["answer"], question_type)
                    else:
                        result["answer"] = cls.format_gaia_answer(answer_dict["answer"])
                else:
                    # Convert non-string to string
                    try:
                        if question:
                            question_type = cls.detect_question_type(question)
                            result["answer"] = cls.extract_answer_by_type(str(answer_dict["answer"]), question_type)
                        else:
                            result["answer"] = cls.format_gaia_answer(str(answer_dict["answer"]))
                    except Exception as e:
                        logger.error(f"Error formatting answer: {str(e)}")
                        result["answer"] = "Error formatting answer"
            
            # Include reasoning if present (optional) with type validation
            if "reasoning" in answer_dict:
                reasoning = answer_dict["reasoning"]
                if reasoning is None:
                    result["reasoning"] = ""
                elif isinstance(reasoning, str):
                    result["reasoning"] = reasoning
                else:
                    # Convert non-string to string
                    try:
                        result["reasoning"] = str(reasoning)
                    except Exception:
                        result["reasoning"] = ""
            
            # Include sources if present (optional) with type validation
            if "sources" in answer_dict:
                sources = answer_dict["sources"]
                if isinstance(sources, list):
                    # Sanitize sources to ensure they're serializable
                    safe_sources = []
                    for source in sources:
                        if isinstance(source, dict):
                            # Keep only string values with limited length
                            safe_source = {str(k): str(v)[:100] for k, v in source.items()}
                            safe_sources.append(safe_source)
                        else:
                            # Add the source as a simple string
                            try:
                                safe_sources.append(str(source))
                            except:
                                pass  # Skip non-serializable sources
                    result["sources"] = safe_sources
                else:
                    result["sources"] = []
            
            # Ensure all required fields exist
            if "answer" not in result or not result["answer"]:
                result["answer"] = "No answer available"
                
            if "reasoning" not in result:
                result["reasoning"] = ""
                
            if "sources" not in result:
                result["sources"] = []
            
            # Test JSON serialization of the result before returning
            try:
                json.dumps(result)
            except Exception as e:
                logger.error(f"Result is not JSON serializable: {str(e)}")
                # Return a simplified result that is guaranteed to be serializable
                return {
                    "answer": "Error: Could not serialize response. Please check formatter logs.",
                    "reasoning": "",
                    "sources": []
                }
            
            return result
        except Exception as e:
            logger.error(f"Error formatting GAIA answer dictionary: {str(e)}")
            # Ultimate fallback
            return {
                "answer": "Error formatting answer",
                "reasoning": "",
                "sources": []
            }

    @classmethod
    def extract_gaia_answer(cls, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract GaiaAnswer data from the agent's state data.
        
        Args:
            state_data: Serialized state data from the agent
            
        Returns:
            Dictionary containing answer, reasoning, and sources
        """
        # Helper function to extract content from various object types
        def extract_content(obj):
            # Handle None
            if obj is None:
                return ""
            
            # Special pattern matching for strings that look like AIMessage representations
            # This handles the case we're seeing in logs: content='- broccoli\n- celery...' additional_kwargs={...}
            if isinstance(obj, str):
                # Check for the pattern content='...'
                if obj.startswith("content='") and "'" in obj[9:]:
                    try:
                        end_index = obj[9:].find("'")
                        if end_index > 0:
                            extracted = obj[9:9+end_index]
                            # Replace encoded newlines
                            extracted = extracted.replace('\\n', '\n')
                            return extracted
                    except Exception as e:
                        logger.error(f"Error extracting content from string pattern: {str(e)}")
                
                # Try regex to extract content between quotes after 'content='
                try:
                    import re
                    content_match = re.search(r"content=['\"]([^'\"]+)['\"]", obj)
                    if content_match:
                        extracted = content_match.group(1)
                        # Replace encoded newlines
                        extracted = extracted.replace('\\n', '\n')
                        return extracted
                except Exception as e:
                    logger.error(f"Error with regex content extraction: {str(e)}")
                
                # If no special pattern, return the string as-is
                return obj
            
            # Check specifically for LangChain AIMessage class by name
            obj_type = type(obj).__name__
            if obj_type == "AIMessage":
                logger.debug("Detected LangChain AIMessage object")
                
                # Extract only the content field from AIMessage
                if hasattr(obj, 'content'):
                    content = obj.content
                    
                    # If content is a string, apply our string extraction logic recursively
                    if isinstance(content, str):
                        return extract_content(content)
                    
                    # Handle object with nested content
                    if hasattr(content, 'content'):
                        if isinstance(content.content, str):
                            return content.content
                        else:
                            # Apply string extraction to the stringified nested content
                            return extract_content(str(content.content))
                    
                    # Return stringified content as last resort for AIMessage.content
                    return extract_content(str(content))
                
                # If we can't access content attribute, stringify the object and extract
                return extract_content(str(obj))
                
            # Handle dict-like objects with 'content' attribute
            if hasattr(obj, 'content'):
                content = obj.content
                # Process content recursively using our extraction logic
                return extract_content(content)
                    
            # Handle dict with 'content' key
            if isinstance(obj, dict) and 'content' in obj:
                content = obj['content']
                # Process content recursively using our extraction logic
                return extract_content(content)
            
            # Last resort, convert to string and attempt extraction
            obj_str = str(obj)
            return extract_content(obj_str)
        
        # Default response in case of failure
        default_response = {
            "answer": "No answer could be extracted from the state data",
            "reasoning": "",
            "sources": []
        }
        
        try:
            # Validate state_data
            if not isinstance(state_data, dict):
                logger.error(f"state_data is not a dictionary but {type(state_data)}")
                return default_response
            
            # Multiple extraction strategies
            answer = ""
            reasoning = ""
            sources = []
            
            # EXTRACTION STRATEGY 1: From gaia_answer_object
            if "gaia_answer_object" in state_data:
                logger.debug("Found gaia_answer_object in state data")
                answer_obj = state_data["gaia_answer_object"]
                
                if isinstance(answer_obj, dict):
                    answer = extract_content(answer_obj.get("answer", ""))
                    reasoning = extract_content(answer_obj.get("reasoning", ""))
                    sources = answer_obj.get("sources", [])
                    
                    if answer:
                        logger.debug(f"Extracted from gaia_answer_object: {answer[:50]}...")
                        return {
                            "answer": answer,
                            "reasoning": reasoning,
                            "sources": sources
                        }
                else:
                    # Try to convert non-dict to string
                    answer = extract_content(answer_obj)
                        
            # EXTRACTION STRATEGY 2: From intermediate_steps_log
            if not answer and "intermediate_steps_log" in state_data:
                logger.debug("Checking intermediate_steps_log for final_answer")
                steps = state_data.get("intermediate_steps_log", [])
                
                if isinstance(steps, list):
                    for step in reversed(steps):  # Search from the latest step
                        if not isinstance(step, dict):
                            continue
                        
                        if step.get("type") == "final_answer":
                            content = step.get("content", {})
                            if isinstance(content, dict):
                                answer = content.get("answer", "")
                                reasoning = content.get("reasoning", "")
                                sources = content.get("sources", [])
                                
                                if answer:
                                    logger.debug(f"Extracted from intermediate_steps_log: {answer[:50]}...")
                                    return {
                                        "answer": answer,
                                        "reasoning": reasoning,
                                        "sources": sources
                                    }
            
            # EXTRACTION STRATEGY 3: From messages
            if not answer and "messages" in state_data:
                logger.debug("Checking messages for answer")
                messages = state_data.get("messages", [])
                
                if isinstance(messages, list):
                    for msg in reversed(messages):  # Search from the latest message
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if isinstance(content, str) and content.strip() and not content.startswith("Self-assessment:"):
                                answer = content
                                logger.debug(f"Extracted from messages: {answer[:50]}...")
                                break
            
            # If we found an answer, format it
            if answer:
                # Process the answer
                formatted_answer = cls.format_gaia_answer(answer)
                
                return {
                    "answer": formatted_answer,
                    "reasoning": reasoning if reasoning else "",
                    "sources": sources if sources else []
                }
            
            return default_response
            
        except Exception as e:
            logger.error(f"Error extracting GAIA answer: {str(e)}")
            return default_response
"""
Text Processor MCP Server

This MCP server provides tools for advanced text processing operations, including:
- Pattern extraction from text
- Text parsing and transformation
- Content validation
"""

import json
import os
import re
import asyncio
import sys
from typing import Dict, Any, List, Optional, Union, Pattern, Callable

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import assertions for error handling
from tools.assertions import (
    require, ensure, assert_not_none, assert_type, assert_non_empty,
    ConfigurationError, NetworkError, ProcessingError
)

# Import the base MCP server class
from mcp_servers.mcp_base import MCPServer, MCPToolDefinition
from tools.logging import get_logger

# Configure logging
logger = get_logger("text_processor_server")

class TextProcessorServer(MCPServer):
    """MCP server that provides tools for advanced text processing operations."""
    
    def __init__(self):
        super().__init__(
            name="text_processor_server",
            description="Provides tools for advanced text processing and extraction operations"
        )
    
    def initialize(self):
        """Initialize the server with tools and resources."""
        logger.info("Initializing TextProcessorServer with tools")
        
        # Register the tools this server provides
        extract_patterns_tool = MCPToolDefinition(
            name="extract_patterns",
            description="Extract content matching specific patterns from text",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to process"
                    },
                    "pattern_type": {
                        "type": "string",
                        "description": "Type of pattern to extract (regex, list_items, code_blocks, etc.)"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "The pattern to match (regex pattern or named pattern type)"
                    },
                    "flags": {
                        "type": "string",
                        "description": "Optional regex flags (e.g., 'i' for case-insensitive)"
                    }
                },
                "required": ["text", "pattern_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "matches": {"type": "array", "items": {"type": "string"}, "description": "Array of pattern matches"},
                    "match_count": {"type": "integer", "description": "Number of matches found"}
                }
            },
            function=self.extract_patterns
        )
        self.register_tool(extract_patterns_tool)
        
        parse_structured_text_tool = MCPToolDefinition(
            name="parse_structured_text",
            description="Parse structured text into a structured format",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to parse"
                    },
                    "format_type": {
                        "type": "string",
                        "description": "The format to parse into (json, list, dict, table, etc.)"
                    },
                    "structure_hints": {
                        "type": "object",
                        "description": "Optional hints about the structure (column names, expected fields, etc.)"
                    }
                },
                "required": ["text", "format_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "parsed_data": {"type": ["object", "array"], "description": "Parsed structured data"},
                    "format": {"type": "string", "description": "Format used for parsing"},
                    "success": {"type": "boolean", "description": "Whether parsing was successful"}
                }
            },
            function=self.parse_structured_text
        )
        self.register_tool(parse_structured_text_tool)
        
        validate_extraction_tool = MCPToolDefinition(
            name="validate_extraction",
            description="Validate if extracted content is complete and accurate",
            input_schema={
                "type": "object",
                "properties": {
                    "original_text": {
                        "type": "string",
                        "description": "The original text that items were extracted from"
                    },
                    "extracted_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The list of extracted items to validate"
                    },
                    "extraction_type": {
                        "type": "string",
                        "description": "The type of extraction performed (list_items, code_blocks, etc.)"
                    }
                },
                "required": ["original_text", "extracted_items", "extraction_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "is_valid": {"type": "boolean", "description": "Whether extraction is valid and complete"},
                    "missing_items": {"type": "array", "items": {"type": "string"}, "description": "Items that may be missing from extraction"},
                    "validation_details": {"type": "string", "description": "Details about the validation process"}
                }
            },
            function=self.validate_extraction
        )
        self.register_tool(validate_extraction_tool)
        
        direct_extraction_tool = MCPToolDefinition(
            name="direct_extraction",
            description="Extract specific information directly from text based on instructions",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to extract information from"
                    },
                    "extraction_goal": {
                        "type": "string",
                        "description": "Description of what to extract (e.g., 'Extract all list items')"
                    },
                    "format": {
                        "type": "string",
                        "description": "Desired output format (list, json, etc.)"
                    }
                },
                "required": ["text", "extraction_goal"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "extracted_data": {"type": ["array", "object"], "description": "Extracted data in requested format"},
                    "format": {"type": "string", "description": "Format used for extraction"},
                    "success": {"type": "boolean", "description": "Whether extraction was successful"}
                }
            },
            function=self.direct_extraction
        )
        self.register_tool(direct_extraction_tool)
    
    async def extract_patterns(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content matching specific patterns from text."""
        assert_not_none(params, "pattern extraction parameters")
        
        text = params.get("text", "")
        pattern_type = params.get("pattern_type", "")
        pattern = params.get("pattern", "")
        flags = params.get("flags", "")
        
        assert_non_empty(text, "text to process")
        assert_non_empty(pattern_type, "pattern_type")
        
        matches = []
        
        # Handle different pattern types
        if pattern_type == "regex":
            assert_non_empty(pattern, "regex pattern")
            
            # Process regex flags
            regex_flags = 0
            if 'i' in flags:
                regex_flags |= re.IGNORECASE
            if 'm' in flags:
                regex_flags |= re.MULTILINE
            if 's' in flags:
                regex_flags |= re.DOTALL
            
            # Compile and apply regex
            regex = re.compile(pattern, regex_flags)
            raw_matches = regex.finditer(text)
            
            for match in raw_matches:
                match_text = match.group(0)
                # Get some context around the match
                start_pos = max(0, match.start() - 20)
                end_pos = min(len(text), match.end() + 20)
                context = text[start_pos:end_pos]
                
                matches.append({
                    "text": match_text,
                    "start": match.start(),
                    "end": match.end(),
                    "groups": match.groups(),
                    "context": context
                })
        
        elif pattern_type == "list_items":
            # Extract items that appear to be in a list format
            bullet_pattern = r'[-•*]\s*([^-•*\n]+)'
            numbered_pattern = r'\d+\.\s*([^\n]+)'
            
            bullet_matches = re.finditer(bullet_pattern, text)
            for match in bullet_matches:
                matches.append({
                    "text": match.group(1).strip(),
                    "type": "bullet",
                    "start": match.start(),
                    "end": match.end()
                })
            
            numbered_matches = re.finditer(numbered_pattern, text)
            for match in numbered_matches:
                matches.append({
                    "text": match.group(1).strip(),
                    "type": "numbered",
                    "start": match.start(),
                    "end": match.end()
                })
        
        elif pattern_type == "code_blocks":
            # Extract code blocks (markdown style or indented)
            markdown_code_pattern = r'```(?:\w+)?\n(.*?)\n```'
            
            # Find markdown-style code blocks
            markdown_matches = re.finditer(markdown_code_pattern, text, re.DOTALL)
            for match in markdown_matches:
                matches.append({
                    "text": match.group(1),
                    "type": "markdown_code_block",
                    "start": match.start(),
                    "end": match.end()
                })
            
            # Find indented code blocks
            indented_lines = []
            current_block = []
            in_block = False
            
            for line in text.split('\n'):
                if re.match(r'^ {4,}|\t+', line):
                    if not in_block:
                        in_block = True
                    current_block.append(line)
                else:
                    if in_block:
                        indented_lines.append('\n'.join(current_block))
                        current_block = []
                        in_block = False
            
            if current_block:  # Don't forget the last block
                indented_lines.append('\n'.join(current_block))
            
            for block in indented_lines:
                matches.append({
                    "text": block,
                    "type": "indented_code_block"
                })
        
        else:
            require(
                False,
                f"Unsupported pattern type: {pattern_type}",
                context={"supported_types": ["regex", "list_items", "code_blocks"]}
            )
        
        return {
            "matches": matches,
            "match_count": len(matches),
            "pattern_type": pattern_type
        }
    
    async def parse_structured_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse structured text into a structured format."""
        assert_not_none(params, "parsing parameters")
        
        text = params.get("text", "")
        format_type = params.get("format_type", "")
        structure_hints = params.get("structure_hints", {})
        
        assert_non_empty(text, "text to parse")
        assert_non_empty(format_type, "format_type")
        assert_type(structure_hints, dict, "structure_hints")
        
        parsed_data = None
        
        if format_type == "json":
            # Try to parse as JSON
            parsed_data = json.loads(text)
        
        elif format_type == "list":
            # Parse as a list of items
            list_items = []
            
            # Try different list patterns
            bullet_matches = re.findall(r'[-•*]\s*([^-•*\n]+)', text)
            numbered_matches = re.findall(r'\d+\.\s*([^\n]+)', text)
            
            list_items.extend([item.strip() for item in bullet_matches])
            list_items.extend([item.strip() for item in numbered_matches])
            
            # If no list items found, try splitting by newlines or commas
            if not list_items:
                if ',' in text:
                    list_items = [item.strip() for item in text.split(',')]
                else:
                    list_items = [item.strip() for item in text.split('\n') if item.strip()]
            
            parsed_data = list_items
        
        else:
            require(
                False,
                f"Unsupported format type: {format_type}",
                context={"supported_formats": ["json", "list"]}
            )
        
        return {
            "parsed_data": parsed_data,
            "format": format_type,
            "success": True
        }
    
    async def validate_extraction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if extracted content is complete and accurate."""
        assert_not_none(params, "validation parameters")
        
        original_text = params.get("original_text", "")
        extracted_items = params.get("extracted_items", [])
        extraction_type = params.get("extraction_type", "")
        
        assert_non_empty(original_text, "original_text")
        assert_type(extracted_items, list, "extracted_items")
        assert_non_empty(extraction_type, "extraction_type")
        
        # First, perform our own extraction to compare with the provided extraction
        extraction_result = await self.extract_patterns({
            "text": original_text,
            "pattern_type": extraction_type
        })
        
        our_matches = extraction_result.get("matches", [])
        our_items = [match.get("text", "") for match in our_matches]
        
        # Compare the two lists
        missing_items = []
        for item in our_items:
            if item not in extracted_items:
                missing_items.append(item)
        
        is_valid = len(missing_items) == 0
        validation_details = f"Found {len(our_items)} items in original text, {len(extracted_items)} provided. Missing: {len(missing_items)}"
        
        return {
            "is_valid": is_valid,
            "missing_items": missing_items,
            "validation_details": validation_details
        }
    
    async def direct_extraction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specific information directly from text based on instructions."""
        assert_not_none(params, "extraction parameters")
        
        text = params.get("text", "")
        extraction_goal = params.get("extraction_goal", "")
        format_type = params.get("format", "list")
        
        assert_non_empty(text, "text to extract from")
        assert_non_empty(extraction_goal, "extraction_goal")
        
        # Simple rule-based extraction based on common goals
        extracted_data = []
        
        if "list" in extraction_goal.lower():
            # Extract list items
            result = await self.extract_patterns({
                "text": text,
                "pattern_type": "list_items"
            })
            extracted_data = [match.get("text", "") for match in result.get("matches", [])]
        
        elif "code" in extraction_goal.lower():
            # Extract code blocks
            result = await self.extract_patterns({
                "text": text,
                "pattern_type": "code_blocks"
            })
            extracted_data = [match.get("text", "") for match in result.get("matches", [])]
        
        else:
            # Default to extracting lines that aren't empty
            extracted_data = [line.strip() for line in text.split('\n') if line.strip()]
        
        return {
            "extracted_data": extracted_data,
            "format": format_type,
            "success": True
        }


def main():
    """Main entry point for the text processor server."""
    server = TextProcessorServer()
    server.start()

if __name__ == "__main__":
    main()
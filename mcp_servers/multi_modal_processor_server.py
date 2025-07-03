#!/usr/bin/env python3
"""
Multi-Modal Processor MCP Server

This server provides multi-modal processing capabilities for the GAIA system,
specifically handling image analysis using LLaVA model via Ollama.

Key features:
- Image analysis and description generation
- Chart/graph data extraction
- Visual question answering
- Image content summarization
"""

import asyncio
import json
import logging
import os
import sys
import base64
import io
from typing import Dict, List, Any, Optional
import requests
from PIL import Image

from mcp_servers.mcp_base import MCPServer, MCPToolDefinition
from tools.assertions import (
    require, ensure, assert_not_none, assert_non_empty,
    assert_api_response, NetworkError, ProcessingError
)
from tools.langfuse_monitor import trace_span, create_tracer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModalProcessorServer(MCPServer):
    """MCP Server for multi-modal processing operations."""
    
    def __init__(self):
        super().__init__("multimodal_processor", "Multimodal processing for images, audio, and documents")
        self.vision_model_name = None
        self.ollama_base_url = None
        self.default_confidence = None
        self.request_timeout = None
        self.mock_response_enabled = None
        
    def initialize(self):
        """Initialize the multi-modal processor server."""
        self.vision_model_name = os.getenv("VISION_MODEL_NAME", "llava:13b")
        self.ollama_base_url = os.getenv("VISION_OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.default_confidence = float(os.getenv("VISION_DEFAULT_CONFIDENCE", "0.8"))
        self.request_timeout = int(os.getenv("VISION_REQUEST_TIMEOUT", "60"))
        self.mock_response_enabled = os.getenv("VISION_MOCK_RESPONSE_ENABLED", "true").lower() == "true"
        
        # Validate configuration with assertions
        assert_not_none(self.vision_model_name, "vision_model_name", {"operation": "initialize"})
        assert_not_none(self.ollama_base_url, "ollama_base_url", {"operation": "initialize"})
        require(
            0.0 <= self.default_confidence <= 1.0,
            "Default confidence must be between 0.0 and 1.0",
            context={"confidence": self.default_confidence}
        )
        require(
            self.request_timeout > 0,
            "Request timeout must be positive",
            context={"timeout": self.request_timeout}
        )
        
        logger.info(f"Multi-modal processor initialized with model: {self.vision_model_name}")
        logger.info(f"Ollama base URL: {self.ollama_base_url}")
        logger.info(f"Default confidence: {self.default_confidence}")
        logger.info(f"Request timeout: {self.request_timeout}s")
        logger.info(f"Mock response enabled: {self.mock_response_enabled}")
        
        # Register tools using the base class method with Langfuse tracing
        from mcp_servers.mcp_base import MCPToolDefinition
        import asyncio
        
        # Register analyze_image tool
        analyze_image_tool = MCPToolDefinition(
            name="analyze_image",
            description="Analyze an image and provide detailed description, extract text, or answer questions about the image content.",
            input_schema={
                "type": "object",
                "properties": {
                    "image_data": {
                        "type": "string",
                        "description": "Base64 encoded image data or image URL"
                    },
                    "question": {
                        "type": "string",
                        "description": "Specific question about the image (optional)",
                        "default": "Describe this image in detail."
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis to perform",
                        "enum": ["description", "text_extraction", "chart_analysis", "question_answering"],
                        "default": "description"
                    }
                },
                "required": ["image_data"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "extracted_text": {"type": "string"},
                    "confidence": {"type": "number"},
                    "analysis_type": {"type": "string"}
                }
            },
            function=lambda args: asyncio.run(self._analyze_image_with_tracing(args))
        )
        self.register_tool(analyze_image_tool)
        
        # Register extract_chart_data tool
        extract_chart_tool = MCPToolDefinition(
            name="extract_chart_data",
            description="Extract data from charts, graphs, and tables in images.",
            input_schema={
                "type": "object",
                "properties": {
                    "image_data": {
                        "type": "string",
                        "description": "Base64 encoded image data or image URL"
                    },
                    "chart_type": {
                        "type": "string",
                        "description": "Type of chart to extract data from",
                        "enum": ["bar", "line", "pie", "table", "scatter", "auto"],
                        "default": "auto"
                    }
                },
                "required": ["image_data"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "extracted_data": {"type": "array"},
                    "chart_type": {"type": "string"},
                    "confidence": {"type": "number"}
                }
            },
            function=lambda args: asyncio.run(self._extract_chart_data_with_tracing(args))
        )
        self.register_tool(extract_chart_tool)
        
        # Register visual question answering tool
        visual_qa_tool = MCPToolDefinition(
            name="visual_question_answering",
            description="Answer specific questions about image content with high accuracy.",
            input_schema={
                "type": "object",
                "properties": {
                    "image_data": {
                        "type": "string",
                        "description": "Base64 encoded image data"
                    },
                    "question": {
                        "type": "string",
                        "description": "Specific question about the image"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context for the question (optional)"
                    }
                },
                "required": ["image_data", "question"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"}
                }
            },
            function=lambda args: asyncio.run(self._visual_question_answering(args))
        )
        self.register_tool(visual_qa_tool)
    
    async def _analyze_image_with_tracing(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image with Langfuse tracing."""
        tracer = create_tracer("image_analysis", "analyze_image")
        if tracer:
            with trace_span(tracer, "multimodal_analyze_image", arguments) as span:
                try:
                    result = await self._analyze_image(arguments)
                    span["set_output"](result)
                    return result
                except Exception as e:
                    span["set_output"]({"error": str(e)})
                    raise
        else:
            return await self._analyze_image(arguments)
    
    async def _extract_chart_data_with_tracing(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Extract chart data with Langfuse tracing."""
        tracer = create_tracer("chart_extraction", "extract_chart_data")
        if tracer:
            with trace_span(tracer, "multimodal_extract_chart", arguments) as span:
                try:
                    result = await self._extract_chart_data(arguments)
                    span["set_output"](result)
                    return result
                except Exception as e:
                    span["set_output"]({"error": str(e)})
                    raise
        else:
            return await self._extract_chart_data(arguments)
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a multi-modal processing tool."""
        assert_not_none(tool_name, "tool_name", {"operation": "execute_tool"})
        assert_not_none(arguments, "arguments", {"operation": "execute_tool"})
        
        available_tools = ["analyze_image", "extract_chart_data", "visual_question_answering"]
        require(
            tool_name in available_tools,
            f"Tool '{tool_name}' must be one of: {available_tools}",
            context={"tool_name": tool_name, "available_tools": available_tools}
        )
        
        result = None
        if tool_name == "analyze_image":
            result = await self._analyze_image_with_tracing(arguments)
        elif tool_name == "extract_chart_data":
            result = await self._extract_chart_data_with_tracing(arguments)
        elif tool_name == "visual_question_answering":
            result = await self._visual_question_answering(arguments)
        
        ensure(
            isinstance(result, dict),
            "Tool execution result must be a dictionary",
            context={"tool_name": tool_name, "result_type": type(result).__name__}
        )
        
        return result
    
    async def _analyze_image(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an image using the vision model."""
        image_data = arguments.get("image_data", "")
        question = arguments.get("question", "Describe this image in detail.")
        analysis_type = arguments.get("analysis_type", "description")
        
        assert_non_empty(image_data, "image_data", {"operation": "analyze_image"})
        
        valid_analysis_types = ["description", "text_extraction", "chart_analysis", "question_answering"]
        require(
            analysis_type in valid_analysis_types,
            f"Analysis type must be one of: {valid_analysis_types}",
            context={"analysis_type": analysis_type, "valid_types": valid_analysis_types}
        )
        
        # Prepare the prompt based on analysis type
        prompt_map = {
            "description": "Provide a detailed description of this image, including all visible objects, people, text, and context.",
            "text_extraction": "Extract all text visible in this image. Provide the text exactly as it appears.",
            "chart_analysis": "Analyze this chart or graph. Describe the type of chart, the data it shows, trends, and key insights.",
            "question_answering": question
        }
        prompt = prompt_map.get(analysis_type, question)
        
        # Call the vision model
        result = await self._call_vision_model(image_data, prompt)
        
        ensure(
            isinstance(result, dict),
            "Vision model result must be a dictionary",
            context={"result_type": type(result).__name__}
        )
        
        require(
            "error" not in result,
            "Vision model returned an error",
            context={"error": result.get("error", "Unknown error")}
        )
        
        response_data = {
            "analysis": result.get("response", ""),
            "extracted_text": result.get("response", "") if analysis_type == "text_extraction" else "",
            "confidence": self.default_confidence,
            "analysis_type": analysis_type
        }
        
        ensure(
            isinstance(response_data["analysis"], str),
            "Analysis result must be a string",
            context={"analysis_type": type(response_data["analysis"]).__name__}
        )
        
        return response_data
    
    async def _extract_chart_data(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from charts and graphs."""
        image_data = arguments.get("image_data", "")
        chart_type = arguments.get("chart_type", "auto")
        
        assert_non_empty(image_data, "image_data", {"operation": "extract_chart_data"})
        
        valid_chart_types = ["bar", "line", "pie", "scatter", "table", "auto"]
        require(
            chart_type in valid_chart_types,
            f"Chart type must be one of: {valid_chart_types}",
            context={"chart_type": chart_type, "valid_types": valid_chart_types}
        )
        
        # Specialized prompt for chart data extraction
        prompt = """Analyze this chart or graph and extract the following information:
1. Chart type (bar, line, pie, scatter, table, etc.)
2. Title of the chart
3. Data points and their values
4. Labels for axes or categories
5. Any trends or patterns visible

Format your response as structured data that can be easily parsed."""
        
        result = await self._call_vision_model(image_data, prompt)
        
        ensure(
            isinstance(result, dict),
            "Vision model result must be a dictionary",
            context={"result_type": type(result).__name__}
        )
        
        require(
            "error" not in result,
            "Vision model returned an error",
            context={"error": result.get("error", "Unknown error")}
        )
        
        # Parse the response to extract structured data
        response_text = result.get("response", "")
        assert_not_none(response_text, "response_text", {"operation": "extract_chart_data"})
        
        chart_data = {
            "chart_type": chart_type if chart_type != "auto" else "unknown",
            "data_points": [],  # Would be extracted from response
            "labels": [],       # Would be extracted from response
            "title": "",        # Would be extracted from response
            "description": response_text
        }
        
        ensure(
            isinstance(chart_data["description"], str),
            "Chart description must be a string",
            context={"description_type": type(chart_data["description"]).__name__}
        )
        
        return chart_data
    
    async def _visual_question_answering(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Answer specific questions about image content."""
        image_data = arguments.get("image_data", "")
        question = arguments.get("question", "")
        context = arguments.get("context", "")
        
        assert_non_empty(image_data, "image_data", {"operation": "visual_question_answering"})
        assert_non_empty(question, "question", {"operation": "visual_question_answering"})
        
        # Prepare the prompt with context if provided
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease answer the question based on what you see in the image." if context else f"Question: {question}\n\nPlease answer the question based on what you see in the image."
        
        result = await self._call_vision_model(image_data, prompt)
        
        ensure(
            isinstance(result, dict),
            "Vision model result must be a dictionary",
            context={"result_type": type(result).__name__}
        )
        
        require(
            "error" not in result,
            "Vision model returned an error",
            context={"error": result.get("error", "Unknown error")}
        )
        
        answer_data = {
            "answer": result.get("response", ""),
            "confidence": self.default_confidence,
            "reasoning": "Based on visual analysis of the provided image."
        }
        
        ensure(
            isinstance(answer_data["answer"], str),
            "Answer must be a string",
            context={"answer_type": type(answer_data["answer"]).__name__}
        )
        
        return answer_data
    
    async def _call_vision_model(self, image_data: str, prompt: str) -> Dict[str, Any]:
        """Call the LLaVA vision model via Ollama."""
        assert_non_empty(image_data, "image_data", {"operation": "call_vision_model"})
        assert_non_empty(prompt, "prompt", {"operation": "call_vision_model"})
        
        # Prepare the request for Ollama
        payload = {
            "model": self.vision_model_name,
            "prompt": prompt,
            "images": [image_data],  # LLaVA expects base64 encoded images
            "stream": False
        }
        
        # Make the request to Ollama
        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json=payload,
            timeout=self.request_timeout
        )
        
        require(
            response.status_code == 200,
            f"Ollama API request failed with status {response.status_code}",
            context={"status_code": response.status_code, "response_text": response.text}
        )
        
        data = response.json()
        return {
            "response": data.get("response", ""),
            "model": self.vision_model_name
        }
    
    def _get_mock_vision_response(self, prompt: str) -> Dict[str, Any]:
        """Generate mock vision response for testing."""
        mock_message = os.getenv("VISION_MOCK_MESSAGE", 
            "Mock vision analysis for prompt: '{prompt}'. This would normally contain detailed image analysis from the LLaVA model.")
        
        return {
            "response": mock_message.format(prompt=prompt),
            "model": "mock_vision_model"
        }
    
    def _validate_image_data(self, image_data: str) -> bool:
        """Validate that the image data is properly formatted."""
        assert_non_empty(image_data, "image_data", {"operation": "validate_image_data"})
        
        # Check if it's base64 encoded
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
        
        # Try to decode base64
        decoded = base64.b64decode(image_data)
        
        # Try to open as image
        image = Image.open(io.BytesIO(decoded))
        return True

def main():
    """Main entry point for the multi-modal processor server."""
    server = MultiModalProcessorServer()
    server.start()

if __name__ == "__main__":
    main()
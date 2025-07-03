# mcp_servers/basic_tools_server.py
"""
Basic Tools MCP Server

This server provides fundamental tools for web search and code execution.
"""

import json
import os
import sys
import ast
from typing import Dict, Any
from simpleeval import simple_eval

# Update imports to work with new directory structure
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import assertions for error handling
from tools.assertions import (
    require, ensure, assert_not_none, assert_type, assert_non_empty,
    ConfigurationError, NetworkError, ProcessingError
)

# Set up logging to stderr before any potential logging
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("basic_tools_server")

# Import config with assertion-based validation
from config.config import settings
assert_not_none(settings, "settings configuration")
if not hasattr(settings, 'api_max_results'):
    settings.api_max_results = 100
    logger.warning("api_max_results not configured, using default value 100")
if not hasattr(settings, 'tavily_api_key'):
    settings.tavily_api_key = None
    logger.warning("tavily_api_key not configured")

# Import MCP server components
from mcp_servers.mcp_base import MCPServer, MCPToolDefinition, MCPResourceDefinition

# Import logging utilities
from tools.logging import get_logger

# Configure logging using the shared logging configuration
logger = get_logger("basic_tools_server")


class BasicToolsServer(MCPServer):
    """MCP server providing basic tools for web search and code execution."""
    
    def __init__(self):
        # Define tool configurations
        tools_config = [
            {
                "name": "web_search",
                "description": "Search the web for information based on a query",
                "input_schema": {
                    "query": {"type": "string", "description": "The search query string"}
                },
                "output_schema": {
                    "results": {"type": "string", "description": "Search results as JSON string"}
                },
                "function_name": "web_search"
            },
            {
                "name": "code_execution",
                "description": "Execute a given snippet of Python code",
                "input_schema": {
                    "code": {"type": "string", "description": "A snippet of Python code to execute"}
                },
                "output_schema": {
                    "result": {"type": "string", "description": "Result of the code execution"}
                },
                "function_name": "code_execution"
            }
        ]
        
        # Define resource configurations
        resources_config = [
            {
                "uri_pattern": "search://web/",
                "description": "Access cached web search results",
                "function_name": "get_search_resource"
            }
        ]
        
        super().__init__(
            name="basic_tools",
            description="Provides essential tools for web search and code execution"
        )
        self.tools_config = tools_config
        self.resources_config = resources_config
    
    def initialize(self):
        """Initialize the server with tools and resources."""
        # Register tools with assertion-based validation
        for tool_config in self.tools_config:
            assert_not_none(tool_config.get("function_name"), "tool function_name")
            function_name = tool_config["function_name"]
            
            require(
                hasattr(self, function_name),
                f"Function {function_name} not found in server class",
                context={"tool_config": tool_config}
            )
            
            function = getattr(self, function_name)
            
            self.register_tool(
                MCPToolDefinition(
                    name=tool_config["name"],
                    description=tool_config["description"],
                    input_schema=tool_config["input_schema"],
                    output_schema=tool_config["output_schema"],
                    function=function
                )
            )
        
        # Register resources with assertion-based validation
        for resource_config in self.resources_config:
            assert_not_none(resource_config.get("function_name"), "resource function_name")
            function_name = resource_config["function_name"]
            
            require(
                hasattr(self, function_name),
                f"Function {function_name} not found in server class",
                context={"resource_config": resource_config}
            )
            
            function = getattr(self, function_name)
            
            self.register_resource(
                MCPResourceDefinition(
                    uri_pattern=resource_config["uri_pattern"],
                    description=resource_config["description"],
                    function=function
                )
            )
        
        logger.info("BasicToolsServer initialized successfully")
    
    def web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search the web for information.
        
        Args:
            args: Dictionary containing the search query
            
        Returns:
            Dictionary containing the search results
        """
        assert_not_none(args, "web search arguments")
        
        query = args.get("query", "")
        assert_non_empty(query.strip(), "search query")
        
        logger.info(f"Executing web_search_tool with query: '{query}'")
        
        # Check if Tavily API key is available
        if not settings.tavily_api_key:
            logger.warning("TAVILY_API_KEY not set. Using mocked search results.")
            # Mock search results for common queries
            if "capital of france" in query.lower():
                return {"results": "The capital of France is Paris."}
            return {"results": f"Mocked search results for: {query}. (TAVILY_API_KEY not set)"}
        
        # Use Tavily API for real search
        from tavily import TavilyClient
        
        tavily = TavilyClient(api_key=settings.tavily_api_key)
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        
        assert_not_none(response, "Tavily API response")
        assert_type(response, dict, "Tavily API response")
        
        results_data = response.get("results", [])
        assert_type(results_data, list, "Tavily search results")
        
        results = json.dumps([
            {"url": res["url"], "content": res["content"]} 
            for res in results_data
        ])
        
        logger.info(f"Web search successful.")
        return {"results": results}
    
    def code_execution(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a given snippet of Python code.
        
        Args:
            args: Dictionary containing the code to execute
            
        Returns:
            Dictionary containing the execution result
        """
        assert_not_none(args, "code execution arguments")
        
        code = args.get("code", "")
        assert_non_empty(code.strip(), "code to execute")
        
        logger.info(f"Executing code_execution_tool.")
        logger.warning("SECURITY WARNING: Using simplified sandbox. Use proper sandboxing in production.")
        
        resolved_code = code.strip()
        
        # Handle print statements
        if resolved_code.startswith("print("):
            inner_content_str = resolved_code[len("print("):-1]
            if (inner_content_str.startswith("'") and inner_content_str.endswith("'")) or \
               (inner_content_str.startswith('"') and inner_content_str.endswith('"')):
                result = inner_content_str[1:-1]
            else:
                # Use simple_eval for safer expression evaluation
                result = str(simple_eval(inner_content_str))
            logger.info(f"Code execution (print) successful.")
            return {"result": result}
        else:
            # Use simple_eval for safer expression evaluation
            result = str(simple_eval(resolved_code))
            logger.info(f"Code execution (eval) successful.")
            return {"result": f"Execution Result: {result}"}
            
    def get_search_resource(self, uri: str) -> Dict[str, Any]:
        """
        Get cached search results by URI.
        
        Args:
            uri: URI of the search resource
            
        Returns:
            Dictionary containing the search results
        """
        assert_not_none(uri, "search resource URI")
        assert_non_empty(uri.strip(), "search resource URI")
        
        # Extract query from URI
        query = uri.replace("search://web/", "")
        require(
            len(query) > 0,
            "Invalid search URI. Expected format: search://web/your-query",
            context={"uri": uri}
        )
        
        # Simple implementation just performs a new search
        # In a real implementation, this might check a cache first
        return self.web_search({"query": query})


def main():
    """Main entry point for the basic tools server."""
    server = BasicToolsServer()
    server.start()

if __name__ == "__main__":
    main()
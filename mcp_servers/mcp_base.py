# mcp_servers/mcp_base.py
"""
Base framework for Model Context Protocol (MCP) servers with Zero-Space Programming.

This module provides the common functionality needed by all MCP servers,
including communication via standard input/output or HTTP Server-Sent Events (SSE),
enhanced with assertion-based validation.
"""

import json
import sys
import os
import time
import uuid
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union, TypeVar, Generic, Type
from dataclasses import dataclass, asdict, field

# Import shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    # Import from tools package (new location)
    from tools.logging import configure_logging
    from tools.assertions import (
        require, ensure, invariant, assert_not_none, assert_type,
        contract, GaiaAssertionError
    )
except ImportError:
    # Fallback to basic logging if imports fail
    import logging
    
    def configure_logging(name, **kwargs):
        """Configure and return a basic logger."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        # Add console handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    # Fallback assertion functions
    def require(condition, message, context=None):
        assert condition, message
    
    def ensure(condition, message, context=None):
        assert condition, message
    
    def assert_not_none(value, name, context=None):
        assert value is not None, f"{name} must not be None"
    
    def assert_type(value, expected_type, name, context=None):
        assert isinstance(value, expected_type), f"{name} must be of type {expected_type.__name__}"
    
    def contract(preconditions=None, postconditions=None):
        def decorator(func):
            return func
        return decorator

# Configure logging using the shared logging configuration
logger = configure_logging('mcp_server', log_file='mcp_server.log')

# Type definitions for MCP protocol
T = TypeVar('T')
ToolResult = Dict[str, Any]
ResourceResult = Dict[str, Any]

@dataclass
class MCPToolDefinition:
    """Definition of an MCP tool with assertion validation."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    function: Callable[[Dict[str, Any]], Any]
    
    def __post_init__(self):
        """Validate tool definition with assertions."""
        assert_not_none(self.name, "name", {"operation": "tool_definition"})
        assert_type(self.name, str, "name", {"operation": "tool_definition"})
        require(
            len(self.name.strip()) > 0,
            "Tool name must not be empty",
            context={"name": self.name}
        )
        
        assert_not_none(self.description, "description", {"operation": "tool_definition"})
        assert_type(self.description, str, "description", {"operation": "tool_definition"})
        
        assert_not_none(self.input_schema, "input_schema", {"operation": "tool_definition"})
        assert_type(self.input_schema, dict, "input_schema", {"operation": "tool_definition"})
        
        assert_not_none(self.output_schema, "output_schema", {"operation": "tool_definition"})
        assert_type(self.output_schema, dict, "output_schema", {"operation": "tool_definition"})
        
        assert_not_none(self.function, "function", {"operation": "tool_definition"})
        require(
            callable(self.function),
            "Function must be callable",
            context={"function": str(self.function)}
        )

@dataclass
class MCPResourceDefinition:
    """Definition of an MCP resource with assertion validation."""
    uri_pattern: str
    description: str
    function: Callable[[str], Any]
    
    def __post_init__(self):
        """Validate resource definition with assertions."""
        assert_not_none(self.uri_pattern, "uri_pattern", {"operation": "resource_definition"})
        assert_type(self.uri_pattern, str, "uri_pattern", {"operation": "resource_definition"})
        require(
            len(self.uri_pattern.strip()) > 0,
            "URI pattern must not be empty",
            context={"uri_pattern": self.uri_pattern}
        )
        
        assert_not_none(self.description, "description", {"operation": "resource_definition"})
        assert_type(self.description, str, "description", {"operation": "resource_definition"})
        
        assert_not_none(self.function, "function", {"operation": "resource_definition"})
        require(
            callable(self.function),
            "Function must be callable",
            context={"function": str(self.function)}
        )

@dataclass
class MCPServerInfo:
    """Information about the MCP server with assertion validation."""
    name: str
    description: str
    version: str = "1.0.0"
    tools: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate server info with assertions."""
        assert_not_none(self.name, "name", {"operation": "server_info"})
        assert_type(self.name, str, "name", {"operation": "server_info"})
        require(
            len(self.name.strip()) > 0,
            "Server name must not be empty",
            context={"name": self.name}
        )
        
        assert_not_none(self.description, "description", {"operation": "server_info"})
        assert_type(self.description, str, "description", {"operation": "server_info"})
        
        assert_type(self.version, str, "version", {"operation": "server_info"})
        assert_type(self.tools, list, "tools", {"operation": "server_info"})
        assert_type(self.resources, list, "resources", {"operation": "server_info"})

class MCPServer(ABC):
    """Base class for all MCP servers with assertion validation."""
    
    def __init__(self, name: str, description: str):
        assert_type(name, str, "name", {"operation": "mcp_server_init"})
        require(
            len(name.strip()) > 0,
            "Server name must not be empty",
            context={"name": name}
        )
        
        assert_type(description, str, "description", {"operation": "mcp_server_init"})
        require(
            len(description.strip()) > 0,
            "Server description must not be empty",
            context={"description": description}
        )
        
        self.name = name
        self.description = description
        self.version = "1.0.0"
        self.tools: Dict[str, MCPToolDefinition] = {}
        self.resources: Dict[str, MCPResourceDefinition] = {}
        self.server_info = MCPServerInfo(name=name, description=description)
    
    @contract(
        preconditions=[lambda self, tool_def: tool_def is not None],
        postconditions=[]
    )
    def register_tool(self, tool_def: MCPToolDefinition) -> None:
        """Register a tool with this MCP server using assertion validation."""
        assert_not_none(tool_def, "tool_def", {"operation": "tool_registration"})
        assert_type(tool_def, MCPToolDefinition, "tool_def", {"operation": "tool_registration"})
        
        require(
            tool_def.name not in self.tools,
            "Tool name must be unique",
            context={"tool_name": tool_def.name, "existing_tools": list(self.tools.keys())}
        )
        
        self.tools[tool_def.name] = tool_def
        tool_info = {
            "name": tool_def.name,
            "description": tool_def.description,
            "input_schema": tool_def.input_schema,
            "output_schema": tool_def.output_schema
        }
        self.server_info.tools.append(tool_info)
        logger.info(f"Registered tool: {tool_def.name}")
    
    @contract(
        preconditions=[lambda self, resource_def: resource_def is not None],
        postconditions=[]
    )
    def register_resource(self, resource_def: MCPResourceDefinition) -> None:
        """Register a resource with this MCP server using assertion validation."""
        assert_not_none(resource_def, "resource_def", {"operation": "resource_registration"})
        assert_type(resource_def, MCPResourceDefinition, "resource_def", {"operation": "resource_registration"})
        
        require(
            resource_def.uri_pattern not in self.resources,
            "Resource URI pattern must be unique",
            context={"uri_pattern": resource_def.uri_pattern, "existing_patterns": list(self.resources.keys())}
        )
        
        self.resources[resource_def.uri_pattern] = resource_def
        resource_info = {
            "uri_pattern": resource_def.uri_pattern,
            "description": resource_def.description
        }
        self.server_info.resources.append(resource_info)
        logger.info(f"Registered resource: {resource_def.uri_pattern}")
    
    @contract(
        preconditions=[],
        postconditions=[lambda result, self: isinstance(result, dict)]
    )
    def handle_info_request(self) -> Dict[str, Any]:
        """Handle a request for server information with assertion validation."""
        result = asdict(self.server_info)
        
        ensure(
            isinstance(result, dict) and "name" in result,
            "Server info must be valid dictionary",
            context={"server_name": self.name}
        )
        
        return result
    
    @contract(
        preconditions=[
            lambda self, tool_name, arguments: isinstance(tool_name, str),
            lambda self, tool_name, arguments: isinstance(arguments, dict)
        ],
        postconditions=[lambda result, self, tool_name, arguments: isinstance(result, dict)]
    )
    def handle_tool_request(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Handle a tool execution request with assertion validation."""
        assert_type(tool_name, str, "tool_name", {"operation": "tool_execution"})
        assert_type(arguments, dict, "arguments", {"operation": "tool_execution"})
        
        require(
            tool_name in self.tools,
            "Tool must exist",
            context={"tool_name": tool_name, "available_tools": list(self.tools.keys())}
        )
        
        tool = self.tools[tool_name]
        logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")
        result = tool.function(arguments)
        logger.info(f"Tool {tool_name} executed successfully")
        
        response = {"result": result}
        
        ensure(
            isinstance(response, dict),
            "Tool response must be dictionary",
            context={"tool_name": tool_name}
        )
        
        return response
    
    @contract(
        preconditions=[lambda self, uri: isinstance(uri, str) and len(uri.strip()) > 0],
        postconditions=[lambda result, self, uri: isinstance(result, dict)]
    )
    def handle_resource_request(self, uri: str) -> ResourceResult:
        """Handle a resource access request with assertion validation."""
        assert_type(uri, str, "uri", {"operation": "resource_access"})
        require(
            len(uri.strip()) > 0,
            "URI must not be empty",
            context={"uri": uri}
        )
        
        # Find matching resource handler
        for pattern, resource_def in self.resources.items():
            # Simple pattern matching for now, could be enhanced with regex
            if uri.startswith(pattern.split("*")[0]):
                logger.info(f"Accessing resource: {uri}")
                result = resource_def.function(uri)
                logger.info(f"Resource {uri} accessed successfully")
                
                response = {"result": result}
                
                ensure(
                    isinstance(response, dict),
                    "Resource response must be dictionary",
                    context={"uri": uri, "pattern": pattern}
                )
                
                return response
        
        # No matching resource found
        error_response = {"error": f"Resource not found: {uri}"}
        logger.error(f"Resource not found: {uri}")
        return error_response
    
    @contract(
        preconditions=[],
        postconditions=[lambda result, self: isinstance(result, list)]
    )
    def get_tool_manifest(self) -> List[Dict[str, Any]]:
        """Get a standardized manifest of all tools with assertion validation."""
        manifest = []
        for tool_name, tool_def in self.tools.items():
            manifest.append({
                "name": tool_name,
                "description": tool_def.description,
                "server_name": self.name,
                "input_schema": tool_def.input_schema,
                "output_schema": tool_def.output_schema
            })
        
        ensure(
            isinstance(manifest, list),
            "Tool manifest must be list",
            context={"server_name": self.name, "tool_count": len(self.tools)}
        )
        
        return manifest
    
    @contract(
        preconditions=[],
        postconditions=[lambda result, self: isinstance(result, dict)]
    )
    def handle_manifest_request(self) -> Dict[str, Any]:
        """Handle a request for the tool manifest with assertion validation."""
        result = {"manifest": self.get_tool_manifest()}
        
        ensure(
            isinstance(result, dict) and "manifest" in result,
            "Manifest response must be valid dictionary",
            context={"server_name": self.name}
        )
        
        return result
    
    @contract(
        preconditions=[lambda self, request: isinstance(request, dict)],
        postconditions=[lambda result, self, request: isinstance(result, dict)]
    )
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request with assertion validation."""
        assert_type(request, dict, "request", {"operation": "request_handling"})
        
        request_type = request.get("type")
        request_id = request.get("id", str(uuid.uuid4()))
        
        response = {
            "id": request_id,
            "type": f"{request_type}_response",
            "status": "success"
        }
        
        if request_type == "info":
            response["content"] = self.handle_info_request()
        elif request_type == "manifest":
            response["content"] = self.handle_manifest_request()
        elif request_type == "tool":
            tool_name = request.get("tool", "")
            arguments = request.get("arguments", {})
            result = self.handle_tool_request(tool_name, arguments)
            
            # Check if the result contains an error
            if "error" in result:
                response["status"] = "error"
                response["error"] = result["error"]
            else:
                response["content"] = result
        elif request_type == "resource":
            uri = request.get("uri", "")
            result = self.handle_resource_request(uri)
            
            # Check if the result contains an error
            if "error" in result:
                response["status"] = "error"
                response["error"] = result["error"]
            else:
                response["content"] = result
        elif request_type == "ping":
            # Handle ping requests - simple acknowledgment
            response["type"] = "ping_response"
            response["content"] = {"timestamp": request.get("timestamp", time.time())}
        else:
            response["status"] = "error"
            response["error"] = f"Unknown request type: {request_type}"
        
        ensure(
            isinstance(response, dict) and "status" in response,
            "Response must be valid dictionary with status",
            context={"request_type": request_type, "request_id": request_id}
        )
        
        return response
    
    @contract(
        preconditions=[],
        postconditions=[]
    )
    def start_stdio(self) -> None:
        """Start the MCP server using standard input/output communication with assertion validation."""
        logger.info(f"Starting MCP server {self.name} in stdio mode")
        
        # Send initial manifest message when the server starts
        initial_manifest = {
            "id": str(uuid.uuid4()),
            "type": "manifest_announcement",
            "status": "success",
            "content": {
                "server_name": self.name,
                "server_description": self.description,
                "manifest": self.get_tool_manifest()
            }
        }
        # Validate manifest serialization with assertions
        manifest_json = json.dumps(initial_manifest)
        ensure(
            isinstance(manifest_json, str) and len(manifest_json) > 0,
            "Manifest JSON must be valid non-empty string",
            context={"server_name": self.name, "tool_count": len(self.tools)}
        )
        
        sys.stdout.write(manifest_json + "\n")
        sys.stdout.flush()
        logger.info(f"Sent initial manifest with {len(self.tools)} tools")
        
        # Process incoming requests
        for line in sys.stdin:
            line_stripped = line.strip()
            require(
                len(line_stripped) > 0,
                "Request line must not be empty",
                context={"server_name": self.name}
            )
            
            request = json.loads(line_stripped)
            ensure(
                isinstance(request, dict),
                "Request must be valid JSON dictionary",
                context={"server_name": self.name, "request_line": line_stripped}
            )
            
            response = self.handle_request(request)
            ensure(
                isinstance(response, dict),
                "Response must be dictionary",
                context={"server_name": self.name, "request": request}
            )
            
            response_json = json.dumps(response)
            ensure(
                isinstance(response_json, str) and len(response_json) > 0,
                "Response JSON must be valid non-empty string",
                context={"server_name": self.name, "response": response}
            )
            
            sys.stdout.write(response_json + "\n")
            sys.stdout.flush()
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the MCP server with tools and resources."""
        pass
    
    def start(self, mode: str = "stdio") -> None:
        """Start the MCP server in the specified mode with assertion validation."""
        assert_type(mode, str, "mode", {"operation": "server_start"})
        
        self.initialize()
        
        # Convert to 'assert' as in the rest of a code
        if mode == "stdio":
            self.start_stdio()
        else:
            assert False, f"Unsupported mode: {mode} (Supported modes are: ['stdio'])"
            assert False, f"Unsupported mode: {mode} (Supported modes are: ['stdio'])"

class EnhancedMCPServer(MCPServer):
    """Enhanced base class for MCP servers with common functionality and assertion validation."""
    
    @contract(
        preconditions=[
            lambda self, name, description, tools_config, resources_config: isinstance(name, str) and len(name.strip()) > 0,
            lambda self, name, description, tools_config, resources_config: isinstance(description, str) and len(description.strip()) > 0,
            lambda self, name, description, tools_config, resources_config: tools_config is None or isinstance(tools_config, list),
            lambda self, name, description, tools_config, resources_config: resources_config is None or isinstance(resources_config, list)
        ],
        postconditions=[]
    )
    def __init__(self, name: str, description: str, tools_config: List[Dict] = None, resources_config: List[Dict] = None):
        """Initialize the server with assertion validation."""
        super().__init__(name, description)
        
        assert_type(tools_config, list, "tools_config", {"operation": "enhanced_server_init"})
        
        assert_type(resources_config, list, "resources_config", {"operation": "enhanced_server_init"})
        
        self.tools_config = tools_config or []
        self.resources_config = resources_config or []
    
    def initialize(self):
        """Initialize the server with tools and resources from configuration using assertions."""
        # Register tools from configuration
        for tool_config in self.tools_config:
            assert_type(tool_config, dict, "tool_config", {"operation": "tool_config_registration"})
            self.register_tool_from_config(tool_config)
        
        # Register resources from configuration
        for resource_config in self.resources_config:
            assert_type(resource_config, dict, "resource_config", {"operation": "resource_config_registration"})
            self.register_resource_from_config(resource_config)
        
        # Call custom initialization
        self.custom_initialize()
    
    @contract(
        preconditions=[lambda self, config: isinstance(config, dict)],
        postconditions=[]
    )
    def register_tool_from_config(self, config: Dict):
        """Register a tool from configuration with assertion validation."""
        assert_type(config, dict, "config", {"operation": "tool_config_registration"})
        
        require(
            "function_name" in config,
            "Tool config must contain function_name",
            context={"config_keys": list(config.keys())}
        )
        
        require(
            "name" in config,
            "Tool config must contain name",
            context={"config_keys": list(config.keys())}
        )
        
        function_name = config["function_name"]
        require(
            hasattr(self, function_name),
            "Function must exist on server instance",
            context={"function_name": function_name, "server_class": self.__class__.__name__}
        )
        
        function = getattr(self, function_name)
        
        self.register_tool(
            MCPToolDefinition(
                name=config["name"],
                description=config.get("description", ""),
                input_schema=config.get("input_schema", {}),
                output_schema=config.get("output_schema", {}),
                function=function
            )
        )
    
    @contract(
        preconditions=[lambda self, config: isinstance(config, dict)],
        postconditions=[]
    )
    def register_resource_from_config(self, config: Dict):
        """Register a resource from configuration with assertion validation."""
        assert_type(config, dict, "config", {"operation": "resource_config_registration"})
        
        require(
            "function_name" in config,
            "Resource config must contain function_name",
            context={"config_keys": list(config.keys())}
        )
        
        require(
            "uri_pattern" in config,
            "Resource config must contain uri_pattern",
            context={"config_keys": list(config.keys())}
        )
        
        function_name = config["function_name"]
        require(
            hasattr(self, function_name),
            "Function must exist on server instance",
            context={"function_name": function_name, "server_class": self.__class__.__name__}
        )
        
        function = getattr(self, function_name)
        
        self.register_resource(
            MCPResourceDefinition(
                uri_pattern=config["uri_pattern"],
                description=config.get("description", ""),
                function=function
            )
        )
    
    def custom_initialize(self):
        """Custom initialization to be implemented by derived classes."""
        pass
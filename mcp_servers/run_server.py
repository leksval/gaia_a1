#!/usr/bin/env python3
"""
MCP Server Runner
Dynamic runner for individual MCP server files using config-based server discovery.
"""

import sys
import os
import importlib.util
import glob
from pathlib import Path

# Add project root to Python path for proper imports when run as subprocess
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Validate that project root exists and is accessible
assert os.path.exists(project_root), f"Project root must exist: {project_root}"
assert os.path.isdir(project_root), f"Project root must be a directory: {project_root}"

# Import assertions for error handling
from tools.assertions import (
    require, ensure, assert_not_none, assert_type, assert_non_empty,
    ConfigurationError, NetworkError, ProcessingError
)

def auto_discover_servers():
    """Auto-discover available MCP servers from filesystem."""
    # Get mcp_servers directory
    mcp_servers_dir = Path(__file__).parent
    assert_not_none(mcp_servers_dir, "mcp_servers directory")
    
    # Auto-discover server files
    server_files = glob.glob(str(mcp_servers_dir / "*_server.py"))
    discovered_mapping = {}

    for server_file in server_files:
        filename = os.path.basename(server_file)
        
        if filename.endswith("_server.py") and not filename.startswith("run_"):
            # Extract server name (remove _server.py suffix)
            server_name = filename[:-10]  # Remove "_server.py"
            discovered_mapping[server_name] = filename

    return discovered_mapping

def main():
    require(
        len(sys.argv) == 2,
        "Usage: python run_server.py <server_name>",
        context={"args_provided": len(sys.argv) - 1, "expected": 1}
    )
    
    server_name = sys.argv[1]
    assert_non_empty(server_name, "server_name")
    
    # Get server mapping from auto-discovery (same logic as config)
    server_file_mapping = auto_discover_servers()
    assert_not_none(server_file_mapping, "server file mapping")
    
    require(
        server_name in server_file_mapping,
        f"Unknown server: {server_name}",
        context={
            "server_name": server_name,
            "available_servers": list(server_file_mapping.keys())
        }
    )
    
    # Get the server file path
    server_file = server_file_mapping[server_name]
    server_path = Path(__file__).parent / server_file
    
    require(
        server_path.exists(),
        f"Server file not found: {server_path}",
        context={"server_path": str(server_path)}
    )
    
    # Load and run the server module
    spec = importlib.util.spec_from_file_location(server_name, server_path)
    require(
        spec is not None and spec.loader is not None,
        f"Failed to load server spec: {server_path}",
        context={"server_path": str(server_path)}
    )
        
    module = importlib.util.module_from_spec(spec)
    sys.modules[server_name] = module
    spec.loader.exec_module(module)
    
    # Look for main function or server startup
    if hasattr(module, 'main'):
        module.main()
    elif hasattr(module, 'run_server'):
        module.run_server()
    elif hasattr(module, 'start_server'):
        module.start_server()
    elif hasattr(module, 'run'):
        module.run()
    else:
        available_functions = [attr for attr in dir(module) if not attr.startswith('_')]
        require(
            False,
            f"No main/run_server/start_server/run function found in {server_file}",
            context={
                "server_file": server_file,
                "available_functions": available_functions
            }
        )

if __name__ == "__main__":
    main()
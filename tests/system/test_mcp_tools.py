#!/usr/bin/env python3
"""Test MCP tool discovery after fixes."""

from tools.mcp_client import MCPServerManager
import asyncio
import sys

async def test_mcp_tools():
    print('Testing MCP tool discovery...')
    client = MCPServerManager()
    
    # Start servers first
    print('\n=== Starting MCP Servers ===')
    try:
        start_results = client.start_all_servers()
        print(f'Server start results: {start_results}')
        
        # Wait a moment for servers to initialize
        import time
        time.sleep(2)
        
        # Check which servers are running
        running_servers = client.get_running_servers()
        print(f'Running servers: {running_servers}')
        
    except Exception as e:
        print(f'Error starting servers: {e}')
    
    # Test all available tools
    print('\n=== Testing All Available Tools ===')
    tools = []
    try:
        tools = client.get_available_tools()
        print(f'Total tools found: {len(tools)}')
        
        # Group tools by server
        servers = {}
        for tool in tools:
            server_name = tool.get("server_name", "unknown")
            if server_name not in servers:
                servers[server_name] = []
            servers[server_name].append(tool)
        
        # Display tools by server
        for server_name, server_tools in servers.items():
            print(f'\n=== {server_name} Server ===')
            print(f'Tools: {len(server_tools)} found')
            for tool in server_tools:
                name = tool.get("name", "unknown")
                desc = tool.get("description", "no description")[:80]
                print(f'  - {name}: {desc}...')
                
    except Exception as e:
        print(f'Error getting tools: {e}')
    
    # Test a simple tool if available
    if len(tools) > 0:
        print('\n=== Testing Tool Usage ===')
        try:
            # Try to use a math calculator tool which should work with simple input
            math_tool = None
            for tool in tools:
                if tool.get("server_name") == "math_calculator" and tool.get("name") == "calculate":
                    math_tool = tool
                    break
            
            if math_tool:
                tool_name = math_tool.get("name")
                server_name = math_tool.get("server_name")
                print(f'Testing tool: {tool_name} from {server_name}')
                
                # Use a simple calculation with proper arguments
                result = client.use_tool(server_name, tool_name, {"expression": "2 + 2"})
                print(f'Tool result: {result}')
            else:
                print('No math calculator tool found to test')
                
        except Exception as e:
            print(f'Error testing tool usage: {e}')

if __name__ == '__main__':
    asyncio.run(test_mcp_tools())
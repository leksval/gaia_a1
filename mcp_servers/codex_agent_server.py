#!/usr/bin/env python3
"""
LangChain-powered Codex Agent MCP Server

Simple, clean implementation using LangChain's PythonREPLTool for secure code execution.
"""

import logging
from typing import Dict, Any, List, Union

from langchain_experimental.tools import PythonREPLTool
from mcp_servers.mcp_base import MCPServer, MCPToolDefinition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codex_agent_server")


class CodexAgentServer(MCPServer):
    """LangChain-powered Codex Agent MCP Server."""
    
    def __init__(self):
        super().__init__(
            name="codex_agent",
            description="LangChain-powered secure code execution and problem solving agent"
        )
        self.python_repl = PythonREPLTool()
        
    def initialize(self):
        """Initialize the Codex agent with LangChain tools."""
        
        self.register_tool(
            MCPToolDefinition(
                name="execute_code",
                description="Execute Python code securely using LangChain's PythonREPLTool",
                input_schema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                        "description": {"type": "string", "description": "Description of the code"}
                    },
                    "required": ["code"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "string", "description": "Execution result"},
                        "success": {"type": "boolean", "description": "Whether execution was successful"},
                        "error": {"type": "string", "description": "Error message if failed"}
                    }
                },
                function=self.execute_code
            )
        )
        
        self.register_tool(
            MCPToolDefinition(
                name="compute_math",
                description="Perform mathematical computations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression to solve"},
                        "operation_type": {"type": "string", "default": "calculate"}
                    },
                    "required": ["expression"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "string", "description": "Computation result"},
                        "success": {"type": "boolean", "description": "Whether computation was successful"},
                        "error": {"type": "string", "description": "Error message if failed"}
                    }
                },
                function=self.compute_math
            )
        )
        
        self.register_tool(
            MCPToolDefinition(
                name="analyze_data",
                description="Analyze data sets and perform statistical computations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Data array to analyze"},
                        "analysis_type": {"type": "string", "default": "descriptive"}
                    },
                    "required": ["data"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "analysis_results": {"type": "string", "description": "Analysis results"},
                        "success": {"type": "boolean", "description": "Whether analysis was successful"},
                        "error": {"type": "string", "description": "Error message if failed"}
                    }
                },
                function=self.analyze_data
            )
        )
        
        self.register_tool(
            MCPToolDefinition(
                name="solve_problem",
                description="Solve problems intelligently using code generation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "problem": {"type": "string", "description": "Problem to solve"},
                        "context": {"type": "string", "description": "Additional context"}
                    },
                    "required": ["problem"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "solution": {"type": "string", "description": "Problem solution"},
                        "success": {"type": "boolean", "description": "Whether problem was solved"},
                        "error": {"type": "string", "description": "Error message if failed"}
                    }
                },
                function=self.solve_problem
            )
        )
        
        logger.info("LangChain Codex Agent Server initialized")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema
            }
            for tool in self.tools.values()
        ]
    
    def execute_code(self, code: str, description: str = "") -> Dict[str, Any]:
        """Execute code using LangChain's PythonREPLTool."""
        try:
            result = self.python_repl.run(code)
            return {
                "result": str(result),
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "result": "",
                "success": False,
                "error": str(e)
            }
    
    def compute_math(self, expression: str, operation_type: str = "calculate") -> Dict[str, Any]:
        """Perform mathematical computations."""
        code = f"""
import math
import statistics

expression = "{expression}"

try:
    # Replace common math notation
    expr = expression.replace('^', '**').replace('sqrt', 'math.sqrt')
    result = eval(expr)
    print(f"Result: {{result}}")
except Exception as e:
    print(f"Error: {{e}}")
"""
        return self.execute_code(code, f"Math: {expression}")
    
    def analyze_data(self, data: List[Union[int, float]], analysis_type: str = "descriptive") -> Dict[str, Any]:
        """Analyze data with statistical methods."""
        code = f"""
import statistics

data = {data}
print(f"Analyzing {{len(data)}} data points")

if data:
    print(f"Mean: {{statistics.mean(data)}}")
    print(f"Median: {{statistics.median(data)}}")
    print(f"Min: {{min(data)}}")
    print(f"Max: {{max(data)}}")
    if len(data) > 1:
        print(f"Std Dev: {{statistics.stdev(data)}}")
"""
        result = self.execute_code(code, f"Data analysis: {analysis_type}")
        return {
            "analysis_results": result.get("result", ""),
            "success": result.get("success", False),
            "error": result.get("error")
        }
    
    def solve_problem(self, problem: str, context: str = "") -> Dict[str, Any]:
        """Solve problems intelligently."""
        code = f"""
import re
import math

problem = "{problem}"
print(f"Solving: {{problem}}")

# Extract numbers
numbers = [float(x) for x in re.findall(r'\\d+(?:\\.\\d+)?', problem)]
print(f"Numbers found: {{numbers}}")

# Simple problem solving
if "sum" in problem.lower() or "add" in problem.lower():
    result = sum(numbers)
    print(f"Sum: {{result}}")
elif "product" in problem.lower() or "multiply" in problem.lower():
    result = 1
    for n in numbers: result *= n
    print(f"Product: {{result}}")
elif len(numbers) >= 2:
    print(f"{{numbers[0]}} + {{numbers[1]}} = {{numbers[0] + numbers[1]}}")
"""
        result = self.execute_code(code, f"Problem: {problem}")
        return {
            "solution": result.get("result", ""),
            "success": result.get("success", False),
            "error": result.get("error")
        }
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool calls."""
        try:
            if tool_name == "execute_code":
                return self.execute_code(arguments["code"], arguments.get("description", ""))
            elif tool_name == "compute_math":
                return self.compute_math(arguments["expression"], arguments.get("operation_type", "calculate"))
            elif tool_name == "analyze_data":
                return self.analyze_data(arguments["data"], arguments.get("analysis_type", "descriptive"))
            elif tool_name == "solve_problem":
                return self.solve_problem(arguments["problem"], arguments.get("context", ""))
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def main():
    """Main entry point."""
    server = CodexAgentServer()
    server.start()


if __name__ == "__main__":
    main()
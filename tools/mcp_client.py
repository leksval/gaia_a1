"""
MCP (Model Context Protocol) Client

This module provides functionality for:
1. Managing MCP servers (starting, stopping, discovering tools)
2. Using MCP tools through a simple client interface
3. Providing LangChain-compatible tool definitions

Enhanced with zero-space programming assertions and LangFuse integration.
"""

import os
import logging
import subprocess
import time
import json
import signal
import atexit
import uuid
import threading
import select
from typing import Dict, List, Optional, Any

from langchain_core.tools import tool

# Import from new directory structure
from config.config import settings
from tools.assertions import (
    require, ensure, invariant, assert_not_none, assert_type,
    assert_non_empty, assert_mcp_server_response, assert_in_range,
    GaiaAssertionError, NetworkError, ProcessingError
)
from tools.langfuse_monitor import LangfuseMonitor
from tools.mcp_monitoring import (
    log_manifest_operation, log_tool_usage, get_mcp_system_health,
    get_performance_metrics, generate_mcp_recommendations
)

# Set up logging
logger = logging.getLogger(__name__)

# Global instance for singleton pattern
_mcp_manager_instance = None


def sanitize_for_json(data):
    """
    Recursively sanitize data to ensure JSON serialization compatibility.
    Converts non-serializable types to strings with assertion validation.
    """
    assert_not_none(data, "data", {"operation": "json_sanitization"})
    
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    elif isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_for_json(item) for item in data]
    else:
        return str(data)


class MCPServerManager:
    """
    Manages MCP servers with assertion-based validation and zero-space programming.
    
    This class handles starting, stopping, and communicating with MCP servers,
    providing a unified interface for tool discovery and execution.
    """
    
    def __init__(self):
        """Initialize the MCP server manager with assertion validation."""
        self.servers: Dict[str, Dict] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.tool_manifests: Dict[str, List[Dict]] = {}
        self.enabled = getattr(settings, 'enable_mcp_servers', True)
        
        # Get list of enabled servers from configuration
        require(
            hasattr(settings, 'enabled_mcp_servers') and settings.enabled_mcp_servers,
            "MCP servers must be configured in settings",
            context={"operation": "mcp_manager_init"}
        )
        
        enabled_servers = settings.enabled_mcp_servers.split(',')
        assert_non_empty(enabled_servers, "enabled_servers", {"operation": "mcp_manager_init"})
        
        # Register servers from configuration
        for server_name in enabled_servers:
            server_name = server_name.strip()
            
            # Security: Validate server name is in the allowed list from config
            require(
                (server_name in settings.allowed_server_mapping or
                 server_name in settings.allowed_server_mapping.values()),
                f"Server '{server_name}' not in allowed server mapping",
                context={"server_name": server_name, "allowed_servers": list(settings.allowed_server_mapping.keys())}
            )
            
            # Get the actual server file name
            server_file = settings.allowed_server_mapping.get(server_name, server_name)
            
            self.servers[server_name] = {
                "enabled": True,
                "command_args": ["python", "mcp_servers/run_server.py", server_name],
                "server_file": server_file,
                "process": None
            }
        
        logger.info(f"Initialized MCP manager with {len(self.servers)} servers: {list(self.servers.keys())}")
        
        # Ensure initialization completed successfully
        ensure(
            hasattr(self, 'servers') and hasattr(self, 'enabled'),
            "MCP manager must be properly initialized",
            context={"operation": "mcp_manager_init"}
        )
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def start_all_servers(self) -> Dict[str, bool]:
        """Start all enabled MCP servers."""
        require(
            self.enabled,
            "MCP servers must be enabled to start",
            context={"operation": "start_all_servers"}
        )
        
        results = {}
        for server_name in self.servers:
            results[server_name] = self.start_server(server_name)
        
        successful_starts = [name for name, success in results.items() if success]
        logger.info(f"Started {len(successful_starts)}/{len(self.servers)} MCP servers successfully")
        
        ensure(
            len(successful_starts) > 0,
            "At least one MCP server must start successfully",
            context={"results": results}
        )
        
        return results
    
    def start_server(self, server_name: str) -> bool:
        """Start a specific MCP server with assertion validation."""
        assert_non_empty(server_name, "server_name", {"operation": "start_server"})
        
        require(
            server_name in self.servers,
            f"Server '{server_name}' must be registered",
            context={"server_name": server_name, "registered_servers": list(self.servers.keys())}
        )
        
        server_config = self.servers[server_name]
        
        # Check if server is already running
        if self.is_server_running(server_name):
            logger.info(f"Server '{server_name}' is already running")
            return True
        
        logger.info(f"Starting MCP server: {server_name}")
        
        # Start the server process
        process = subprocess.Popen(
            server_config["command_args"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        # Store the process
        server_config["process"] = process
        self.processes[server_name] = process
        
        # Wait a moment for the server to start
        time.sleep(1)
        
        # Check if the process is still running
        if process.poll() is None:
            logger.info(f"Successfully started MCP server: {server_name}")
            
            # Send keep-alive ping to maintain connection
            self._send_keep_alive_ping(server_name)
            
            return True
        else:
            logger.error(f"Failed to start MCP server: {server_name}")
            return False
    
    def _send_keep_alive_ping(self, server_name: str) -> None:
        """Send a keep-alive ping to maintain server connection."""
        assert_non_empty(server_name, "server_name", {"operation": "keep_alive_ping"})
        
        if not self.is_server_running(server_name):
            return
        
        process = self.servers[server_name]["process"]
        if process and process.stdin:
            ping_message = json.dumps({"type": "ping", "timestamp": time.time()}) + "\n"
            process.stdin.write(ping_message)
            process.stdin.flush()
            logger.debug(f"Sent keep-alive ping to server: {server_name}")
    
    def stop_server(self, server_name: str) -> bool:
        """Stop a specific MCP server with assertion validation."""
        assert_non_empty(server_name, "server_name", {"operation": "stop_server"})
        
        require(
            server_name in self.servers,
            f"Server '{server_name}' must be registered",
            context={"server_name": server_name, "registered_servers": list(self.servers.keys())}
        )
        
        if not self.is_server_running(server_name):
            logger.info(f"Server '{server_name}' is not running")
            return True
        
        logger.info(f"Stopping MCP server: {server_name}")
        
        process = self.servers[server_name]["process"]
        if process:
            process.terminate()
            
            # Wait for graceful shutdown
            for _ in range(5):
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            
            # Force kill if still running
            if process.poll() is None:
                process.kill()
                process.wait()
            
            # Clean up
            self.servers[server_name]["process"] = None
            if server_name in self.processes:
                del self.processes[server_name]
            
            logger.info(f"Successfully stopped MCP server: {server_name}")
            return True
        
        return False
    
    def is_server_running(self, server_name: str) -> bool:
        """Check if a specific MCP server is running with assertion validation."""
        assert_non_empty(server_name, "server_name", {"operation": "is_server_running"})
        
        if server_name not in self.servers:
            logger.info(f"DIAGNOSTIC: Server '{server_name}' not in servers dict")
            return False
        
        process = self.servers[server_name].get("process")
        if process is None:
            logger.info(f"DIAGNOSTIC: Server '{server_name}' has no process object")
            return False
        
        # Check if process is still alive
        poll_result = process.poll()
        is_running = poll_result is None
        
        if not is_running:
            logger.info(f"DIAGNOSTIC: Server '{server_name}' process terminated with exit code: {poll_result}")
            # Try to get stderr output if available with assertion validation
            try:
                if process.stderr:
                    stderr_output = process.stderr.read()
                    if stderr_output:
                        assert_type(stderr_output, (str, bytes), "stderr_output", {"server_name": server_name})
                        logger.info(f"DIAGNOSTIC: Server '{server_name}' stderr: {stderr_output}")
            except Exception as e:
                logger.debug(f"Could not read stderr for server '{server_name}': {e}")
        
        ensure(
            isinstance(is_running, bool),
            "Server running status must be boolean",
            context={"server_name": server_name, "poll_result": poll_result}
        )
        
        return is_running
    
    def get_running_servers(self) -> List[str]:
        """Get list of currently running servers with assertion validation."""
        running_servers = []
        for server_name in self.servers:
            is_running = self.is_server_running(server_name)
            logger.info(f"DIAGNOSTIC: Server '{server_name}' running status: {is_running}")
            if is_running:
                running_servers.append(server_name)
        
        logger.debug(f"Running servers: {running_servers}")
        return running_servers
    
    def is_connected(self) -> bool:
        """Check if any MCP servers are connected and running."""
        running_servers = self.get_running_servers()
        return len(running_servers) > 0
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools from running servers with assertion validation."""
        all_tools = []
        running_servers = self.get_running_servers()
        
        logger.info(f"DIAGNOSTIC: Looking for tools in {len(running_servers)} running servers: {running_servers}")
        
        for server_name in running_servers:
            logger.info(f"DIAGNOSTIC: Getting manifest from server '{server_name}'")
            server_tools = self.get_server_manifest(server_name)
            logger.info(f"DIAGNOSTIC: Server '{server_name}' returned {len(server_tools)} tools")
            for tool in server_tools:
                tool["server_name"] = server_name
                all_tools.append(tool)
        
        logger.info(f"Found {len(all_tools)} available tools from {len(running_servers)} servers")
        
        ensure(
            isinstance(all_tools, list),
            "Available tools must be a list",
            context={"tool_count": len(all_tools), "running_servers": running_servers}
        )
        
        return all_tools
    
    def get_server_manifest(self, server_name: str) -> List[Dict[str, Any]]:
        """Get tool manifest from a specific server with assertion validation and retry logic."""
        return self.get_server_manifest_with_retry(server_name, max_retries=3)
    
    def get_server_manifest_with_retry(self, server_name: str, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Get tool manifest from a specific server with retry logic and enhanced error handling."""
        assert_non_empty(server_name, "server_name", {"operation": "get_server_manifest_with_retry"})
        assert_in_range(max_retries, 1, 10, "max_retries", {"operation": "get_server_manifest_with_retry"})
        
        require(
            self.is_server_running(server_name),
            f"Server '{server_name}' must be running to get manifest",
            context={"server_name": server_name, "running_servers": self.get_running_servers()}
        )
        
        # Check cache first
        if server_name in self.tool_manifests:
            cached_manifest = self.tool_manifests[server_name]
            if cached_manifest and len(cached_manifest) > 0:
                return cached_manifest
        
        # Attempt to get manifest with retry logic
        for attempt in range(max_retries):
            try:
                logger.debug(f"Manifest request attempt {attempt + 1}/{max_retries} for server '{server_name}'")
                manifest = self._request_manifest_single_attempt(server_name)
                
                if manifest and len(manifest) > 0:
                    # Cache successful result
                    self.tool_manifests[server_name] = manifest
                    logger.info(f"Retrieved manifest for server '{server_name}': {len(manifest)} tools (attempt {attempt + 1})")
                    return manifest
                else:
                    logger.warning(f"Empty manifest received from server '{server_name}' on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Manifest request attempt {attempt + 1} failed for server '{server_name}': {e}")
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        logger.error(f"Failed to get manifest from server '{server_name}' after {max_retries} attempts")
        return []
    
    def _request_manifest_single_attempt(self, server_name: str) -> List[Dict[str, Any]]:
        """Make a single manifest request attempt to a server."""
        process = self.servers[server_name]["process"]
        
        require(
            process and process.stdin and process.stdout,
            f"Server '{server_name}' process not properly initialized",
            context={"server_name": server_name, "has_process": bool(process)}
        )
        
        # Send manifest request
        manifest_request = json.dumps({
            "type": "manifest",
            "timestamp": time.time(),
            "request_id": str(uuid.uuid4())
        }) + "\n"
        
        process.stdin.write(manifest_request)
        process.stdin.flush()
        
        # Wait for response with timeout
        ready_to_read, _, _ = select.select([process.stdout], [], [], 10.0)
        
        if not ready_to_read:
            raise NetworkError(f"Timeout waiting for manifest response from server '{server_name}'")
        
        response_line = process.stdout.readline().strip()
        if not response_line:
            raise ProcessingError(f"Empty response received from server '{server_name}'")
        
        try:
            manifest_data = json.loads(response_line)
        except json.JSONDecodeError as e:
            raise ProcessingError(f"Invalid JSON response from server '{server_name}': {e}")
        
        # Handle both direct tools array and manifest wrapper
        if "content" in manifest_data and "manifest" in manifest_data["content"]:
            tools = manifest_data["content"]["manifest"]
        else:
            tools = manifest_data.get("tools", manifest_data.get("manifest", []))
        
        ensure(
            isinstance(tools, list),
            f"Manifest from server '{server_name}' must be a list",
            context={"server_name": server_name, "manifest_type": type(tools).__name__}
        )
        
        return tools
    
    def verify_all_servers_ready(self) -> Dict[str, bool]:
        """Verify readiness status of all servers with comprehensive checks using assertion logic."""
        readiness_status = {}
        
        for server_name in self.servers:
            readiness_status[server_name] = self._verify_server_ready(server_name)
            logger.debug(f"Server '{server_name}' readiness: {readiness_status[server_name]}")
        
        ready_count = sum(1 for ready in readiness_status.values() if ready)
        total_count = len(readiness_status)
        logger.info(f"Server readiness check: {ready_count}/{total_count} servers ready")
        
        return readiness_status
    
    def _verify_server_ready(self, server_name: str) -> bool:
        """Verify if a specific server is ready and functional using assertion-based validation."""
        assert_non_empty(server_name, "server_name", {"operation": "verify_server_ready"})
        
        # Check 1: Server must exist in configuration
        if server_name not in self.servers:
            logger.warning(f"Server '{server_name}' does not exist in configuration")
            return False
        
        # Check 2: Process must be running
        if not self.is_server_running(server_name):
            logger.warning(f"Server '{server_name}' process is not running")
            return False
        
        # Check 3: Manifest must be available and non-empty
        manifest = self.get_server_manifest_with_retry(server_name, max_retries=2)
        if not manifest or len(manifest) == 0:
            logger.warning(f"Server '{server_name}' has empty or missing manifest")
            return False
        
        # Check 4: Validate expected tool count (if configured)
        expected_tool_count = self._get_expected_tool_count(server_name)
        if expected_tool_count > 0:
            # Use ensure for non-critical validation (logs warning but doesn't fail)
            ensure(
                len(manifest) == expected_tool_count,
                f"Server '{server_name}' tool count matches expected",
                context={
                    "server_name": server_name,
                    "actual_count": len(manifest),
                    "expected_count": expected_tool_count
                },
                
            )
        
        logger.debug(f"Server '{server_name}' passed all readiness checks")
        return True
    
    def _get_expected_tool_count(self, server_name: str) -> int:
        """Get expected tool count for a server (if configured)."""
        # Define expected tool counts for known servers
        expected_counts = {
            "web_search": 2,
            "multi_modal_processor": 3,
            "text_processor": 4,
            "academic_search": 3,
            "basic_tools": 2,
            "codex_agent": 4,
            "math_calculator": 5
        }
        
        # Handle server name variations (with/without _server suffix)
        base_name = server_name.replace("_server", "")
        return expected_counts.get(base_name, expected_counts.get(server_name, 0))
    
    def wait_for_servers_ready(self, timeout: int = 30, check_interval: int = 2) -> bool:
        """Wait for all servers to become ready within timeout period using assertion validation."""
        assert_in_range(timeout, 5, 300, "timeout", {"operation": "wait_for_servers_ready"})
        assert_in_range(check_interval, 1, 10, "check_interval", {"operation": "wait_for_servers_ready"})
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            readiness_status = self.verify_all_servers_ready()
            ready_servers = [name for name, ready in readiness_status.items() if ready]
            
            if require(
                len(ready_servers) == len(self.servers),
                "All servers must be ready",
                context={
                    "ready_count": len(ready_servers),
                    "total_count": len(self.servers),
                    "ready_servers": ready_servers
                },
                
            ):
                logger.info(f"All {len(self.servers)} servers are ready")
                return True
            
            logger.debug(f"Waiting for servers to be ready: {len(ready_servers)}/{len(self.servers)} ready")
            time.sleep(check_interval)
        
        # Final check and report
        final_status = self.verify_all_servers_ready()
        ready_servers = [name for name, ready in final_status.items() if ready]
        not_ready_servers = [name for name, ready in final_status.items() if not ready]
        
        logger.warning(f"Timeout waiting for servers. Ready: {ready_servers}, Not ready: {not_ready_servers}")
        return False
    
    def verify_critical_tools_available(self, critical_tools_config: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
        """Verify that critical tools are available using provided or config-driven discovery."""
        # Use provided config or get from settings
        if critical_tools_config is None:
            try:
                from config.config import auto_discover_critical_tools
                critical_tools_config = auto_discover_critical_tools()
            except Exception as e:
                logger.warning(f"Failed to auto-discover critical tools: {e}")
                critical_tools_config = {}
        
        assert_not_none(critical_tools_config, "critical_tools_config", {"operation": "verify_critical_tools"})
        require(
            isinstance(critical_tools_config, dict),
            "Critical tools configuration must be a dictionary",
            context={"config_type": type(critical_tools_config).__name__, "operation": "verify_critical_tools"}
        )
        
        missing_tools = {}
        available_tools = self.get_available_tools()
        
        # Create mapping of tool names to their server sources
        tool_to_server = {}
        for tool in available_tools:
            tool_name = tool.get("name", "")
            server_name = tool.get("server_name", "")
            if tool_name and server_name:
                tool_to_server[tool_name] = server_name
        
        available_tool_names = set(tool_to_server.keys())
        
        # Check each server's critical tools
        for server_name, expected_tools in critical_tools_config.items():
            assert_not_none(expected_tools, "expected_tools", {"server_name": server_name, "operation": "critical_tool_check"})
            
            missing = []
            for tool_name in expected_tools:
                if tool_name not in available_tool_names:
                    logger.warning(f"Critical tool '{tool_name}' is missing from server '{server_name}'")
                    missing.append(tool_name)
            
            if missing:
                missing_tools[server_name] = missing
        
        # Log results using unified assertion approach
        if missing_tools:
            logger.warning(f"Missing critical tools detected: {missing_tools}")
        else:
            logger.info("All critical tools are available")
        
        ensure(
            isinstance(missing_tools, dict),
            "Missing tools result must be dictionary",
            context={"result_type": type(missing_tools).__name__, "missing_count": len(missing_tools)}
        )
        
        return missing_tools
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report with config-driven validation."""
        health_report = {
            "timestamp": time.time(),
            "servers": {},
            "tools": {},
            "critical_tools": {},
            "overall_status": "unknown"
        }
        
        # Server health check
        readiness_status = self.verify_all_servers_ready()
        ready_servers = [name for name, ready in readiness_status.items() if ready]
        total_servers = len(self.servers)
        
        health_report["servers"] = {
            "total": total_servers,
            "ready": len(ready_servers),
            "ready_list": ready_servers,
            "not_ready_list": [name for name, ready in readiness_status.items() if not ready],
            "success_rate": (len(ready_servers) / max(total_servers, 1)) * 100
        }
        
        # Tool availability check
        available_tools = self.get_available_tools()
        tools_by_server = {}
        for tool in available_tools:
            server_name = tool.get("server_name", "unknown")
            if server_name not in tools_by_server:
                tools_by_server[server_name] = []
            tools_by_server[server_name].append(tool.get("name", "unnamed"))
        
        health_report["tools"] = {
            "total_available": len(available_tools),
            "by_server": tools_by_server,
            "server_count": len(tools_by_server)
        }
        
        # Critical tools check using config-driven approach
        missing_critical = self.verify_critical_tools_available()
        health_report["critical_tools"] = {
            "missing_by_server": missing_critical,
            "all_critical_available": len(missing_critical) == 0,
            "affected_servers": list(missing_critical.keys())
        }
        
        # Overall system status determination with assertions
        server_health_good = len(ready_servers) >= (total_servers * 0.8)  # 80% servers ready
        tools_available = len(available_tools) >= 15  # At least 15 tools
        critical_tools_ok = len(missing_critical) == 0
        
        if server_health_good and tools_available and critical_tools_ok:
            health_report["overall_status"] = "excellent"
        elif server_health_good and tools_available:
            health_report["overall_status"] = "good"
        elif len(ready_servers) > 0 and len(available_tools) > 0:
            health_report["overall_status"] = "degraded"
        else:
            health_report["overall_status"] = "critical"
        
        # Validate health report structure
        ensure(
            health_report["overall_status"] in ["excellent", "good", "degraded", "critical"],
            "Health report status must be valid",
            context={"status": health_report["overall_status"], "operation": "health_report_generation"}
        )
        
        logger.info(f"System health report: {health_report['overall_status']} - "
                   f"{len(ready_servers)}/{total_servers} servers, "
                   f"{len(available_tools)} tools, "
                   f"{len(missing_critical)} servers missing critical tools")
        
        return health_report
    
    def initialize_with_verification(self, wait_timeout: int = 30) -> Dict[str, Any]:
        """Enhanced initialization with comprehensive verification and config-driven health reporting."""
        assert_in_range(wait_timeout, 10, 300, "wait_timeout", {"operation": "initialize_with_verification"})
        
        require(
            self.enabled,
            "MCP servers must be enabled for initialization",
            context={"operation": "initialize_with_verification"}
        )
        
        logger.info("Starting enhanced MCP initialization with verification...")
        
        # Phase 1: Start all servers
        logger.info("Phase 1: Starting MCP servers...")
        start_results = self.start_all_servers()
        successful_starts = [name for name, success in start_results.items() if success]
        
        ensure(
            len(successful_starts) > 0,
            "At least one MCP server must start successfully",
            context={"successful_starts": successful_starts, "total_servers": len(self.servers)}
        )
        
        # Phase 2: Wait for server readiness
        logger.info(f"Phase 2: Waiting for server readiness (timeout: {wait_timeout}s)...")
        all_ready = self.wait_for_servers_ready(timeout=wait_timeout, check_interval=2)
        
        # Phase 3: Generate comprehensive health report
        logger.info("Phase 3: Generating system health report...")
        health_report = self.get_system_health_report()
        
        # Phase 4: Validation and reporting
        logger.info("Phase 4: Final validation and reporting...")
        
        initialization_result = {
            "success": health_report["overall_status"] in ["excellent", "good"],
            "health_report": health_report,
            "start_results": start_results,
            "all_servers_ready": all_ready,
            "initialization_time": time.time()
        }
        
        # Log final status with unified assertion validation
        status = initialization_result["health_report"]["overall_status"]
        servers_ready = initialization_result["health_report"]["servers"]["ready"]
        total_servers = initialization_result["health_report"]["servers"]["total"]
        tools_available = initialization_result["health_report"]["tools"]["total_available"]
        
        if initialization_result["success"]:
            logger.info(f"✅ MCP initialization completed successfully! "
                       f"Status: {status.upper()}, "
                       f"Servers: {servers_ready}/{total_servers}, "
                       f"Tools: {tools_available}")
        else:
            logger.warning(f"⚠️ MCP initialization completed with issues. "
                          f"Status: {status.upper()}, "
                          f"Servers: {servers_ready}/{total_servers}, "
                          f"Tools: {tools_available}")
        
        ensure(
            isinstance(initialization_result, dict) and "success" in initialization_result,
            "Initialization result must be valid dictionary",
            context={"result_keys": list(initialization_result.keys()), "operation": "initialization_completion"}
        )
        
        return initialization_result
    
    def initialize(self) -> None:
        """Initialize the MCP manager by starting all servers."""
        require(
            self.enabled,
            "MCP servers must be enabled for initialization",
            context={"operation": "initialize"}
        )
        
        # Start all servers
        results = self.start_all_servers()
        
        # Log results
        successful_starts = [name for name, success in results.items() if success]
        logger.info(f"MCP manager initialized successfully. {len(successful_starts)}/{len(self.servers)} servers started.")
        
        ensure(
            len(successful_starts) > 0,
            "At least one server must start successfully during initialization",
            context={"results": results}
        )
    
    # ENHANCED MONITORING AND DIAGNOSTICS - Using dedicated monitoring module
    
    def log_manifest_operation(self, operation: str, server_name: str, details: Dict[str, Any]) -> None:
        """Delegate to monitoring module for manifest operation logging."""
        log_manifest_operation(operation, server_name, details)
    
    def get_mcp_system_health(self) -> Dict[str, Any]:
        """Delegate to monitoring module for comprehensive system health check."""
        return get_mcp_system_health(self)
    
    def log_tool_usage(self, server_name: str, tool_name: str, arguments: Dict[str, Any],
                      result: Dict[str, Any], execution_time: float) -> None:
        """Delegate to monitoring module for tool usage logging."""
        log_tool_usage(server_name, tool_name, arguments, result, execution_time)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Delegate to monitoring module for performance metrics collection."""
        return get_performance_metrics(self)
    
    def cleanup(self) -> None:
        """Clean up all MCP servers."""
        logger.info("Cleaning up MCP servers...")
        
        # Stop all running servers
        results = {}
        for server_name in list(self.servers.keys()):
            if server_name in self.servers and self.servers[server_name].get("process"):
                results[server_name] = self.stop_server(server_name)
        
        # Check results
        failed_stops = [name for name, success in results.items() if not success]
        
        if failed_stops:
            logger.warning(f"Failed to stop servers: {failed_stops}")
        else:
            logger.info("All MCP servers stopped successfully.")
    
    def use_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use a specific tool from an MCP server with assertion validation and monitoring.
        """
        assert_non_empty(server_name, "server_name", {"operation": "use_tool"})
        assert_non_empty(tool_name, "tool_name", {"operation": "use_tool"})
        assert_not_none(arguments, "arguments", {"operation": "use_tool"})
        
        require(
            self.enabled,
            "MCP servers must be enabled to use tools",
            context={"operation": "use_tool", "server_name": server_name, "tool_name": tool_name}
        )
        
        running_servers = self.get_running_servers()
        require(
            server_name in running_servers,
            f"Server '{server_name}' must be running to use tools",
            context={"server_name": server_name, "running_servers": running_servers}
        )
        
        # Get the process for this server
        process = self.servers[server_name]["process"]
        assert_not_none(process, "process", {"server_name": server_name, "operation": "use_tool"})
        
        # Start timing for performance monitoring
        execution_start_time = time.time()
        
        # Prepare the tool execution request
        request = {
            "id": str(uuid.uuid4()),
            "type": "tool",
            "tool": tool_name,
            "arguments": sanitize_for_json(arguments)
        }
        
        try:
            # Clear any pending output from the server before sending new request
            ready_to_read, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready_to_read:
                # Read and discard any pending output
                pending_output = process.stdout.readline().strip()
                if pending_output:
                    logger.warning(f"DIAGNOSTIC: Discarded pending output from {server_name}: {pending_output}")
            
            # Send request to server
            json_request = json.dumps(request) + "\n"
            logger.info(f"DIAGNOSTIC: Sending request to {server_name}: {json_request.strip()}")
            process.stdin.write(json_request)
            process.stdin.flush()
            
            # Wait for response
            ready_to_read, _, _ = select.select([process.stdout], [], [], 30.0)
            
            require(
                ready_to_read,
                f"Tool execution response must be received within timeout",
                context={"server_name": server_name, "tool_name": tool_name}
            )
            
            response_line = process.stdout.readline().strip()
            logger.info(f"DIAGNOSTIC: Received response from {server_name}: {response_line}")
            response_data = json.loads(response_line)
            
            ensure(
                isinstance(response_data, dict),
                "Tool execution response must be a dictionary",
                context={"server_name": server_name, "tool_name": tool_name, "response_type": type(response_data).__name__}
            )
            
            # Calculate execution time and log usage
            execution_time = time.time() - execution_start_time
            self.log_tool_usage(server_name, tool_name, arguments, response_data, execution_time)
            
            return response_data
            
        except Exception as e:
            # : Log failed tool usage
            execution_time = time.time() - execution_start_time
            error_result = {"error": str(e), "success": False}
            self.log_tool_usage(server_name, tool_name, arguments, error_result, execution_time)
            raise


def get_mcp_manager() -> MCPServerManager:
    """Get the singleton MCP manager instance with assertion validation."""
    global _mcp_manager_instance
    
    if _mcp_manager_instance is None:
        logger.info("Creating new MCPServerManager instance")
        _mcp_manager_instance = MCPServerManager()
    
    assert_not_none(_mcp_manager_instance, "_mcp_manager_instance", {"operation": "get_mcp_manager"})
    return _mcp_manager_instance


# LangChain tool integration functions
def get_mcp_tools_as_langchain() -> List:
    """
    Get all available MCP tools as LangChain-compatible tool definitions with assertion validation.
    """
    mcp_manager = get_mcp_manager()
    available_tools = mcp_manager.get_available_tools()
    
    langchain_tools = []
    for tool_def in available_tools:
        server_name = tool_def.get("server_name", "unknown")
        tool_name = tool_def.get("name", "unknown")
        
        assert_non_empty(server_name, "server_name", {"tool_def": tool_def})
        assert_non_empty(tool_name, "tool_name", {"tool_def": tool_def})
        
        # Create a LangChain tool using the @tool decorator
        @tool
        def mcp_tool_wrapper(tool_args: str, server=server_name, tool=tool_name) -> str:
            """Dynamically created MCP tool wrapper with assertion validation."""
            assert_non_empty(tool_args, "tool_args", {"server": server, "tool": tool})
            
            # Parse arguments
            arguments = json.loads(tool_args)
            assert_type(arguments, dict, "arguments", {"server": server, "tool": tool})
            
            # Use the MCP tool
            result = use_mcp_tool(server, tool, arguments)
            
            ensure(
                isinstance(result, dict),
                "MCP tool result must be a dictionary",
                context={"server": server, "tool": tool, "result_type": type(result).__name__}
            )
            
            return json.dumps(result)
        
        # Set the tool name and description
        mcp_tool_wrapper.name = f"{server_name}_{tool_name}"
        mcp_tool_wrapper.description = tool_def.get("description", f"MCP tool {tool_name} from {server_name}")
        
        langchain_tools.append(mcp_tool_wrapper)
    
    logger.info(f"Created {len(langchain_tools)} LangChain-compatible MCP tools")
    
    ensure(
        isinstance(langchain_tools, list),
        "LangChain tools must be a list",
        context={"tool_count": len(langchain_tools)}
    )
    
    return langchain_tools


def use_mcp_tool(server_name: str, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use an MCP tool with comprehensive assertion validation and error handling.
    """
    assert_non_empty(server_name, "server_name", {"operation": "use_mcp_tool"})
    assert_non_empty(tool_name, "tool_name", {"operation": "use_mcp_tool"})
    assert_not_none(tool_args, "tool_args", {"operation": "use_mcp_tool"})
    
    logger.info(f"Calling MCP tool '{server_name}.{tool_name}' with args: {tool_args}")
    
    # Sanitize arguments before passing to use_mcp_tool
    sanitized_args = sanitize_for_json(tool_args)
    
    # Get MCP manager and use the tool
    mcp_manager = get_mcp_manager()
    result = mcp_manager.use_tool(server_name, tool_name, sanitized_args)
    
    # Validate and return result
    assert_mcp_server_response(result, server_name, tool_name, {"arguments": sanitized_args})
    
    logger.info(f"MCP tool '{server_name}.{tool_name}' completed successfully")
    return result
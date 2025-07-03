"""
MCP System Monitoring and Diagnostics Module

This module provides Phase 4 enhanced monitoring capabilities for the MCP system,
including comprehensive health checks, performance metrics, and intelligent diagnostics.
"""

import time
import logging
from typing import Dict, List, Any, Optional

from tools.assertions import (
    require, ensure, assert_not_none, assert_type, assert_non_empty
)
from tools.langfuse_monitor import LangfuseMonitor

logger = logging.getLogger(__name__)


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


def log_manifest_operation(operation: str, server_name: str, details: Dict[str, Any]) -> None:
    """
    Comprehensive logging system for manifest operations with Langfuse integration.
    Phase 4 enhancement for detailed monitoring and diagnostics.
    """
    assert_non_empty(operation, "operation", {"context": "log_manifest_operation"})
    assert_non_empty(server_name, "server_name", {"context": "log_manifest_operation"})
    assert_not_none(details, "details", {"context": "log_manifest_operation"})
    
    # Enhanced logging with structured data
    log_data = {
        "server_name": server_name,
        "operation": operation,
        "details": sanitize_for_json(details),
        "timestamp": time.time()
    }
    
    logger.info(f"MCP_MANIFEST_{operation.upper()}: {server_name}", extra=log_data)
    
    # Langfuse monitoring integration
    try:
        monitor = LangfuseMonitor()
        if monitor.is_enabled():
            # Create a span for the manifest operation
            with monitor.trace_span(
                name=f"mcp_manifest_{operation}",
                metadata={
                    "server_name": server_name,
                    "operation": operation,
                    "component": "mcp_client"
                }
            ) as span:
                span.update(
                    input={"server_name": server_name, "operation": operation},
                    output=details,
                    metadata=log_data
                )
    except Exception as e:
        logger.warning(f"Failed to log manifest operation to Langfuse: {str(e)}")


def log_tool_usage(server_name: str, tool_name: str, arguments: Dict[str, Any],
                  result: Dict[str, Any], execution_time: float) -> None:
    """
    Enhanced tool usage logging with Langfuse integration.
    Phase 4 enhancement for detailed tool usage monitoring.
    """
    assert_non_empty(server_name, "server_name", {"context": "log_tool_usage"})
    assert_non_empty(tool_name, "tool_name", {"context": "log_tool_usage"})
    assert_not_none(arguments, "arguments", {"context": "log_tool_usage"})
    assert_not_none(result, "result", {"context": "log_tool_usage"})
    
    # Enhanced logging
    usage_data = {
        "server_name": server_name,
        "tool_name": tool_name,
        "execution_time": execution_time,
        "arguments_size": len(str(arguments)),
        "result_size": len(str(result)),
        "timestamp": time.time(),
        "success": "error" not in result
    }
    
    logger.info(f"MCP_TOOL_USAGE: {server_name}.{tool_name} "
               f"({execution_time:.3f}s) - {'✅ SUCCESS' if usage_data['success'] else '❌ ERROR'}")
    
    # Langfuse monitoring
    try:
        monitor = LangfuseMonitor()
        if monitor.is_enabled():
            with monitor.trace_span(
                name=f"mcp_tool_{tool_name}",
                metadata={
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "component": "mcp_client"
                }
            ) as span:
                span.update(
                    input={"tool": tool_name, "arguments": sanitize_for_json(arguments)},
                    output=sanitize_for_json(result),
                    metadata=usage_data
                )
    except Exception as e:
        logger.warning(f"Failed to log tool usage to Langfuse: {str(e)}")


def get_mcp_system_health(mcp_manager) -> Dict[str, Any]:
    """
    Health check endpoint providing comprehensive MCP system status.
    Phase 4 enhancement for system monitoring and diagnostics.
    """
    assert_not_none(mcp_manager, "mcp_manager", {"operation": "get_mcp_system_health"})
    
    logger.info("Generating comprehensive MCP system health report...")
    
    # Start Langfuse monitoring for health check
    monitor = LangfuseMonitor()
    health_trace = None
    
    try:
        if monitor.is_enabled():
            health_trace = monitor.start_trace(
                name="mcp_system_health_check",
                metadata={"component": "mcp_client", "operation": "health_check"}
            )
        
        # Get basic system metrics
        running_servers = mcp_manager.get_running_servers()
        total_servers = len(mcp_manager.servers)
        available_tools = mcp_manager.get_available_tools()
        
        # Server status details
        server_details = {}
        for server_name in mcp_manager.servers:
            is_running = mcp_manager.is_server_running(server_name)
            server_details[server_name] = {
                "running": is_running,
                "process_id": mcp_manager.servers[server_name].get("process", {}).get("pid") if is_running else None,
                "tools_count": 0
            }
            
            # Get tool count for running servers
            if is_running:
                try:
                    manifest = mcp_manager.get_server_manifest_with_retry(server_name, max_retries=2)
                    server_details[server_name]["tools_count"] = len(manifest)
                    
                    # Log manifest operation
                    log_manifest_operation(
                        "health_check_manifest",
                        server_name,
                        {"tools_found": len(manifest), "success": True}
                    )
                except Exception as e:
                    logger.warning(f"Failed to get manifest for {server_name} during health check: {str(e)}")
                    log_manifest_operation(
                        "health_check_manifest_failed",
                        server_name,
                        {"error": str(e), "success": False}
                    )
        
        # Critical tools verification
        missing_critical = mcp_manager.verify_critical_tools_available()
        
        # Performance metrics
        health_data = {
            "timestamp": time.time(),
            "servers": {
                "total": total_servers,
                "running": len(running_servers),
                "running_list": running_servers,
                "failed_list": [name for name in mcp_manager.servers if name not in running_servers],
                "success_rate": (len(running_servers) / max(total_servers, 1)) * 100,
                "details": server_details
            },
            "tools": {
                "total_available": len(available_tools),
                "by_server": {},
                "server_count": len([name for name in running_servers if server_details[name]["tools_count"] > 0])
            },
            "critical_tools": {
                "missing_by_server": missing_critical,
                "all_critical_available": len(missing_critical) == 0,
                "affected_servers": list(missing_critical.keys())
            },
            "system_status": "unknown"
        }
        
        # Group tools by server
        for tool in available_tools:
            server_name = tool.get("server_name", "unknown")
            if server_name not in health_data["tools"]["by_server"]:
                health_data["tools"]["by_server"][server_name] = []
            health_data["tools"]["by_server"][server_name].append(tool.get("name", "unnamed"))
        
        # Determine overall system status
        server_health_excellent = len(running_servers) == total_servers
        server_health_good = len(running_servers) >= (total_servers * 0.8)
        tools_excellent = len(available_tools) >= 20
        tools_good = len(available_tools) >= 15
        critical_tools_ok = len(missing_critical) == 0
        
        if server_health_excellent and tools_excellent and critical_tools_ok:
            health_data["system_status"] = "excellent"
        elif server_health_good and tools_good and critical_tools_ok:
            health_data["system_status"] = "good"
        elif server_health_good and len(available_tools) >= 10:
            health_data["system_status"] = "degraded"
        elif len(running_servers) > 0 and len(available_tools) > 0:
            health_data["system_status"] = "poor"
        else:
            health_data["system_status"] = "critical"
        
        # Log comprehensive health status
        logger.info(f"MCP System Health: {health_data['system_status'].upper()} - "
                   f"{len(running_servers)}/{total_servers} servers running, "
                   f"{len(available_tools)} tools available, "
                   f"critical tools: {'✅ OK' if critical_tools_ok else '❌ MISSING'}")
        
        # Update Langfuse trace
        if health_trace:
            health_trace.update(
                output=health_data,
                metadata={
                    "system_status": health_data["system_status"],
                    "servers_running": len(running_servers),
                    "tools_available": len(available_tools)
                }
            )
            health_trace.end()
        
        return health_data
        
    except Exception as e:
        error_msg = f"Failed to generate MCP system health report: {str(e)}"
        logger.error(error_msg)
        
        if health_trace:
            health_trace.update(
                output={"error": error_msg},
                metadata={"error": True}
            )
            health_trace.end()
        
        # Return minimal health data on error
        return {
            "timestamp": time.time(),
            "system_status": "error",
            "error": error_msg,
            "servers": {"total": len(mcp_manager.servers), "running": 0},
            "tools": {"total_available": 0}
        }


def get_performance_metrics(mcp_manager) -> Dict[str, Any]:
    """
    Get detailed performance metrics for MCP system.
    Phase 4 enhancement for performance monitoring.
    """
    assert_not_none(mcp_manager, "mcp_manager", {"operation": "get_performance_metrics"})
    
    logger.info("Collecting MCP system performance metrics...")
    
    try:
        # Collect timing metrics for each server
        server_metrics = {}
        for server_name in mcp_manager.servers:
            start_time = time.time()
            
            if mcp_manager.is_server_running(server_name):
                try:
                    # Test manifest retrieval time
                    manifest_start = time.time()
                    manifest = mcp_manager.get_server_manifest_with_retry(server_name, max_retries=1)
                    manifest_time = time.time() - manifest_start
                    
                    server_metrics[server_name] = {
                        "status": "running",
                        "manifest_response_time": manifest_time,
                        "tools_count": len(manifest),
                        "health_check_time": time.time() - start_time
                    }
                except Exception as e:
                    server_metrics[server_name] = {
                        "status": "error",
                        "error": str(e),
                        "health_check_time": time.time() - start_time
                    }
            else:
                server_metrics[server_name] = {
                    "status": "stopped",
                    "health_check_time": time.time() - start_time
                }
        
        # Calculate aggregate metrics
        running_servers = [name for name, metrics in server_metrics.items()
                         if metrics["status"] == "running"]
        
        avg_manifest_time = 0
        if running_servers:
            manifest_times = [server_metrics[name].get("manifest_response_time", 0)
                            for name in running_servers
                            if "manifest_response_time" in server_metrics[name]]
            avg_manifest_time = sum(manifest_times) / len(manifest_times) if manifest_times else 0
        
        performance_data = {
            "timestamp": time.time(),
            "server_metrics": server_metrics,
            "aggregate_metrics": {
                "total_servers": len(mcp_manager.servers),
                "running_servers": len(running_servers),
                "average_manifest_response_time": avg_manifest_time,
                "system_health_score": (len(running_servers) / len(mcp_manager.servers)) * 100
            }
        }
        
        logger.info(f"Performance metrics collected: {len(running_servers)}/{len(mcp_manager.servers)} servers, "
                   f"avg manifest time: {avg_manifest_time:.3f}s")
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Failed to collect performance metrics: {str(e)}")
        return {
            "timestamp": time.time(),
            "error": str(e),
            "server_metrics": {},
            "aggregate_metrics": {"system_health_score": 0}
        }


def generate_mcp_recommendations(health_report: Dict[str, Any], performance_metrics: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on MCP system status with assertion validation."""
    assert_not_none(health_report, "health_report", {"operation": "generate_recommendations"})
    assert_not_none(performance_metrics, "performance_metrics", {"operation": "generate_recommendations"})
    assert_type(health_report, dict, "health_report", {"operation": "generate_recommendations"})
    assert_type(performance_metrics, dict, "performance_metrics", {"operation": "generate_recommendations"})
    
    recommendations = []
    
    # Server health recommendations with assertion validation
    if health_report["servers"]["success_rate"] < 100:
        failed_servers = health_report["servers"]["failed_list"]
        assert_type(failed_servers, list, "failed_servers", {"operation": "generate_recommendations"})
        if failed_servers:
            recommendations.append(f"Restart failed servers: {', '.join(failed_servers)}")
    
    # Tool availability recommendations
    tools_available = health_report["tools"]["total_available"]
    assert_type(tools_available, int, "tools_available", {"operation": "generate_recommendations"})
    if tools_available < 15:
        recommendations.append("Tool count is below optimal threshold - check server configurations")
    
    # Critical tools recommendations
    if not health_report["critical_tools"]["all_critical_available"]:
        affected_servers = health_report["critical_tools"]["affected_servers"]
        assert_type(affected_servers, list, "affected_servers", {"operation": "generate_recommendations"})
        if affected_servers:
            recommendations.append(f"Address missing critical tools in servers: {', '.join(affected_servers)}")
    
    # Performance recommendations with assertion validation
    avg_response_time = performance_metrics["aggregate_metrics"]["average_manifest_response_time"]
    assert_type(avg_response_time, (int, float), "avg_response_time", {"operation": "generate_recommendations"})
    if avg_response_time > 1.0:
        recommendations.append(f"Manifest response time ({avg_response_time:.3f}s) is high - consider server optimization")
    
    # Overall system recommendations
    overall_status = health_report["system_status"]
    assert_type(overall_status, str, "overall_status", {"operation": "generate_recommendations"})
    if overall_status == "critical":
        recommendations.append("URGENT: System requires immediate attention - multiple components failing")
    elif overall_status == "degraded":
        recommendations.append("System performance is degraded - schedule maintenance")
    
    result = recommendations if recommendations else ["System is operating optimally - no actions required"]
    
    # Validate result with unified assertions
    assert_not_none(result, "result", {"operation": "generate_recommendations"})
    assert_type(result, list, "result", {"operation": "generate_recommendations"})
    ensure(
        len(result) > 0,
        "Recommendations list must not be empty",
        context={"recommendations_count": len(result)}
    )
    
    return result
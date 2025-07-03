"""
LangFuse Monitoring Integration for GAIA Agentic System

This module provides LangFuse integration for monitoring and observability
of the GAIA agent's performance, including:
- Session-level traces for complete question processing
- Span-level tracking for each reasoning stage
- Tool usage tracking for MCP server calls
- Model performance metrics per stage
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from functools import wraps
from contextlib import contextmanager

from config.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# LangFuse client (lazy initialization)
_langfuse_client = None

def get_langfuse_client():
    """Get or create LangFuse client instance."""
    global _langfuse_client
    
    if _langfuse_client is None:
        try:
            from langfuse import Langfuse
            
            if not settings.langfuse_public_key or not settings.langfuse_secret_key:
                logger.warning("LangFuse keys not configured. Monitoring disabled.")
                return None
                
            _langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host
            )
            logger.info("LangFuse client initialized successfully")
            
        except ImportError:
            logger.warning("LangFuse not installed. Monitoring disabled.")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize LangFuse client: {str(e)}")
            return None
    
    return _langfuse_client


def get_current_tracer():
    """Get the current LangFuse tracer if available."""
    # Return the LangFuse client directly for event logging
    return get_langfuse_client()


def set_current_tracer(tracer):
    """Set the current LangFuse tracer."""
    client = get_langfuse_client()
    if client:
        client._current_trace = tracer

class GaiaTracer:
    """Main tracer class for GAIA agent monitoring."""
    
    def __init__(self, session_id: str, question: str):
        """Initialize tracer for a GAIA session."""
        self.session_id = session_id
        self.question = question
        self.client = get_langfuse_client()
        self.trace = None
        self.spans = {}
        self.start_time = time.time()
        
        if self.client:
            self.trace = self.client.trace(
                id=session_id,
                name="GAIA Question Processing",
                input={"question": question},
                metadata={
                    "session_id": session_id,
                    "system": "GAIA Agentic System",
                    "version": "2.0"
                }
            )
            logger.info(f"Started LangFuse trace for session {session_id}")
    
    def start_span(self, name: str, input_data: Dict[str, Any] = None) -> str:
        """Start a new span for tracking a specific operation."""
        if not self.client or not self.trace:
            return ""
            
        span_id = str(uuid.uuid4())
        
        try:
            span = self.trace.span(
                id=span_id,
                name=name,
                input=input_data or {},
                start_time=time.time()
            )
            self.spans[span_id] = {
                "span": span,
                "name": name,
                "start_time": time.time()
            }
            logger.debug(f"Started span '{name}' with ID {span_id}")
            
        except Exception as e:
            logger.error(f"Failed to start span '{name}': {str(e)}")
            
        return span_id
    
    def end_span(self, span_id: str, output_data: Dict[str, Any] = None, 
                 metadata: Dict[str, Any] = None, error: str = None):
        """End a span with output data and metadata."""
        if not span_id or span_id not in self.spans:
            return
            
        span_info = self.spans[span_id]
        span = span_info["span"]
        
        try:
            end_time = time.time()
            duration = end_time - span_info["start_time"]
            
            update_data = {
                "end_time": end_time,
                "output": output_data or {},
                "metadata": {
                    "duration_seconds": duration,
                    **(metadata or {})
                }
            }
            
            if error:
                update_data["level"] = "ERROR"
                update_data["status_message"] = error
            
            span.update(**update_data)
            
            logger.debug(f"Ended span '{span_info['name']}' (duration: {duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"Failed to end span '{span_info['name']}': {str(e)}")
        
        finally:
            del self.spans[span_id]
    
    def log_tool_usage(self, tool_name: str, server_name: str, 
                      arguments: Dict[str, Any], result: Dict[str, Any],
                      duration: float, success: bool):
        """Log MCP tool usage."""
        if not self.client or not self.trace:
            return
            
        try:
            generation = self.trace.generation(
                name=f"MCP Tool: {tool_name}",
                model=f"{server_name}/{tool_name}",
                input=arguments,
                output=result,
                metadata={
                    "tool_type": "mcp_server",
                    "server_name": server_name,
                    "duration_seconds": duration,
                    "success": success
                },
                usage={
                    "input": len(str(arguments)),
                    "output": len(str(result))
                }
            )
            
            logger.debug(f"Logged tool usage: {server_name}.{tool_name}")
            
        except Exception as e:
            logger.error(f"Failed to log tool usage: {str(e)}")
    
    def log_llm_call(self, model_name: str, prompt: str, response: str,
                     token_usage: Dict[str, int] = None, duration: float = 0):
        """Log LLM generation call."""
        if not self.client or not self.trace:
            return
            
        try:
            generation = self.trace.generation(
                name=f"LLM Generation: {model_name}",
                model=model_name,
                input=prompt,
                output=response,
                metadata={
                    "duration_seconds": duration
                },
                usage=token_usage or {}
            )
            
            logger.debug(f"Logged LLM call: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log LLM call: {str(e)}")
    
    def finalize_trace(self, final_answer: str, success: bool = True, 
                      error: str = None, metadata: Dict[str, Any] = None):
        """Finalize the trace with final results."""
        if not self.client or not self.trace:
            return
            
        try:
            end_time = time.time()
            total_duration = end_time - self.start_time
            
            update_data = {
                "output": {"answer": final_answer},
                "metadata": {
                    "total_duration_seconds": total_duration,
                    "success": success,
                    **(metadata or {})
                }
            }
            
            if error:
                update_data["level"] = "ERROR"
                update_data["status_message"] = error
            
            self.trace.update(**update_data)
            
            logger.info(f"Finalized trace for session {self.session_id} "
                       f"(duration: {total_duration:.2f}s, success: {success})")
            
        except Exception as e:
            logger.error(f"Failed to finalize trace: {str(e)}")

# Context manager for easy span tracking
@contextmanager
def trace_span(tracer: GaiaTracer, name: str, input_data: Dict[str, Any] = None):
    """Context manager for automatic span tracking."""
    span_id = tracer.start_span(name, input_data)
    start_time = time.time()
    error = None
    output_data = {}
    
    try:
        yield {"span_id": span_id, "set_output": lambda data: output_data.update(data)}
    except Exception as e:
        error = str(e)
        raise
    finally:
        duration = time.time() - start_time
        metadata = {"duration_seconds": duration}
        tracer.end_span(span_id, output_data, metadata, error)

# Decorator for automatic function tracing
def trace_function(name: str = None):
    """Decorator to automatically trace function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to find tracer in kwargs or args
            tracer = kwargs.get('tracer') or (args[0] if args and hasattr(args[0], 'tracer') else None)
            
            if not tracer or not isinstance(tracer, GaiaTracer):
                # No tracer available, just call function normally
                return func(*args, **kwargs)
            
            function_name = name or f"{func.__module__}.{func.__name__}"
            input_data = {
                "function": function_name,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
            
            with trace_span(tracer, function_name, input_data) as span:
                try:
                    result = func(*args, **kwargs)
                    span["set_output"]({"result_type": type(result).__name__})
                    return result
                except Exception as e:
                    span["set_output"]({"error": str(e)})
                    raise
        
        return wrapper
    return decorator

# Utility function to create tracer from session info
def create_tracer(session_id: str, question: str) -> GaiaTracer:
    """Create a new GaiaTracer instance."""
    return GaiaTracer(session_id, question)

# Health check function
def check_langfuse_health() -> Dict[str, Any]:
    """Check LangFuse connection health."""
    client = get_langfuse_client()
    
    if not client:
        return {
            "status": "disabled",
            "message": "LangFuse client not configured or unavailable"
        }
    
    try:
        # Try to create a simple trace to test connection
        test_trace = client.trace(
            name="Health Check",
            input={"test": True}
        )
        test_trace.update(output={"status": "ok"})
        
        return {
            "status": "healthy",
            "message": "LangFuse connection successful"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"LangFuse connection failed: {str(e)}"
        }


def log_assertion_failure(failure_data: Dict[str, Any]) -> None:
    """Log assertion failure to LangFuse."""
    client = get_langfuse_client()
    
    if not client:
        logger.debug("LangFuse client not available, skipping assertion failure logging")
        return
    
    try:
        # Create an event for the assertion failure
        event = client.event(
            name="assertion_failure",
            input=failure_data,
            metadata={
                "event_type": "assertion_failure",
                "severity": failure_data.get("severity", "unknown"),
                "category": failure_data.get("category", "unknown")
            }
        )
        
        logger.debug("Logged assertion failure to LangFuse")
        
    except Exception as e:
        logger.warning(f"Failed to log assertion failure to LangFuse: {str(e)}")




class LangfuseMonitor:
    """Legacy compatibility class for LangfuseMonitor."""
    
    def __init__(self):
        """Initialize the LangfuseMonitor."""
        self.client = get_langfuse_client()
    
    def is_enabled(self) -> bool:
        """Check if LangFuse monitoring is enabled."""
        return self.client is not None
    
    def create_tracer(self, session_id: str = None, question: str = None) -> Optional[GaiaTracer]:
        """Create a tracer instance."""
        if not session_id:
            session_id = str(uuid.uuid4())
        if not question:
            question = "default_question"
        return GaiaTracer(session_id, question)
    
    def log_event(self, name: str, data: Dict[str, Any]):
        """Log an event to LangFuse."""
        if not self.client:
            return
        
        try:
            self.client.event(
                name=name,
                input=data,
                metadata={"event_type": name}
            )
        except Exception as e:
            logger.warning(f"Failed to log event to LangFuse: {str(e)}")
    
    def trace_span(self, name: str, input_data: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """Create a trace span context manager for compatibility."""
        return LangfuseSpanContext(self, name, input_data, metadata)


class LangfuseSpanContext:
    """Context manager for LangFuse spans."""
    
    def __init__(self, monitor: 'LangfuseMonitor', name: str, input_data: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        self.monitor = monitor
        self.name = name
        self.input_data = input_data or {}
        self.metadata = metadata or {}
        self.span = None
    
    def __enter__(self):
        if not self.monitor.client:
            return self
        
        try:
            # Create a simple trace if none exists
            if not hasattr(self.monitor, '_current_trace') or not self.monitor._current_trace:
                session_id = str(uuid.uuid4())
                self.monitor._current_trace = self.monitor.client.trace(
                    id=session_id,
                    name="MCP Operation",
                    input=self.input_data,
                    metadata=self.metadata
                )
            
            # Create a span within the trace
            span_id = str(uuid.uuid4())
            self.span = self.monitor._current_trace.span(
                id=span_id,
                name=self.name,
                input=self.input_data,
                metadata=self.metadata
            )
            return self.span
            
        except Exception as e:
            logger.warning(f"Failed to create trace span: {str(e)}")
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context manager cleanup - span is automatically handled by LangFuse
        pass
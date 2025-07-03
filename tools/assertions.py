"""
Zero-Space Programming Assertions for GAIA Agentic System

This module implements assertion-based error handling with LangFuse integration
for better debugging, monitoring, and fail-fast behavior.
"""

import functools
import inspect
import traceback
from typing import Any, Callable, Dict, Optional, Union, List
from enum import Enum


class AssertionLevel(Enum):
    """Assertion severity levels for different types of checks."""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DATA_INTEGRITY = "data_integrity"


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class NetworkError(Exception):
    """Exception raised for network-related errors."""
    pass


class FileSystemError(Exception):
    """Exception raised for file system-related errors."""
    pass


class LLMError(Exception):
    """Exception raised for LLM-related errors."""
    pass


class ProcessingError(Exception):
    """Exception raised for processing-related errors."""
    pass


class GaiaAssertionError(AssertionError):
    """Enhanced assertion error with context and LangFuse integration."""
    
    def __init__(
        self,
        message: str,
        assertion_level: AssertionLevel,
        context: Optional[Dict[str, Any]] = None,
        function_name: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None
    ):
        super().__init__(message)
        self.message = message
        self.assertion_level = assertion_level
        self.context = context or {}
        self.function_name = function_name
        self.expected = expected
        self.actual = actual
        self.traceback_str = traceback.format_exc()
    
    def to_langfuse_event(self) -> Dict[str, Any]:
        """Convert assertion failure to LangFuse event."""
        return {
            "name": f"assertion_failure_{self.assertion_level.value}",
            "level": "ERROR",
            "metadata": {
                "assertion_type": self.assertion_level.value,
                "function": self.function_name,
                "expected": str(self.expected) if self.expected is not None else None,
                "actual": str(self.actual) if self.actual is not None else None
            },
            "input": self.context,
            "output": {"error_message": self.message}
        }


def gaia_assert(
    condition: bool,
    message: str,
    assertion_level: AssertionLevel = AssertionLevel.INVARIANT,
    context: Optional[Dict[str, Any]] = None,
    expected: Optional[Any] = None,
    actual: Optional[Any] = None,
    log_to_langfuse: bool = True
) -> None:
    """
    Enhanced assertion with LangFuse integration.
    
    Args:
        condition: Boolean condition to check
        message: Error message if assertion fails
        assertion_level: Type of assertion for categorization
        context: Additional context for debugging
        expected: Expected value (for comparison assertions)
        actual: Actual value (for comparison assertions)
        log_to_langfuse: Whether to log failure to LangFuse
    
    Raises:
        GaiaAssertionError: If condition is False
    """
    if not condition:
        # Get caller function name
        frame = inspect.currentframe().f_back
        function_name = frame.f_code.co_name if frame else "unknown"
        
        error = GaiaAssertionError(
            message=message,
            assertion_level=assertion_level,
            context=context,
            function_name=function_name,
            expected=expected,
            actual=actual
        )
        
        if log_to_langfuse:
            _log_assertion_to_langfuse(error)
        
        raise error


def require(
    condition: bool,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Precondition assertion - checks input requirements.
    
    Args:
        condition: Boolean condition to check
        message: Error message if assertion fails
        context: Additional context for debugging
        **kwargs: Additional arguments for gaia_assert
    """
    gaia_assert(
        condition,
        f"Precondition failed: {message}",
        AssertionLevel.PRECONDITION,
        context,
        **kwargs
    )


def ensure(
    condition: bool,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Postcondition assertion - checks output guarantees.
    
    Args:
        condition: Boolean condition to check
        message: Error message if assertion fails
        context: Additional context for debugging
        **kwargs: Additional arguments for gaia_assert
    """
    gaia_assert(
        condition,
        f"Postcondition failed: {message}",
        AssertionLevel.POSTCONDITION,
        context,
        **kwargs
    )


def invariant(
    condition: bool,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Invariant assertion - checks conditions that should always hold.
    
    Args:
        condition: Boolean condition to check
        message: Error message if assertion fails
        context: Additional context for debugging
        **kwargs: Additional arguments for gaia_assert
    """
    gaia_assert(
        condition,
        f"Invariant violated: {message}",
        AssertionLevel.INVARIANT,
        context,
        **kwargs
    )


def assert_not_none(
    value: Any,
    name: str = "value",
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Assert that a value is not None."""
    require(
        value is not None,
        f"{name} must not be None",
        context=context,
        expected="not None",
        actual=value
    )


def assert_type(
    value: Any,
    expected_type: Union[type, tuple],
    name: str = "value",
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Assert that a value is of expected type."""
    require(
        isinstance(value, expected_type),
        f"{name} must be of type {expected_type}, got {type(value)}",
        context=context,
        expected=expected_type,
        actual=type(value)
    )


def assert_in_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    name: str = "value",
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Assert that a numeric value is within range."""
    require(
        min_val <= value <= max_val,
        f"{name} must be between {min_val} and {max_val}, got {value}",
        context=context,
        expected=f"[{min_val}, {max_val}]",
        actual=value
    )


def assert_non_empty(
    collection: Union[str, list, dict, set],
    name: str = "collection",
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Assert that a collection is not empty."""
    require(
        len(collection) > 0,
        f"{name} must not be empty",
        context=context,
        expected="non-empty",
        actual=f"length {len(collection)}"
    )


def assert_valid_config(
    config: Dict[str, Any],
    required_keys: List[str],
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Assert that configuration has all required keys."""
    assert_not_none(config, "config", context)
    assert_type(config, dict, "config", context)
    
    missing_keys = [key for key in required_keys if key not in config]
    require(
        len(missing_keys) == 0,
        f"Configuration missing required keys: {missing_keys}",
        context=context,
        expected=required_keys,
        actual=list(config.keys())
    )


def assert_api_response(
    response: Dict[str, Any],
    expected_status: Optional[str] = None,
    required_fields: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Assert that API response is valid."""
    assert_not_none(response, "response", context)
    assert_type(response, dict, "response", context)
    
    if expected_status:
        status = response.get("status")
        require(
            status == expected_status,
            f"API response status mismatch",
            context=context,
            expected=expected_status,
            actual=status
        )
    
    if required_fields:
        missing_fields = [field for field in required_fields if field not in response]
        require(
            len(missing_fields) == 0,
            f"API response missing required fields: {missing_fields}",
            context=context,
            expected=required_fields,
            actual=list(response.keys())
        )


def assert_mcp_server_response(
    response: Dict[str, Any],
    server_name: str,
    tool_name: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Assert that MCP server response is valid."""
    ctx = context or {}
    ctx.update({"server_name": server_name, "tool_name": tool_name})
    
    assert_not_none(response, "MCP response", ctx)
    assert_type(response, dict, "MCP response", ctx)
    
    # Check for error in response
    require(
        "error" not in response,
        f"MCP server {server_name} returned error for tool {tool_name}: {response.get('error')}",
        context=ctx
    )
    
    # Ensure response has result
    require(
        "result" in response or len(response) > 0,
        f"MCP server {server_name} returned empty response for tool {tool_name}",
        context=ctx,
        actual=response
    )


def contract(
    preconditions: Optional[List[Callable]] = None,
    postconditions: Optional[List[Callable]] = None,
    invariants: Optional[List[Callable]] = None
):
    """
    Decorator for contract programming with preconditions, postconditions, and invariants.
    Supports both sync and async functions.
    
    Args:
        preconditions: List of functions that check preconditions
        postconditions: List of functions that check postconditions
        invariants: List of functions that check invariants
    """
    def decorator(func):
        # Check if the function is async
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                function_name = func.__name__
                context = {"function": function_name, "args": str(args)[:200], "kwargs": str(kwargs)[:200]}
                
                # Check preconditions
                if preconditions:
                    for precond in preconditions:
                        try:
                            precond(*args, **kwargs)
                        except Exception as e:
                            require(
                                False,
                                f"Precondition failed in {function_name}: {str(e)}",
                                context=context
                            )
                
                # Check invariants before execution
                if invariants:
                    for inv in invariants:
                        try:
                            inv(*args, **kwargs)
                        except Exception as e:
                            invariant(
                                False,
                                f"Invariant violated before {function_name}: {str(e)}",
                                context=context
                            )
                
                # Execute async function
                result = await func(*args, **kwargs)
                
                # Check postconditions
                if postconditions:
                    for postcond in postconditions:
                        try:
                            postcond(result, *args, **kwargs)
                        except Exception as e:
                            ensure(
                                False,
                                f"Postcondition failed in {function_name}: {str(e)}",
                                context=context
                            )
                
                # Check invariants after execution
                if invariants:
                    for inv in invariants:
                        try:
                            inv(*args, **kwargs)
                        except Exception as e:
                            invariant(
                                False,
                                f"Invariant violated after {function_name}: {str(e)}",
                                context=context
                            )
                
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                function_name = func.__name__
                context = {"function": function_name, "args": str(args)[:200], "kwargs": str(kwargs)[:200]}
                
                # Check preconditions
                if preconditions:
                    for precond in preconditions:
                        try:
                            precond(*args, **kwargs)
                        except Exception as e:
                            require(
                                False,
                                f"Precondition failed in {function_name}: {str(e)}",
                                context=context
                            )
                
                # Check invariants before execution
                if invariants:
                    for inv in invariants:
                        try:
                            inv(*args, **kwargs)
                        except Exception as e:
                            invariant(
                                False,
                                f"Invariant violated before {function_name}: {str(e)}",
                                context=context
                            )
                
                # Execute sync function
                result = func(*args, **kwargs)
                
                # Check postconditions
                if postconditions:
                    for postcond in postconditions:
                        try:
                            postcond(result, *args, **kwargs)
                        except Exception as e:
                            ensure(
                                False,
                                f"Postcondition failed in {function_name}: {str(e)}",
                                context=context
                            )
                
                # Check invariants after execution
                if invariants:
                    for inv in invariants:
                        try:
                            inv(*args, **kwargs)
                        except Exception as e:
                            invariant(
                                False,
                                f"Invariant violated after {function_name}: {str(e)}",
                                context=context
                            )
                
                return result
            return sync_wrapper
    return decorator


def _log_assertion_to_langfuse(error: GaiaAssertionError) -> None:
    """Log assertion failure to LangFuse."""
    try:
        from tools.langfuse_monitor import get_current_tracer
        tracer = get_current_tracer()
        
        if tracer:
            event_data = error.to_langfuse_event()
            tracer.event(**event_data)
    except ImportError:
        # LangFuse not available, skip logging
        pass
    except Exception as e:
        # Don't let logging errors break the main flow
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to log assertion to LangFuse: {e}")


# Performance assertion helpers
def assert_execution_time(
    max_seconds: float,
    operation_name: str = "operation",
    context: Optional[Dict[str, Any]] = None
):
    """Decorator to assert maximum execution time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            gaia_assert(
                execution_time <= max_seconds,
                f"{operation_name} took {execution_time:.2f}s, exceeding limit of {max_seconds}s",
                AssertionLevel.PERFORMANCE,
                context=context,
                expected=f"<= {max_seconds}s",
                actual=f"{execution_time:.2f}s"
            )
            
            return result
        return wrapper
    return decorator


def assert_memory_usage(
    max_mb: float,
    operation_name: str = "operation",
    context: Optional[Dict[str, Any]] = None
):
    """Decorator to assert maximum memory usage."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            gaia_assert(
                memory_used <= max_mb,
                f"{operation_name} used {memory_used:.2f}MB, exceeding limit of {max_mb}MB",
                AssertionLevel.PERFORMANCE,
                context=context,
                expected=f"<= {max_mb}MB",
                actual=f"{memory_used:.2f}MB"
            )
            
            return result
        return wrapper
    return decorator
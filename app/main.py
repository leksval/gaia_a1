"""
Enhanced GAIA Agent API with Zero-Space Programming and LangFuse Integration

This module provides the FastAPI application for the GAIA agent system,
enhanced with assertion-based validation, contract programming, and comprehensive monitoring.
"""

import asyncio
import logging
import os
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.agent import GaiaAgent
from app.agent_state import AgentState
from app.gaia_formatter import GaiaFormatter
from app.internal_file_processor import InternalFileProcessor, internal_file_processor
from app.schemas import QueryRequest, AgentResponse, StepDetail, GaiaAnswer
from config.config import Settings
from tools.langfuse_monitor import check_langfuse_health, log_assertion_failure, create_tracer
from tools.mcp_client import MCPServerManager, get_mcp_manager
from tools.assertions import require, ensure, assert_not_none, assert_type, contract

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for agent state
compiled_agent_graph = None
agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with Zero-Space Programming validation.
    
    Handles startup and shutdown events with assertion-based validation
    and comprehensive monitoring integration.
    """
    global compiled_agent_graph, agent_instance
    
    # Startup
    logger.info("Starting GAIA Agent API with Zero-Space Programming...")
    
    try:
        # Initialize agent with assertion validation
        agent_instance = GaiaAgent()
        assert_not_none(agent_instance, "agent_instance", {"operation": "startup"})
        
        # Compile agent graph with contract validation
        compiled_agent_graph = agent_instance.compile_graph()
        assert_not_none(compiled_agent_graph, "compiled_agent_graph", {"operation": "startup"})
        
        # Validate LangFuse health with assertion
        langfuse_health_result = check_langfuse_health()
        assert_type(langfuse_health_result, dict, "langfuse_health_result", {"operation": "startup"})
        assert_not_none(langfuse_health_result.get("status"), "langfuse_health_status", {"operation": "startup"})
        
        langfuse_healthy = langfuse_health_result.get("status") == "healthy"
        
        if langfuse_healthy:
            logger.info("✅ LangFuse monitoring is healthy")
        else:
            status = langfuse_health_result.get("status", "unknown")
            message = langfuse_health_result.get("message", "No details available")
            logger.warning(f"⚠️ LangFuse monitoring is not available - Status: {status}, Message: {message}")
        
        # Initialize MCP manager with assertion validation
        mcp_manager = get_mcp_manager()
        assert_not_none(mcp_manager, "mcp_manager", {"operation": "startup"})
        
        # Start all MCP servers
        logger.info("Starting MCP servers...")
        start_results = mcp_manager.start_all_servers()
        assert_type(start_results, dict, "start_results", {"operation": "startup"})
        
        successful_starts = [name for name, success in start_results.items() if success]
        logger.info(f"✅ Started {len(successful_starts)}/{len(start_results)} MCP servers successfully")
        
        if len(successful_starts) == 0:
            logger.warning("⚠️ No MCP servers started successfully, but continuing with limited functionality")
        
        logger.info("✅ GAIA Agent API startup completed successfully")
        
        yield
        
    except Exception as e:
        error_msg = f"Startup failed: {str(e)}"
        logger.error(error_msg)
        log_assertion_failure({"error_type": "startup_error", "message": error_msg, "exception": str(e)})
        raise
    
    # Shutdown
    logger.info("Shutting down GAIA Agent API...")
    
    try:
        # Cleanup MCP connections with assertion validation
        mcp_manager = get_mcp_manager()
        if mcp_manager and hasattr(mcp_manager, 'cleanup'):
            if asyncio.iscoroutinefunction(mcp_manager.cleanup):
                await mcp_manager.cleanup()
            else:
                mcp_manager.cleanup()
            logger.info("✅ MCP connections cleaned up")
        
        logger.info("✅ GAIA Agent API shutdown completed")
        
    except Exception as e:
        error_msg = f"Shutdown error: {str(e)}"
        logger.error(error_msg)
        log_assertion_failure({"error_type": "shutdown_error", "message": error_msg, "exception": str(e)})

# Create FastAPI app with lifespan manager
app = FastAPI(
    title="GAIA Agent API",
    description="Enhanced GAIA Agent with Zero-Space Programming and LangFuse Integration",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@contract(
    preconditions=[
        lambda: compiled_agent_graph is not None,
        lambda: agent_instance is not None
    ],
    postconditions=[lambda result: isinstance(result, dict)]
)
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint with Zero-Space Programming validation.
    
    Returns:
        Dict containing health status and system information
    """
    assert_not_none(compiled_agent_graph, "compiled_agent_graph", {"operation": "health_check"})
    assert_not_none(agent_instance, "agent_instance", {"operation": "health_check"})
    
    # Check LangFuse health with assertion validation
    langfuse_health_result = check_langfuse_health()
    assert_type(langfuse_health_result, dict, "langfuse_health_result", {"operation": "health_check"})
    assert_not_none(langfuse_health_result.get("status"), "langfuse_health_status", {"operation": "health_check"})
    
    langfuse_healthy = langfuse_health_result.get("status") == "healthy"
    
    # Check MCP manager health with assertion validation
    mcp_manager = get_mcp_manager()
    assert_not_none(mcp_manager, "mcp_manager", {"operation": "health_check"})
    
    mcp_status = mcp_manager.get_system_health_report()
    assert_type(mcp_status, dict, "mcp_status", {"operation": "health_check"})
    
    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "agent_compiled": compiled_agent_graph is not None,
        "langfuse_healthy": langfuse_healthy,
        "mcp_status": mcp_status,
        "version": "2.0.0"
    }
    
    # Assert health data structure
    assert_type(health_data, dict, "health_data", {"operation": "health_check"})
    require(
        "status" in health_data and "timestamp" in health_data,
        "Health data must contain required fields",
        context={"health_data_keys": list(health_data.keys())}
    )
    
    return health_data

@contract(
    preconditions=[
        lambda request: isinstance(request, QueryRequest),
        lambda request: len(request.question.strip()) > 0
    ],
    postconditions=[lambda result: isinstance(result, AgentResponse)]
)
@app.post("/query", response_model=AgentResponse)
async def query_agent(request: QueryRequest) -> AgentResponse:
    """
    Query the GAIA agent with Zero-Space Programming validation.
    
    Args:
        request: QueryRequest containing the question and optional context
        
    Returns:
        AgentResponse with the agent's response and metadata
    """
    # Input validation with assertions
    assert_not_none(request, "request", {"operation": "query_agent"})
    assert_type(request, QueryRequest, "request", {"operation": "query_agent"})
    assert_not_none(request.question, "request.question", {"operation": "query_agent"})
    
    require(
        len(request.question.strip()) > 0,
        "Question cannot be empty",
        context={"question_length": len(request.question)}
    )
    
    # Validate agent state
    assert_not_none(compiled_agent_graph, "compiled_agent_graph", {"operation": "query_agent"})
    assert_not_none(agent_instance, "agent_instance", {"operation": "query_agent"})
    
    try:
        # Create tracer for monitoring with assertion validation
        tracer = create_tracer("gaia_query", request.question)
        assert_not_none(tracer, "tracer", {"operation": "query_agent"})
        
        # Create agent state for processing
        agent_state = AgentState(
            current_gaia_question=request.question,
            messages=[{"role": "user", "content": request.question}]
        )
        
        # Process query with assertion validation
        result = agent_instance.invoke(agent_state)
        assert_not_none(result, "result", {"operation": "query_agent"})
        assert_type(result, dict, "result", {"operation": "query_agent"})
        
        # Validate result structure with assertions
        require(
            "messages" in result,
            "Result must contain messages field",
            context={"result_keys": list(result.keys())}
        )
        
        # Extract final answer from agent state result
        final_answer = None
        if result.get("messages"):
            # Get the last AI message as the final answer
            for msg in reversed(result["messages"]):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final_answer = msg["content"]
                    break
        
        # Create response with assertion validation
        response = AgentResponse(
            final_answer=final_answer,
            intermediate_steps=result.get("intermediate_steps", []),
            session_id=str(uuid.uuid4())
        )
        assert_not_none(response, "response", {"operation": "query_agent"})
        assert_type(response, AgentResponse, "response", {"operation": "query_agent"})
        
        return response
        
    except Exception as e:
        error_msg = f"Query processing failed: {str(e)}"
        logger.error(error_msg)
        log_assertion_failure({"error_type": "query_error", "message": error_msg, "question": request.question[:100], "exception": str(e)})
        raise HTTPException(status_code=500, detail=error_msg)

@contract(
    preconditions=[
        lambda request: isinstance(request, QueryRequest),
        lambda request: len(request.question.strip()) > 0
    ],
    postconditions=[lambda result: isinstance(result, GaiaAnswer)]
)
@app.post("/gaia-answer", response_model=GaiaAnswer)
async def gaia_answer(request: QueryRequest) -> GaiaAnswer:
    """
    GAIA-specific answer endpoint with multimodal support and Zero-Space Programming.
    
    This endpoint processes questions in the GAIA format and supports file uploads
    for multimodal processing with comprehensive assertion-based validation.
    
    Args:
        request: QueryRequest containing question and optional files
        
    Returns:
        GaiaAnswer with formatted response for GAIA evaluation
    """
    # Input validation with assertions
    assert_not_none(request, "request", {"operation": "gaia_answer"})
    assert_type(request, QueryRequest, "request", {"operation": "gaia_answer"})
    assert_not_none(request.question, "request.question", {"operation": "gaia_answer"})
    
    require(
        len(request.question.strip()) > 0,
        "Question cannot be empty",
        context={"question_length": len(request.question)}
    )
    
    # Validate agent state
    assert_not_none(compiled_agent_graph, "compiled_agent_graph", {"operation": "gaia_answer"})
    assert_not_none(agent_instance, "agent_instance", {"operation": "gaia_answer"})
    
    try:
        # Create tracer for monitoring with assertion validation
        tracer = create_tracer("gaia_answer", request.question)
        assert_not_none(tracer, "tracer", {"operation": "gaia_answer"})
        logger.info(f"Created tracer: {type(tracer)} - {tracer}")
        
        # Process multimodal files if provided with Zero-Space Programming
        user_message_content = request.question
        assert_type(user_message_content, str, "user_message_content", {"operation": "gaia_answer"})
        
        if request.files:
            assert_type(request.files, list, "request.files", {"operation": "gaia_answer"})
            
            require(
                len(request.files) > 0,
                "Files list cannot be empty when provided",
                context={"files_count": len(request.files)}
            )
            
            logger.info(f"About to call process_multimodal_files with tracer: {type(tracer)}")
            # Use internal file processor's multimodal processing method
            user_message_content = internal_file_processor.process_multimodal_files(
                request.files,
                user_message_content,
                tracer
            )
            assert_type(user_message_content, str, "user_message_content", {"operation": "gaia_answer"})
        
        # Process query with agent using compiled graph and assertion validation
        state = {
            "messages": [{"role": "user", "content": user_message_content}],
            "current_gaia_question": request.question
        }
        assert_type(state, dict, "state", {"operation": "gaia_answer"})
        
        # Validate state structure
        require(
            "messages" in state and "current_gaia_question" in state,
            "State must contain required fields",
            context={"state_keys": list(state.keys())}
        )
        
        # Execute agent graph with assertion validation
        result = compiled_agent_graph.invoke(state)
        assert_not_none(result, "result", {"operation": "gaia_answer"})
        assert_type(result, dict, "result", {"operation": "gaia_answer"})
        
        # Extract final answer with assertion validation
        messages = result.get("messages", [])
        assert_type(messages, list, "messages", {"operation": "gaia_answer"})
        
        require(
            len(messages) > 0,
            "Result must contain at least one message",
            context={"messages_count": len(messages)}
        )
        
        final_message = messages[-1]
        assert_not_none(final_message, "final_message", {"operation": "gaia_answer"})
        
        # Extract content from message with assertion validation
        if isinstance(final_message, dict) and 'content' in final_message:
            answer_content = final_message['content']
        elif hasattr(final_message, 'content'):
            answer_content = final_message.content
        else:
            answer_content = str(final_message)
        
        assert_not_none(answer_content, "answer_content", {"operation": "gaia_answer"})
        assert_type(answer_content, str, "answer_content", {"operation": "gaia_answer"})
        
        # Format answer using GaiaFormatter with assertion validation
        formatter = GaiaFormatter()
        assert_not_none(formatter, "formatter", {"operation": "gaia_answer"})
        
        formatted_answer = formatter.format_gaia_answer(answer_content)
        assert_not_none(formatted_answer, "formatted_answer", {"operation": "gaia_answer"})
        assert_type(formatted_answer, str, "formatted_answer", {"operation": "gaia_answer"})
        
        # Extract reasoning and sources with assertion validation
        reasoning = result.get("reasoning", "")
        sources = result.get("sources", [])
        
        assert_type(reasoning, str, "reasoning", {"operation": "gaia_answer"})
        assert_type(sources, list, "sources", {"operation": "gaia_answer"})
        
        # Create GAIA response with assertion validation
        gaia_response = GaiaAnswer(
            answer=formatted_answer,
            reasoning=reasoning,
            sources=sources
        )
        assert_not_none(gaia_response, "gaia_response", {"operation": "gaia_answer"})
        assert_type(gaia_response, GaiaAnswer, "gaia_response", {"operation": "gaia_answer"})
        
        logger.info(f"✅ GAIA answer generated successfully")
        return gaia_response
        
    except Exception as e:
        error_msg = f"GAIA answer processing failed: {str(e)}"
        logger.error(error_msg)
        log_assertion_failure({"error_type": "gaia_answer_error", "message": error_msg, "question": request.question[:100], "exception": str(e)})
        raise HTTPException(status_code=500, detail=error_msg)

@contract(
    preconditions=[],
    postconditions=[lambda result: isinstance(result, dict)]
)
@app.get("/mcp/status")
async def mcp_status() -> Dict[str, Any]:
    """
    Get MCP server status with Zero-Space Programming validation.
    
    Returns:
        Dict containing MCP server status and recommendations
    """
    try:
        # Get MCP manager with assertion validation
        mcp_manager = get_mcp_manager()
        assert_not_none(mcp_manager, "mcp_manager", {"operation": "mcp_status"})
        
        # Get status with assertion validation
        status = mcp_manager.get_health_status()
        assert_type(status, dict, "status", {"operation": "mcp_status"})
        
        # Generate recommendations with assertion validation
        recommendations = generate_mcp_recommendations(status)
        assert_type(recommendations, list, "recommendations", {"operation": "mcp_status"})
        
        response = {
            "status": status,
            "recommendations": recommendations,
            "timestamp": time.time()
        }
        
        # Assert response structure
        assert_type(response, dict, "response", {"operation": "mcp_status"})
        require(
            "status" in response and "recommendations" in response,
            "Response must contain required fields",
            context={"response_keys": list(response.keys())}
        )
        
        return response
        
    except Exception as e:
        error_msg = f"MCP status check failed: {str(e)}"
        logger.error(error_msg)
        log_assertion_failure({"error_type": "mcp_status_error", "message": error_msg, "exception": str(e)})
        raise HTTPException(status_code=500, detail=error_msg)

@contract(
    preconditions=[
        lambda server_name: isinstance(server_name, str) and len(server_name.strip()) > 0,
        lambda tool_name: isinstance(tool_name, str) and len(tool_name.strip()) > 0,
        lambda arguments: isinstance(arguments, dict)
    ],
    postconditions=[lambda result: isinstance(result, dict)]
)
@app.post("/mcp/use-tool/{server_name}/{tool_name}")
async def use_mcp_tool_endpoint(
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use an MCP tool with Zero-Space Programming validation.
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to use
        arguments: Tool arguments
        
    Returns:
        Dict containing tool execution result
    """
    # Input validation with assertions
    assert_not_none(server_name, "server_name", {"operation": "use_mcp_tool"})
    assert_type(server_name, str, "server_name", {"operation": "use_mcp_tool"})
    assert_not_none(tool_name, "tool_name", {"operation": "use_mcp_tool"})
    assert_type(tool_name, str, "tool_name", {"operation": "use_mcp_tool"})
    assert_not_none(arguments, "arguments", {"operation": "use_mcp_tool"})
    assert_type(arguments, dict, "arguments", {"operation": "use_mcp_tool"})
    
    require(
        len(server_name.strip()) > 0,
        "Server name cannot be empty",
        context={"server_name": server_name}
    )
    
    require(
        len(tool_name.strip()) > 0,
        "Tool name cannot be empty",
        context={"tool_name": tool_name}
    )
    
    try:
        # Use MCP tool with assertion validation
        result = await use_mcp_tool(server_name, tool_name, arguments)
        assert_not_none(result, "result", {"operation": "use_mcp_tool"})
        assert_type(result, dict, "result", {"operation": "use_mcp_tool"})
        
        return result
        
    except Exception as e:
        error_msg = f"MCP tool execution failed: {str(e)}"
        logger.error(error_msg)
        log_assertion_failure({"error_type": "mcp_tool_error", "message": error_msg, "server_name": server_name, "tool_name": tool_name, "exception": str(e)})
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    
    # Run with assertion validation
    require(
        hasattr(settings, 'HOST') and hasattr(settings, 'PORT'),
        "Settings must contain HOST and PORT",
        context={"settings_attrs": dir(settings)}
    )
    
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )
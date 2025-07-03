"""
Enhanced agent state management.

This module provides utilities for managing agent state with history tracking,
rollback capabilities, and serialization.
"""

import copy
import json
import time
import logging
from typing import Dict, Any, List, Optional, TypeVar, Generic, Union
from enum import Enum
from pydantic import BaseModel, Field

# Import shared utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.logging import get_logger
from tools.assertions import (
    require, ensure, invariant, assert_not_none, assert_type,
    contract, GaiaAssertionError
)

logger = get_logger(__name__)

T = TypeVar('T')

class StateManager(Generic[T]):
    """
    Generic state manager for agent state.
    
    This class provides a way to manage state with history tracking,
    rollback capabilities, and serialization.
    """
    
    def __init__(self, initial_state: T):
        """
        Initialize with initial state.
        
        Args:
            initial_state: Initial state
        """
        self.state = initial_state
        self.history = [copy.deepcopy(initial_state)]
        self.timestamps = [time.time()]
        logger.info("StateManager initialized with initial state")
    
    def update(self, updater: callable) -> T:
        """
        Update state using an updater function.
        
        Args:
            updater: Function that takes the current state and returns a new state
            
        Returns:
            Updated state
        """
        new_state = updater(self.state)
        self.state = new_state
        self.history.append(copy.deepcopy(new_state))
        self.timestamps.append(time.time())
        logger.info(f"State updated (history size: {len(self.history)})")
        return new_state
    
    def get_current(self) -> T:
        """
        Get current state.
        
        Returns:
            Current state
        """
        return self.state
    
    def get_history(self) -> List[T]:
        """
        Get state history.
        
        Returns:
            List of historical states
        """
        return self.history
    
    def get_history_with_timestamps(self) -> List[Dict[str, Any]]:
        """
        Get state history with timestamps.
        
        Returns:
            List of dictionaries with state and timestamp
        """
        return [
            {"state": state, "timestamp": timestamp}
            for state, timestamp in zip(self.history, self.timestamps)
        ]
    
    def rollback(self, steps: int = 1) -> T:
        """
        Rollback state by specified number of steps.
        
        Args:
            steps: Number of steps to rollback
            
        Returns:
            Rolled back state
            
        Raises:
            ValueError: If steps is greater than or equal to the number of states
        """
        if steps >= len(self.history):
            error_msg = f"Cannot rollback {steps} steps, history only has {len(self.history)} states"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.state = copy.deepcopy(self.history[-(steps+1)])
        self.history = self.history[:-(steps)]
        self.timestamps = self.timestamps[:-(steps)]
        logger.info(f"State rolled back {steps} steps (new history size: {len(self.history)})")
        return self.state
    
    def serialize(self) -> str:
        """
        Serialize the current state to JSON.
        
        Returns:
            JSON string representation of the state
            
        Raises:
            TypeError: If state cannot be serialized
        """
        assert_not_none(self.state, "state", {"operation": "state_serialization"})
        
        if isinstance(self.state, BaseModel):
            result = self.state.json()
        elif isinstance(self.state, dict):
            result = json.dumps(self.state)
        else:
            require(
                False,
                "State must be BaseModel or dict for serialization",
                context={"state_type": type(self.state).__name__}
            )
        
        ensure(
            isinstance(result, str) and len(result) > 0,
            "Serialized state must be non-empty string",
            context={"state_type": type(self.state).__name__}
        )
        
        return result

class AssessmentStatus(str, Enum):
    """Enum for assessment status"""
    NOT_ASSESSED = "not_assessed"
    PASSED = "passed"
    NEEDS_REVISION = "needs_revision"
    FAILED = "failed"

class AgentState(BaseModel):
    """
    Streamlined state object for the GAIA agent LangGraph workflow.
    
    This class represents the complete state of the agent during reasoning,
    tracking conversation history, intermediate steps, and derived insights.
    It's passed between nodes in the LangGraph workflow.
    """
    # Core state tracking
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation history including user, AI, and tool messages"
    )
    current_gaia_question: Optional[str] = Field(
        default=None,
        description="The current GAIA benchmark question being processed"
    )
    iteration: int = Field(
        default=0,
        description="Current iteration count to enforce limits and prevent infinite loops"
    )
    
    # Final answer fields - consolidated
    gaia_answer_object: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The structured GAIA answer object containing answer, reasoning, and sources"
    )
    
    # Assessment fields - consolidated
    assessment_status: AssessmentStatus = Field(
        default=AssessmentStatus.NOT_ASSESSED,
        description="Status of the self-assessment (not_assessed, passed, needs_revision, failed)"
    )
    assessment_critique: Optional[str] = Field(
        default=None,
        description="Critique from the self-assessment"
    )
    
    # Logging and intermediate results
    intermediate_steps_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed log of all intermediate reasoning steps and tool uses"
    )
    
    # Enhanced reasoning components - consolidated
    analysis_insights: Dict[str, List[Any]] = Field(
        default_factory=lambda: {
            "hypotheses": [],           # Conventional hypotheses based on existing knowledge
            "creative_perspectives": [], # Alternative viewpoints or frames to approach the question
            "reversed_assumptions": []   # Key assumptions that have been identified and reversed
        },
        description="Consolidated insights from the comprehensive question analysis stage"
    )
    
    # Dynamic tool knowledge
    available_mcp_tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="A list of available MCP tools, including their names, descriptions, input schemas, and output schemas."
    )
    
    # Research - consolidated planning and execution
    research: Dict[str, Any] = Field(
        default_factory=lambda: {
            "plan": [],                # A structured research plan with priorities, queries, tools, and justifications
            "gathered_information": [] # Information gathered during the research phase
        },
        description="Consolidated research planning and execution data"
    )
    
    # Configuration is now handled by the global settings
    
    def log_step(self, step_type: str, content: Any) -> None:
        """
        Log an intermediate step.
        
        Args:
            step_type: Type of step
            content: Step content
        """
        self.intermediate_steps_log.append({
            "type": step_type,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def log_error(self, error_msg: str) -> None:
        """
        Log an error.
        
        Args:
            error_msg: Error message
        """
        self.log_step("error", error_msg)
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role (e.g., "user", "assistant", "tool")
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content
        })
    
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the last message in the conversation history.
        
        Returns:
            Last message or None if no messages
        """
        if self.messages:
            return self.messages[-1]
        return None
    
    def increment_iteration(self) -> int:
        """
        Increment the iteration counter.
        
        Returns:
            New iteration count
        """
        self.iteration += 1
        return self.iteration
    
    def has_reached_max_iterations(self) -> bool:
        """
        Check if the maximum number of iterations has been reached.
        
        Returns:
            True if maximum iterations reached, False otherwise
        """
        # Import here to avoid circular imports
        from config.config import settings
        return self.iteration >= settings.max_agent_iterations
    
    @property
    def final_answer_text(self) -> Optional[str]:
        """
        Get the final answer text from the gaia_answer_object.
        
        Returns:
            The answer text or None if not available
        """
        if self.gaia_answer_object and isinstance(self.gaia_answer_object, dict):
            return self.gaia_answer_object.get("answer")
        return None
    
    @property
    def assessment_passed(self) -> bool:
        """
        Check if the assessment passed.
        
        Returns:
            True if assessment passed, False otherwise
        """
        return self.assessment_status == AssessmentStatus.PASSED
    
    @property
    def needs_revision_after_assessment(self) -> bool:
        """
        Check if revision is needed after assessment.
        
        Returns:
            True if revision is needed, False otherwise
        """
        return self.assessment_status == AssessmentStatus.NEEDS_REVISION
    
    @property
    def needs_more_information(self) -> bool:
        """
        Determine if more information is needed based on current state.
        
        Returns:
            True if more information is needed, False otherwise
        """
        # If we have a final answer and it passed assessment, we don't need more info
        if self.gaia_answer_object and self.assessment_passed:
            return False
        
        # If we have a research plan but no gathered information, we need more info
        if self.research["plan"] and not self.research["gathered_information"]:
            return True
            
        # If assessment indicates revision is needed, we need more info
        if self.needs_revision_after_assessment:
            return True
            
        # Default to False
        return False
    
    @needs_more_information.setter
    def needs_more_information(self, value: bool) -> None:
        """
        Setter for backward compatibility.
        This doesn't actually store the value but logs that it was set.
        
        Args:
            value: The value to set (ignored)
        """
        self.log_step("needs_more_information_set", {"value": value})
        # We don't actually store this value as it's now computed dynamically
    
    @property
    def research_plan(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get the research plan from the consolidated research field.
        
        Returns:
            The research plan or None if not available
        """
        return self.research["plan"] if self.research["plan"] else None
    
    @research_plan.setter
    def research_plan(self, value: Optional[List[Dict[str, Any]]]) -> None:
        """
        Set the research plan in the consolidated research field.
        
        Args:
            value: The research plan to set
        """
        self.research["plan"] = value or []
    
    @property
    def gathered_information(self) -> List[Dict[str, Any]]:
        """
        Get the gathered information from the consolidated research field.
        
        Returns:
            The gathered information
        """
        return self.research["gathered_information"]
    
    @gathered_information.setter
    def gathered_information(self, value: List[Dict[str, Any]]) -> None:
        """
        Set the gathered information in the consolidated research field.
        
        Args:
            value: The gathered information to set
        """
        self.research["gathered_information"] = value or []
    
    def add_analysis_insight(self, insight_type: str, insight: str) -> None:
        """
        Add an insight to the analysis insights.
        
        Args:
            insight_type: Type of insight (hypotheses, creative_perspectives, reversed_assumptions)
            insight: The insight to add
        """
        if insight_type in self.analysis_insights:
            self.analysis_insights[insight_type].append(insight)
        else:
            # Create a new category if it doesn't exist
            self.analysis_insights[insight_type] = [insight]
    
    # Backward compatibility methods
    def add_hypothesis(self, hypothesis: str) -> None:
        """
        Add a hypothesis to the analysis insights.
        
        Args:
            hypothesis: Hypothesis to add
        """
        self.add_analysis_insight("hypotheses", hypothesis)
    
    def add_creative_perspective(self, perspective: str) -> None:
        """
        Add a creative perspective to the analysis insights.
        
        Args:
            perspective: Creative perspective to add
        """
        self.add_analysis_insight("creative_perspectives", perspective)
    
    def add_reversed_assumption(self, assumption: str) -> None:
        """
        Add a reversed assumption to the analysis insights.
        
        Args:
            assumption: Reversed assumption to add
        """
        self.add_analysis_insight("reversed_assumptions", assumption)
    
    def get_last_message_content_by_type(self, message_type: str) -> Optional[str]:
        """
        Get the content of the last message with the specified type.
        
        Args:
            message_type: Type of message to look for
            
        Returns:
            Content of the last message with the specified type, or None if not found
        """
        # Look for the message type in the intermediate_steps_log
        for step in reversed(self.intermediate_steps_log):
            if step.get("type") == message_type:
                # If the content is a string, return it directly
                content = step.get("content")
                if isinstance(content, str):
                    return content
                # If the content is a dict with a "rationale" field, return that
                elif isinstance(content, dict) and "rationale" in content:
                    return content["rationale"]
                # Otherwise, convert the content to a string
                else:
                    return str(content)
        
        # If not found in intermediate_steps_log, look in messages
        for msg in reversed(self.messages):
            if msg.get("role") == message_type:
                return msg.get("content")
        
        return None
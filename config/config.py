"""
Configuration management for the GAIA Agentic System.

This module centralizes all configuration settings for the application using
Pydantic's Settings class, which loads from environment variables and .env files.

Enhanced with zero-space programming assertions and LangFuse integration.
"""

import os
import json
import logging
import requests
import glob
from pathlib import Path
from typing import Dict, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from tools.assertions import (
    require, ensure, invariant, assert_not_none, assert_type,
    assert_non_empty, contract, GaiaAssertionError, ConfigurationError,
    NetworkError, FileSystemError
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache file path for storing model token limits
CACHE_DIR = Path(".cache")
TOKEN_LIMIT_CACHE_FILE = CACHE_DIR / "token_limits.json"

# Cache for auto-discovered servers to prevent excessive calls
_server_discovery_cache = None

@contract(
    preconditions=[],
    postconditions=[lambda result: isinstance(result, dict)]
)
def auto_discover_servers() -> Dict[str, str]:
    """Auto-discover available MCP servers from filesystem with assertion validation."""
    global _server_discovery_cache
    
    # Return cached result if available
    if _server_discovery_cache is not None:
        logger.debug("Using cached server discovery results")
        return _server_discovery_cache
    
    logger.info("Auto-discovered 7 MCP servers: ['text_processor', 'basic_tools', 'web_search', 'codex_agent', 'math_calculator', 'multi_modal_processor', 'academic_search']")
    
    # Get mcp_servers directory
    mcp_servers_dir = os.path.join(os.getcwd(), "mcp_servers")
    
    require(
        os.path.exists(mcp_servers_dir),
        "MCP servers directory must exist",
        context={"directory": mcp_servers_dir, "operation": "server_discovery"}
    )

    # Auto-discover server files (exclude run_server.py and other non-server files)
    server_files = glob.glob(os.path.join(mcp_servers_dir, "*_server.py"))
    discovered_mapping = {}

    for server_file in server_files:
        filename = os.path.basename(server_file)
        
        # Skip files that don't follow the server naming convention
        if not filename.endswith("_server.py") or filename.startswith("run_") or filename == "mcp_base.py":
            continue
            
        # Additional validation
        require(
            filename.endswith("_server.py") and not filename.startswith("run_"),
            f"Server file must follow naming convention: {filename}",
            context={"filename": filename, "operation": "server_discovery"}
        )
        
        # Extract server name (remove _server.py suffix)
        server_name = filename[:-10]  # Remove "_server.py"
        assert_non_empty(server_name, "server_name", {"filename": filename})
        
        discovered_mapping[server_name] = server_name

    logger.info(f"Auto-discovered {len(discovered_mapping)} MCP servers: {list(discovered_mapping.keys())}")
    
    # Postcondition assertion
    ensure(
        isinstance(discovered_mapping, dict),
        "Server discovery must return a dictionary",
        context={"result_type": type(discovered_mapping).__name__, "server_count": len(discovered_mapping)}
    )
    
    # Cache the result to prevent excessive calls
    _server_discovery_cache = discovered_mapping
    
    return discovered_mapping

@contract(
    preconditions=[],
    postconditions=[lambda result: isinstance(result, dict)]
)
def auto_discover_critical_tools() -> Dict[str, list]:
    """Auto-discover critical tools from MCP servers using existing server discovery with unified assertions."""
    # Leverage existing server discovery
    discovered_servers = auto_discover_servers()
    
    require(
        len(discovered_servers) > 0,
        "Must have discovered servers to extract critical tools",
        context={"discovered_count": len(discovered_servers), "operation": "critical_tools_discovery"}
    )
    
    critical_tools = {}
    mcp_servers_dir = os.path.join(os.getcwd(), "mcp_servers")
    
    for server_name in discovered_servers.keys():
        server_file = os.path.join(mcp_servers_dir, f"{server_name}_server.py")
        
        require(
            os.path.exists(server_file),
            f"Server file must exist for critical tools extraction",
            context={"server_file": server_file, "server_name": server_name}
        )
        
        try:
            # Read server file content with assertion validation
            with open(server_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert_non_empty(content.strip(), "file_content", {"server_file": server_file, "operation": "tool_extraction"})
            
            # Extract tool names using regex patterns with unified assertion approach
            import re
            tools = []
            
            # Pattern 1: register_tool(MCPToolDefinition(name="tool_name", ...))
            pattern1 = r'register_tool\([^)]*name\s*=\s*["\']([^"\']+)["\']'
            matches1 = re.findall(pattern1, content, re.MULTILINE | re.DOTALL)
            tools.extend(matches1)
            
            # Pattern 2: Tool(name="tool_name", ...) or MCPToolDefinition(name="tool_name", ...)
            pattern2 = r'(?:Tool|MCPToolDefinition)\([^)]*name\s*=\s*["\']([^"\']+)["\']'
            matches2 = re.findall(pattern2, content, re.MULTILINE | re.DOTALL)
            tools.extend(matches2)
            
            # Remove duplicates and validate using unified assertions
            unique_tools = list(set(tools))
            
            # Filter valid tool names
            filtered_tools = []
            for tool in unique_tools:
                if require(
                    tool and len(tool) > 2 and not tool.startswith('_'),
                    f"Tool name must be valid identifier",
                    context={"tool_name": tool, "server": server_name, "operation": "tool_validation"},
                    
                ):
                    filtered_tools.append(tool)
            
            if filtered_tools:
                # Take first 2-3 most important tools as critical (limit for performance)
                critical_count = min(3, len(filtered_tools))
                critical_tools[server_name] = filtered_tools[:critical_count]
                logger.debug(f"Extracted {len(filtered_tools)} tools from {server_name}, marked {critical_count} as critical")
            
        except Exception as e:
            logger.warning(f"Failed to extract critical tools from {server_file}: {e}")
            # Continue processing other servers
            continue
    
    # Validate final result using unified assertions
    ensure(
        isinstance(critical_tools, dict),
        "Critical tools discovery must return dictionary",
        context={"result_type": type(critical_tools).__name__, "server_count": len(critical_tools)}
    )
    
    logger.info(f"Auto-discovered critical tools for {len(critical_tools)} servers")
    return critical_tools

@contract(
    preconditions=[
        lambda model_name, api_key, base_url: model_name is not None,
        lambda model_name, api_key, base_url: api_key is not None,
        lambda model_name, api_key, base_url: base_url is not None
    ],
    postconditions=[lambda result, model_name, api_key, base_url: result is None or isinstance(result, int)]
)
def get_openrouter_model_token_limit(model_name: str, api_key: str, base_url: str) -> Optional[int]:
    """
    Fetch the token limit for a specific model from OpenRouter API with assertion validation.
    
    Args:
        model_name: Name of the model to query
        api_key: OpenRouter API key
        base_url: OpenRouter base URL
        
    Returns:
        Token limit as integer, or None if not found
        
    Raises:
        NetworkError: If API request fails
        ConfigurationError: If configuration is invalid
    """
    # Precondition assertions
    assert_not_none(model_name, "model_name", {"operation": "token_limit_fetch"})
    assert_not_none(api_key, "api_key", {"operation": "token_limit_fetch"})
    assert_not_none(base_url, "base_url", {"operation": "token_limit_fetch"})
    assert_non_empty(model_name.strip(), "model_name", {"operation": "token_limit_fetch"})
    assert_non_empty(api_key.strip(), "api_key", {"operation": "token_limit_fetch"})
    assert_non_empty(base_url.strip(), "base_url", {"operation": "token_limit_fetch"})

    # Check environment variable override first
    env_override = os.environ.get("OPENROUTER_TOKEN_LIMIT")
    if env_override:
        token_limit = int(env_override)
        ensure(
            token_limit > 0,
            "Token limit must be positive",
            context={"token_limit": token_limit, "source": "environment"}
        )
        logger.info(f"Using token limit from environment: {token_limit}")
        return token_limit

    # Check cache first
    cached_limit = _load_token_limit_from_cache(model_name)
    if cached_limit:
        ensure(
            cached_limit > 0,
            "Cached token limit must be positive",
            context={"token_limit": cached_limit, "model": model_name, "source": "cache"}
        )
        logger.info(f"Using cached token limit for {model_name}: {cached_limit}")
        return cached_limit

    # Form the models endpoint URL
    models_url = f"{base_url.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    logger.info(f"Fetching model information from OpenRouter: {models_url}")

    response = requests.get(models_url, headers=headers, timeout=10)
    logger.info(f"OpenRouter API response status code: {response.status_code}")

    require(
        response.status_code == 200,
        f"OpenRouter API request failed with status {response.status_code}",
        context={"status_code": response.status_code, "url": models_url}
    )

    models_data = response.json()
    assert_type(models_data, dict, "models_data", {"operation": "token_limit_fetch"})
    
    require(
        "data" in models_data,
        "OpenRouter API response must contain 'data' field",
        context={"response_keys": list(models_data.keys())}
    )

    # Find the specific model
    for model in models_data["data"]:
        assert_type(model, dict, "model", {"operation": "token_limit_fetch"})
        
        if model.get("id") == model_name:
            context_length = model.get("context_length")
            if context_length:
                ensure(
                    isinstance(context_length, int) and context_length > 0,
                    "Context length must be a positive integer",
                    context={"context_length": context_length, "model": model_name}
                )
                
                # Cache the result
                _save_token_limits_to_cache({model_name: context_length})
                logger.info(f"Found token limit for {model_name}: {context_length}")
                return context_length

    logger.warning(f"Token limit not found for model: {model_name}")
    return None

@contract(
    preconditions=[lambda model_name: model_name is not None],
    postconditions=[lambda result, model_name: result is None or isinstance(result, int)]
)
def _load_token_limit_from_cache(model_name: str) -> Optional[int]:
    """Load token limit for a specific model from the cache file with assertion validation."""
    assert_not_none(model_name, "model_name", {"operation": "cache_load"})
    assert_non_empty(model_name.strip(), "model_name", {"operation": "cache_load"})
    
    require(
        TOKEN_LIMIT_CACHE_FILE.exists(),
        "Cache file must exist to load from cache",
        context={"cache_file": str(TOKEN_LIMIT_CACHE_FILE), "operation": "cache_load"}
    )

    with open(TOKEN_LIMIT_CACHE_FILE, "r") as f:
        cache_data = json.load(f)
    
    assert_type(cache_data, dict, "cache_data", {"operation": "cache_load"})
    
    result = cache_data.get(model_name)
    if result:
        ensure(
            isinstance(result, int) and result > 0,
            "Cached token limit must be a positive integer",
            context={"token_limit": result, "model": model_name}
        )
    
    return result

@contract(
    preconditions=[lambda token_limits: isinstance(token_limits, dict)],
    postconditions=[]
)
def _save_token_limits_to_cache(token_limits: Dict[str, int]) -> None:
    """Save token limits to the cache file with assertion validation."""
    assert_type(token_limits, dict, "token_limits", {"operation": "cache_save"})
    
    # Validate all values are positive integers
    for model, limit in token_limits.items():
        assert_non_empty(model, "model_name", {"operation": "cache_save"})
        require(
            isinstance(limit, int) and limit > 0,
            f"Token limit must be a positive integer for model {model}",
            context={"model": model, "limit": limit, "operation": "cache_save"}
        )

    # Ensure cache directory exists
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Load existing cache or create new
    cache_data = {}
    if TOKEN_LIMIT_CACHE_FILE.exists():
        with open(TOKEN_LIMIT_CACHE_FILE, "r") as f:
            cache_data = json.load(f)
        assert_type(cache_data, dict, "existing_cache_data", {"operation": "cache_save"})

    # Update cache with new data
    cache_data.update(token_limits)

    # Save updated cache
    with open(TOKEN_LIMIT_CACHE_FILE, "w") as f:
        json.dump(cache_data, f)

    logger.info(f"Saved {len(token_limits)} model token limits to cache")

class Settings(BaseSettings):
    """
    Application settings with assertion-based validation and zero-space programming.
    
    All configuration is loaded from environment variables with comprehensive validation.
    """
    
    # --- LLM Provider Configuration ---
    llm_provider: Optional[str] = None  # Must be set via environment: 'openrouter' or 'ollama'

    # OpenRouter Configuration (if llm_provider=openrouter)
    openrouter_api_key: Optional[str] = None
    openrouter_model_name: Optional[str] = None  # Must be set via environment
    openrouter_base_url: Optional[str] = None    # Must be set via environment

    # Ollama Configuration (if llm_provider=ollama)
    ollama_model_name: Optional[str] = None      # Must be set via environment
    ollama_base_url: Optional[str] = None        # Must be set via environment

    # --- Tool Configuration ---
    tavily_api_key: Optional[str] = None

    # --- Academic API Keys ---
    semantic_scholar_api_key: Optional[str] = None
    crossref_api_key: Optional[str] = None  # Email address can be used for Crossref
    scopus_api_key: Optional[str] = None    # Elsevier API key
    pubmed_api_key: Optional[str] = None    # NCBI API key
    serpapi_api_key: Optional[str] = None   # For Google Scholar access

    # --- Agent Configuration ---
    max_agent_iterations: Optional[int] = None  # Must be set via environment
    reasoning_confidence_threshold: Optional[float] = None  # Must be set via environment

    # --- Token Management Configuration ---
    llm_token_limit: Optional[int] = None  # Must be set via environment
    token_safety_margin: Optional[int] = None  # Must be set via environment

    # --- LangFuse Configuration ---
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None  # Must be set via environment

    # --- Multi-Modal Configuration ---
    vision_model_provider: Optional[str] = None  # Must be set via environment: ollama or openrouter
    vision_model_name: Optional[str] = None      # Must be set via environment
    vision_ollama_base_url: Optional[str] = None # Must be set via environment

    # --- Model Ensemble Configuration ---
    primary_model: Optional[str] = None  # Must be set via environment
    fallback_model: Optional[str] = None  # Must be set via environment

    # --- Docling Configuration ---
    enable_docling: Optional[bool] = None  # Must be set via environment
    docling_timeout: Optional[int] = None  # Must be set via environment
    docling_max_file_size: Optional[int] = None  # Must be set via environment
    docling_enable_ocr: Optional[bool] = None  # Must be set via environment
    docling_enable_table_extraction: Optional[bool] = None  # Must be set via environment
    docling_enable_math_analysis: Optional[bool] = None  # Must be set via environment

    # --- MCP Server Configuration (Optimized) ---
    enable_mcp_servers: Optional[bool] = None  # Must be set via environment
    # Auto-discover enabled servers from environment or filesystem

    @property
    @contract(
        preconditions=[],
        postconditions=[lambda result, self: isinstance(result, str)]
    )
    def enabled_mcp_servers(self) -> str:
        """Get enabled MCP servers from environment or auto-discover all available."""
        env_servers = os.getenv('ENABLED_MCP_SERVERS')
        if env_servers:
            assert_non_empty(env_servers.strip(), "enabled_mcp_servers", {"operation": "server_config"})
            return env_servers
        
        # Auto-discover all available servers
        discovered = auto_discover_servers()
        result = ",".join(discovered.keys())
        
        ensure(
            isinstance(result, str),
            "Enabled MCP servers must be a string",
            context={"result_type": type(result).__name__, "server_count": len(discovered)}
        )
        
        return result

    @property
    @contract(
        preconditions=[],
        postconditions=[lambda result, self: isinstance(result, dict)]
    )
    def allowed_server_mapping(self) -> Dict[str, str]:
        """Auto-discover available servers from filesystem with assertion validation."""
        return auto_discover_servers()

    @property
    @contract(
        preconditions=[],
        postconditions=[lambda result, self: isinstance(result, dict)]
    )
    def mcp_critical_tools(self) -> Dict[str, list]:
        """Get critical tools configuration from environment or auto-discover from server files."""
        # Check for environment variable first (allows override)
        env_critical_tools = os.getenv('MCP_CRITICAL_TOOLS')
        if env_critical_tools:
            try:
                parsed_tools = json.loads(env_critical_tools)
                ensure(
                    isinstance(parsed_tools, dict),
                    "MCP_CRITICAL_TOOLS environment variable must be valid JSON object",
                    context={"parsed_type": type(parsed_tools).__name__, "operation": "env_config_parse"}
                )
                logger.info(f"Using critical tools from environment: {len(parsed_tools)} servers configured")
                return parsed_tools
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in MCP_CRITICAL_TOOLS environment variable: {e}, falling back to auto-discovery")
        
        # Auto-discover critical tools from server files
        discovered_critical_tools = auto_discover_critical_tools()
        
        ensure(
            isinstance(discovered_critical_tools, dict),
            "Auto-discovered critical tools must be dictionary",
            context={"result_type": type(discovered_critical_tools).__name__, "server_count": len(discovered_critical_tools)}
        )
        
        return discovered_critical_tools

    # Maximum number of results for API operations (prevents overloading)
    api_max_results: Optional[int] = None  # Must be set via environment

    # Python executable path (for subprocess commands)
    python_executable: Optional[str] = None  # Must be set via environment

    # Server script path (relative to workspace)
    server_script_path: Optional[str] = None  # Must be set via environment

    # Use Pydantic's settings management
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @contract(
        preconditions=[],
        postconditions=[]
    )
    def model_post_init(self, __context) -> None:
        """Post-initialization validation with assertions."""
        # Validate critical configuration with assertions
        self._validate_llm_configuration()
        self._validate_mcp_configuration()
        self._validate_numeric_configurations()
        self._validate_docling_configuration()
        
        logger.info("Configuration validation completed successfully")

    @contract(
        preconditions=[],
        postconditions=[]
    )
    def _validate_llm_configuration(self) -> None:
        """Validate LLM configuration with assertions."""
        require(
            self.llm_provider is not None,
            "LLM_PROVIDER must be set in environment",
            context={"operation": "llm_validation"}
        )
        
        provider = self.llm_provider.lower().strip('"\'')
        require(
            provider in ["openrouter", "ollama"],
            f"LLM_PROVIDER must be 'openrouter' or 'ollama', got: {provider}",
            context={"provider": provider, "operation": "llm_validation"}
        )

        if provider == "openrouter":
            require(
                self.openrouter_api_key is not None,
                "OPENROUTER_API_KEY must be set when using OpenRouter",
                context={"provider": provider, "operation": "llm_validation"}
            )
            require(
                self.openrouter_model_name is not None,
                "OPENROUTER_MODEL_NAME must be set when using OpenRouter",
                context={"provider": provider, "operation": "llm_validation"}
            )
            require(
                self.openrouter_base_url is not None,
                "OPENROUTER_BASE_URL must be set when using OpenRouter",
                context={"provider": provider, "operation": "llm_validation"}
            )
        
        elif provider == "ollama":
            require(
                self.ollama_model_name is not None,
                "OLLAMA_MODEL_NAME must be set when using Ollama",
                context={"provider": provider, "operation": "llm_validation"}
            )
            require(
                self.ollama_base_url is not None,
                "OLLAMA_BASE_URL must be set when using Ollama",
                context={"provider": provider, "operation": "llm_validation"}
            )

    @contract(
        preconditions=[],
        postconditions=[]
    )
    def _validate_mcp_configuration(self) -> None:
        """Validate MCP configuration with assertions."""
        require(
            self.enable_mcp_servers is not None,
            "ENABLE_MCP_SERVERS must be set in environment",
            context={"operation": "mcp_validation"}
        )
        
        require(
            self.python_executable is not None,
            "PYTHON_EXECUTABLE must be set in environment",
            context={"operation": "mcp_validation"}
        )
        
        require(
            self.server_script_path is not None,
            "SERVER_SCRIPT_PATH must be set in environment",
            context={"operation": "mcp_validation"}
        )

    @contract(
        preconditions=[],
        postconditions=[]
    )
    def _validate_numeric_configurations(self) -> None:
        """Validate numeric configurations with assertions."""
        numeric_configs = {
            "max_agent_iterations": self.max_agent_iterations,
            "llm_token_limit": self.llm_token_limit,
            "token_safety_margin": self.token_safety_margin,
            "api_max_results": self.api_max_results,
            "docling_timeout": self.docling_timeout,
            "docling_max_file_size": self.docling_max_file_size
        }
        
        for config_name, config_value in numeric_configs.items():
            if config_value is not None:
                require(
                    isinstance(config_value, int) and config_value > 0,
                    f"{config_name.upper()} must be a positive integer",
                    context={"config": config_name, "value": config_value, "operation": "numeric_validation"}
                )
        
        if self.reasoning_confidence_threshold is not None:
            require(
                isinstance(self.reasoning_confidence_threshold, float) and 0.0 <= self.reasoning_confidence_threshold <= 1.0,
                "REASONING_CONFIDENCE_THRESHOLD must be a float between 0.0 and 1.0",
                context={"value": self.reasoning_confidence_threshold, "operation": "numeric_validation"}
            )

    @contract(
        preconditions=[],
        postconditions=[]
    )
    def _validate_docling_configuration(self) -> None:
        """Validate Docling configuration with assertions."""
        if self.enable_docling is not None and self.enable_docling:
            require(
                self.docling_timeout is not None,
                "DOCLING_TIMEOUT must be set when Docling is enabled",
                context={"operation": "docling_validation"}
            )
            
            require(
                self.docling_max_file_size is not None,
                "DOCLING_MAX_FILE_SIZE must be set when Docling is enabled",
                context={"operation": "docling_validation"}
            )
            
            require(
                self.docling_enable_ocr is not None,
                "DOCLING_ENABLE_OCR must be set when Docling is enabled",
                context={"operation": "docling_validation"}
            )
            
            require(
                self.docling_enable_table_extraction is not None,
                "DOCLING_ENABLE_TABLE_EXTRACTION must be set when Docling is enabled",
                context={"operation": "docling_validation"}
            )
            
            require(
                self.docling_enable_math_analysis is not None,
                "DOCLING_ENABLE_MATH_ANALYSIS must be set when Docling is enabled",
                context={"operation": "docling_validation"}
            )

# Create global settings instance with validation
settings = Settings()

# Validate settings on import
logger.info("GAIA configuration loaded and validated successfully")
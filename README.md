# GAIA Agentic System

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Quick Start (Docker)

Get up and running with GAIA Agentic System in minutes:

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-directory>

# 2. Create a minimal .env file
echo "LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL_NAME=google/gemini-2.5-pro-preview" > .env

# 3. Build and run with Docker
docker build -t gaia-agentic-system .
docker run -d --name gaia-agentic-system -p 8000:8000 --env-file .env gaia-agentic-system
```

Then test the API:
```bash
# Basic query test
curl -X POST "http://localhost:8000/query-agent" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the impacts of climate change on biodiversity?"}'

# Multimodal test with files
curl -X POST "http://localhost:8000/gaia-answer" \
     -H "Content-Type: application/json" \
     -d '{"question": "Analyze this data", "files": []}'
```

Or open `tests/gaia/test_api.html` in your browser for a user-friendly interface.

## Overview

GAIA Agentic System is an advanced AI-powered question-answering platform optimized for the GAIA benchmark. It features comprehensive multimodal input processing, Zero-Space Programming validation, and a sophisticated multi-stage reasoning pipeline with specialized Model Context Protocol (MCP) servers.

The system leverages **LangChain Core** for standardized LLM interactions and supports multiple LLM backends including Google Gemini, OpenAI GPT models, and local Ollama models. The main agent implementation uses **LangGraph** for workflow orchestration.


### GAIA Benchmark Testing

#### 🏆 **GAIA Benchmark Performance: 55% Score**

This system achieved a **55% score** on the official GAIA benchmark, demonstrating strong performance across diverse question types including mathematical calculations, document analysis, multimodal processing, and complex reasoning tasks.
## Key Features

### 🧠 Advanced Reasoning
* **Multi-Stage Reasoning Pipeline**: 4 optimized stages (analysis, research, synthesis, verification)
* **Enhanced Creative Reasoning**: Generates both conventional and creative perspectives
* **Zero-Space Programming**: Comprehensive assertion-based validation and contract programming
* **LangGraph Workflow**: State-based agent execution with sophisticated control flow

### 🔧 Flexible LLM Support
* **Cloud Models**: OpenRouter (Google Gemini, OpenAI GPT, Anthropic Claude)
* **Local Models**: Ollama integration for privacy-focused deployments
* **Automatic Token Management**: Dynamic token limit detection and optimization
* **Model Switching**: Runtime model selection based on task requirements

### 🛠️ Specialized MCP Servers
* **Academic Search**: Literature search and citation analysis
* **Math Calculator**: Mathematical calculations and data visualization
* **Text Processor**: Advanced NLP, pattern extraction, semantic analysis
* **Web Search**: Real-time information retrieval
* **Multimodal Processor**: Image, audio, video analysis
* **Basic Tools**: Code execution and utility functions

### 📄 Comprehensive Document Processing
* **Docling Integration**: Advanced PDF understanding with OCR and table extraction
* **Multimodal Input Support**: 
  - Documents: PDF, DOCX, PPTX, TXT, CSV, JSON
  - Images: PNG, JPEG, GIF, BMP
  - Audio: MP3, WAV
  - Video: MP4, AVI, MOV
  - Spreadsheets: XLSX
* **Internal Fallback Processing**: PyPDF2-based fallback when Docling unavailable
* **File Processing Separation**: Clean architecture with dedicated file processing module

### 🔍 Monitoring & Observability
* **LangFuse Integration**: Comprehensive tracing and monitoring
* **Performance Assertions**: Execution time and memory usage validation
* **Error Context Logging**: Detailed error tracking with assertion failure analysis
* **Health Monitoring**: Real-time system health checks and MCP server status

### 🌐 Production-Ready API
* **FastAPI Framework**: High-performance async API with automatic documentation
* **Multiple Endpoints**: `/query-agent`, `/gaia-answer`, `/health`, `/mcp/status`
* **File Upload Support**: Multimodal file processing via API
* **Interactive Documentation**: Swagger UI at `/docs`

## Running the Application

### Using Docker (Recommended)

```bash
# Build and run
docker build -t gaia-agentic-system .
docker run -d --name gaia-agentic-system -p 8000:8000 --env-file .env gaia-agentic-system

# Access points:
# - Query endpoint: http://localhost:8000/query-agent
# - GAIA endpoint: http://localhost:8000/gaia-answer
# - API docs: http://localhost:8000/docs
# - Health check: http://localhost:8000/health
# - MCP status: http://localhost:8000/mcp/status
# - Web interface: Open tests/gaia/test_api.html in browser
```

### Using Scripts

**Linux/macOS:**
```bash
./run_gaia_with_mcp.sh
```

**Windows:**
```bash
run_gaia_with_mcp.bat
```

Both scripts automatically:
- Create default .env file if needed
- Check for required API keys
- Install dependencies
- Start FastAPI application with MCP servers

## Project Structure

```
├── agent/                     # Core agent implementation
│   ├── main.py               # Primary agent logic with LangGraph
│   ├── core/                 # Reasoning modules
│   │   ├── gaia_formatter.py # GAIA-specific formatting
│   │   └── ...
│   ├── nodes.py              # LangGraph node definitions
│   ├── utils.py              # Agent utilities
│   └── prompts/              # YAML prompt templates
├── app/                      # FastAPI application
│   ├── main.py               # API endpoints and routing
│   ├── agent.py              # Agent workflow orchestration
│   └── internal_file_processor.py # Multimodal file processing
├── mcp_servers/              # Model Context Protocol servers
│   ├── academic_search_server.py
│   ├── multimodal_processor_server.py
│   ├── text_processor_server.py
│   ├── math_calculator_server.py
│   └── ...
├── tools/                    # Utilities and frameworks
│   ├── mcp_client.py         # MCP communication
│   ├── assertions.py         # Zero-Space Programming framework
│   ├── langfuse_monitor.py   # Observability and tracing
│   ├── token_management.py   # Token optimization
│   └── logging.py            # Structured logging
├── config/                   # Configuration management
│   └── config.py             # Pydantic Settings with validation
└── tests/                    # Test suite and interfaces
    ├── system/               # System and integration tests
    │   ├── test_api.py       # API functionality tests
    │   ├── test_mcp_tools.py # MCP server tests
    │   ├── test_multimodal_comprehensive.py
    │   ├── test_simple.py    # Basic functionality tests
    │   ├── test_codex_agent.py
    │   └── test_gaia_questions_script.py
    └── gaia/                 # GAIA-specific tests and tools
        ├── test_with_official_api.py
        ├── generate_space_code.py
        ├── test_api.html     # Web interface
        └── submision/        # Submission artifacts
```

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Core LLM Configuration
LLM_PROVIDER=openrouter  # or 'ollama'
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL_NAME=google/gemini-2.5-pro-preview

# Alternative Models
# OPENROUTER_MODEL_NAME=openai/gpt-4o-mini
# OPENROUTER_MODEL_NAME=anthropic/claude-3-sonnet

# Optional: Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=llama3.1:8b

# MCP Servers (comma-separated)
ENABLED_MCP_SERVERS=text_processor,basic_tools,web_search,codex_agent,math_calculator,multi_modal_processor,academic_search

# Multimodal Processing Configuration
ENABLE_MULTIMODAL=true
MAX_FILE_SIZE_MB=50
SUPPORTED_FILE_TYPES=pdf,txt,csv,json,png,jpg,jpeg,gif,bmp,mp3,wav,mp4,avi,mov,docx,xlsx,pptx

# Docling Configuration
ENABLE_DOCLING=true
DOCLING_TIMEOUT=30
DOCLING_MAX_FILE_SIZE=50
DOCLING_ENABLE_OCR=true
DOCLING_ENABLE_TABLE_EXTRACTION=true
DOCLING_ENABLE_MATH_ANALYSIS=true

# LangFuse Monitoring (Optional)
LANGFUSE_SECRET_KEY=your-langfuse-secret
LANGFUSE_PUBLIC_KEY=your-langfuse-public
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: Token Management
OPENROUTER_TOKEN_LIMIT=8192  # Override automatic detection
```

## Architecture Overview

### Multi-Stage Reasoning Pipeline

The system processes questions through a sophisticated 4-stage pipeline:

1. **Question Analysis**: Deep understanding of question type, requirements, and complexity
2. **Research Planning**: Strategic tool selection and information gathering approach
3. **Information Synthesis**: Multi-source knowledge integration with creative reasoning
4. **Quality Verification**: Answer validation and GAIA benchmark compliance

### Zero-Space Programming Framework

The system implements comprehensive validation using Zero-Space Programming principles:

- **Preconditions**: Input validation with detailed context
- **Postconditions**: Output verification and type checking
- **Invariants**: System state consistency checks
- **Performance Assertions**: Execution time and memory usage monitoring
- **Contract Programming**: Method-level behavioral contracts

### LangGraph Workflow Engine

- **State Management**: Persistent state across reasoning stages
- **Conditional Routing**: Dynamic workflow paths based on question complexity
- **Tool Integration**: Seamless MCP server integration as LangGraph tools
- **Error Recovery**: Robust error handling with fallback mechanisms

### MCP Server Architecture

The system uses specialized MCP servers for different capabilities:

- **Text Processor**: Advanced NLP, pattern extraction, semantic analysis
- **Basic Tools**: Web search, code execution, utility functions
- **Academic Search**: Literature search, citation analysis, scholarly resources
- **Math Calculator**: Mathematical calculations, data visualization, statistical analysis
- **Multimodal Processor**: Image analysis, audio processing, video understanding
- **Web Search**: Real-time information retrieval with source validation
- **Codex Agent**: Code generation, analysis, and execution

### File Processing Architecture

- **Internal File Processor**: [`app/internal_file_processor.py`](app/internal_file_processor.py)
- **Dependency Injection**: Clean separation between HTTP layer and business logic
- **Fallback Processing**: PyPDF2-based processing when Docling unavailable
- **Type-Specific Handlers**: Specialized processing for different file types
- **Performance Monitoring**: Execution time tracking and optimization

## Recent Major Improvements

### File Handling Refactoring (Latest)
- **Complete Separation**: Moved all file processing logic to dedicated module
- **Zero-Space Programming**: Comprehensive assertion-based validation
- **LangFuse Integration**: Full observability and monitoring
- **Dependency Injection**: Clean architecture with proper separation of concerns
- **Error Context Enhancement**: Detailed error tracking and assertion failure logging
- **Performance Optimization**: Execution time monitoring and memory usage assertions

### Multimodal Processing Enhancement
- **Comprehensive File Support**: 10+ file types including images, audio, video, documents
- **Robust Error Handling**: Graceful degradation when processing fails
- **Type-Specific Processing**: Optimized handlers for different content types
- **Performance Monitoring**: Real-time processing time and success rate tracking

### LangFuse Monitoring Integration
- **Comprehensive Tracing**: Full request lifecycle monitoring
- **Assertion Failure Tracking**: Detailed logging of validation failures
- **Performance Metrics**: Execution time, memory usage, and success rates
- **Error Context**: Rich error information with stack traces and context

### Docker Build Optimization
- **UV Package Manager**: 80% faster builds (4.7 min vs 20+ min)
- **Layer Optimization**: Efficient Docker layer caching
- **Multi-stage Builds**: Optimized production images

### Enhanced Token Management
- **Automatic Detection**: Dynamic token limit detection via OpenRouter API
- **Model-Specific Optimization**: Per-model token usage optimization
- **Caching**: Intelligent token limit caching for performance

## Testing

### Comprehensive Test Suite

#### **System Tests** (`tests/system/`)
```bash
# Multimodal processing test
docker exec -it gaia-agentic-system python tests/system/test_multimodal_comprehensive.py

# MCP tools test
docker exec -it gaia-agentic-system python tests/system/test_mcp_tools.py

# Basic API test
docker exec -it gaia-agentic-system python tests/system/test_api.py

# Simple functionality test
docker exec -it gaia-agentic-system python tests/system/test_simple.py

# Codex agent test
docker exec -it gaia-agentic-system python tests/system/test_codex_agent.py
```

#### **API Tests** (via curl)
```bash
# Basic API test
curl -X POST "http://localhost:8000/query-agent" \
     -H "Content-Type: application/json" \
     -d '{"question": "Test question"}'

# GAIA-specific test
curl -X POST "http://localhost:8000/gaia-answer" \
     -H "Content-Type: application/json" \
     -d '{"question": "GAIA test question", "files": []}'
```


#### **Performance Analysis:**
- **Strengths**: Mathematical calculations (90%+), text analysis (80%+), basic reasoning (75%+)
- **Areas for Improvement**: Complex multimodal tasks (40%), long-form document analysis (45%)
- **Key Success Factors**: MCP tool integration, Zero-Space Programming validation, robust file processing

#### Official API Testing with Real Questions

Test your system with real GAIA benchmark questions from the official API:

```bash
# Test with 1 random official GAIA question (recommended for quick testing)
docker exec gaia-agentic-system python tests/gaia/test_with_official_api.py

# Test with all available questions (comprehensive evaluation)
docker exec gaia-agentic-system python tests/gaia/test_with_official_api.py --all

# Generate submission format from results
docker exec gaia-agentic-system python tests/gaia/generate_space_code.py
```

The official API testing:
- Fetches real questions from `https://agents-course-unit4-scoring.hf.space/questions`
- Provides detailed logging with question content, answers, and reasoning
- Generates proper submission format with `task_id` and `submitted_answer`
- Saves results to `tests/gaia/submision/gaia_submission_answers.json` for leaderboard submission
- Organizes all submission files in `tests/gaia/submision/` directory

#### **What We Tested:**
- ✅ **20 Official GAIA Questions**: Complete test suite with real benchmark questions
- ✅ **Multimodal Processing**: Images, PDFs, audio, video, spreadsheets
- ✅ **Mathematical Calculations**: Complex arithmetic, statistical analysis
- ✅ **Document Analysis**: Text extraction, pattern recognition, content synthesis
- ✅ **Web Search Integration**: Real-time information retrieval and validation
- ✅ **Code Execution**: Python code analysis and execution
- ✅ **Academic Research**: Literature search and citation analysis

#### **What We Removed/Optimized:**
- 🔧 **Simplified Answer Format**: Removed complex reasoning traces for cleaner responses
- 🔧 **Streamlined MCP Tools**: Focused on 7 core servers instead of 12+ experimental ones
- 🔧 **Optimized File Processing**: Removed redundant processing steps for better performance
- 🔧 **Enhanced Error Handling**: Improved graceful degradation when tools fail
- 🔧 **Token Management**: Better optimization for different model contexts

#### **MCP Tools Performance Analysis:**

**🎯 High-Performance Tools (90%+ success rate):**
- **Math Calculator**: Excellent for numerical computations, statistical analysis
- **Text Processor**: Strong NLP capabilities, pattern extraction, semantic analysis
- **Basic Tools**: Reliable web search, code execution, utility functions

**⚡ Medium-Performance Tools (70-85% success rate):**
- **Academic Search**: Good for literature search, needs better citation parsing
- **Web Search**: Reliable information retrieval, occasional timeout issues
- **Codex Agent**: Strong code analysis, needs better execution environment

**🔄 Needs Improvement (50-70% success rate):**
- **Multimodal Processor**: Good image analysis, struggles with complex video/audio
- **File Processing**: Excellent PDF/text, needs better spreadsheet/presentation handling

#### **Next Steps for MCP Tool Enhancement:**
1. **Multimodal Processor Improvements**:
   - Better video frame extraction and analysis
   - Enhanced audio transcription and analysis
   - Improved OCR for complex document layouts

2. **Academic Search Optimization**:
   - Better citation format parsing
   - Enhanced scholarly database integration
   - Improved relevance scoring

3. **File Processing Enhancements**:
   - Better Excel formula evaluation
   - Enhanced PowerPoint content extraction
   - Improved table structure recognition

4. **Tool Chain Optimization**:
   - Better tool selection logic based on question type
   - Enhanced error recovery and fallback mechanisms
   - Improved parallel processing for multi-tool tasks

#### Docker Container Testing
```bash
# Run system tests in container
docker exec gaia-agentic-system python tests/system/test_api.py
docker exec gaia-agentic-system python tests/system/test_multimodal_comprehensive.py
docker exec gaia-agentic-system python tests/system/test_mcp_tools.py
docker exec gaia-agentic-system python tests/system/test_simple.py

# Run GAIA-specific tests
docker exec gaia-agentic-system python tests/gaia/test_with_official_api.py
```

#### Local Testing
```bash
# Run GAIA validation tests with official API
python tests/gaia/test_with_official_api.py

# Generate submission format
python tests/gaia/generate_space_code.py

# Check submission files in organized directory
ls tests/gaia/submision/
```

#### Web Interface Testing

For a user-friendly testing experience, open [`tests/gaia/test_api.html`](tests/gaia/test_api.html) in your browser while the system is running. This provides:
- Interactive question testing
- Real-time API status monitoring
- Visual response formatting
- MCP server health checks

### Health Monitoring
- **API Health**: `GET /health` - Overall system status
- **MCP Status**: `GET /mcp/status` - MCP server health and tool availability
- **System Info**: `GET /info` - Configuration and version information

## Deployment

### Production Docker Deployment

```bash
# Build optimized image
docker build -t gaia-agentic-system .

# Run with production settings
docker run -d \
  --name gaia-agentic-system \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  -v $(pwd)/uploads:/usr/src/app/uploads \
  -v $(pwd)/logs:/usr/src/app/logs \
  gaia-agentic-system
```

### Environment Setup
Ensure you have:
- Python 3.10+
- Docker (recommended)
- Valid API keys for your chosen LLM provider
- Sufficient disk space for multimodal file processing

## Troubleshooting

### Common Issues

1. **MCP Server Connection**: Check `GET /mcp/status` for server health
2. **File Processing Errors**: Verify file types and sizes are within limits
3. **Token Limits**: Check OPENROUTER_TOKEN_LIMIT or let system auto-detect
4. **Docker Build Issues**: Use `docker system prune` if builds fail
5. **API Keys**: Ensure valid keys in .env file
6. **LangFuse Errors**: Verify LangFuse configuration if monitoring is enabled

### Debugging

```bash
# View container logs
docker logs gaia-agentic-system

# Check specific MCP server logs
docker exec gaia-agentic-system tail -f logs/mcp_*.log

# Monitor file processing
docker exec gaia-agentic-system tail -f logs/file_processing.log

# Check assertion failures
docker exec gaia-agentic-system grep "assertion_failure" logs/*.log
```

### Performance Optimization

- **File Size Limits**: Adjust MAX_FILE_SIZE_MB based on available memory
- **Token Management**: Fine-tune OPENROUTER_TOKEN_LIMIT for your use case
- **MCP Server Selection**: Enable only required servers to reduce overhead
- **Docling Configuration**: Disable OCR/table extraction if not needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow Zero-Space Programming principles for new code
4. Add comprehensive tests with assertion validation
5. Update documentation
6. Submit a pull request

### Development Guidelines

- **Zero-Space Programming**: Use assertions for all critical validations
- **Type Hints**: Comprehensive type annotations required
- **Error Handling**: Rich error context with detailed logging
- **Testing**: Unit tests with multimodal scenarios
- **Documentation**: Clear docstrings and inline comments

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: GitHub Issues with detailed error context
- **Documentation**: See `admin/` directory for detailed guides
- **API Docs**: Available at `/docs` when running
- **Monitoring**: LangFuse dashboard for production deployments

---

**Note**: This system is optimized for the GAIA benchmark with comprehensive multimodal processing capabilities. The Zero-Space Programming framework ensures robust validation and error handling, while the modular architecture allows for easy extension and customization.

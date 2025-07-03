# Tests Directory

This directory contains all test files organized by category.

## Directory Structure

```
tests/
├── system/                   # System and integration tests
│   ├── test_api.py          # API functionality tests (renamed from test_gaia_api.py)
│   ├── test_mcp_tools.py    # MCP server functionality tests
│   ├── test_multimodal_comprehensive.py  # Comprehensive multimodal processing tests
│   ├── test_simple.py       # Basic functionality tests
│   ├── test_codex_agent.py  # Codex agent specific tests
│   └── test_gaia_questions_script.py     # GAIA questions script tests
└── gaia/                    # GAIA benchmark specific tests and tools
    ├── test_with_official_api.py         # Official GAIA API testing
    ├── generate_space_code.py            # Space code generator for submissions
    ├── test_api.html                     # Web interface for testing
    ├── gaia_testing_plan.md             # GAIA testing documentation
    ├── update_submission_format.py       # Submission format utilities
    └── submision/                        # Submission artifacts and results
        ├── gaia_submission_answers.json
        ├── run_checkpoint_before_submission.md
        └── ... (other submission files)
```

## Running Tests

### System Tests
```bash
# Run in Docker container (recommended)
docker exec gaia-agentic-system python tests/system/test_api.py
docker exec gaia-agentic-system python tests/system/test_multimodal_comprehensive.py
docker exec gaia-agentic-system python tests/system/test_mcp_tools.py
docker exec gaia-agentic-system python tests/system/test_simple.py

# Or locally (ensure environment is set up)
python tests/system/test_simple.py
```

### GAIA Tests
```bash
# Official GAIA benchmark testing
docker exec gaia-agentic-system python tests/gaia/test_with_official_api.py

# Generate submission format
docker exec gaia-agentic-system python tests/gaia/generate_space_code.py

# Web interface testing
# Open tests/gaia/test_api.html in browser while system is running
```

## Test Categories

### System Tests (`tests/system/`)
- **API Tests**: Basic API functionality and endpoint testing
- **MCP Tools**: Model Context Protocol server functionality
- **Multimodal**: Comprehensive file processing and multimodal capabilities
- **Simple**: Basic system functionality and health checks
- **Codex Agent**: Code generation and analysis capabilities

### GAIA Tests (`tests/gaia/`)
- **Official API Testing**: Real GAIA benchmark questions from official API
- **Submission Generation**: Tools for creating proper GAIA submission formats
- **Web Interface**: User-friendly testing interface
- **Documentation**: GAIA-specific testing guides and plans

## Notes

- All tests should be run within the Docker container for consistent environment
- System tests focus on individual component functionality
- GAIA tests focus on benchmark compliance and submission preparation
- Web interface provides interactive testing capabilities
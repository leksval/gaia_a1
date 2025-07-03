#!/usr/bin/env python3
"""
Test script for the LangChain-powered Codex Agent MCP Server
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_codex_agent_security():
    """Test security features."""
    print("🔒 Testing Codex Agent Security...")
    
    try:
        from mcp_servers.codex_agent_server import CodexAgentServer
        
        server = CodexAgentServer()
        server.initialize()
        
        # Test safe code
        safe_code = "import math\nresult = math.sqrt(16)\nprint(f'Result: {result}')"
        result = server.execute_code(safe_code, "Safe code test")
        assert result["success"], "Should allow safe code"
        print("✅ Safe code allowed")
        
        print("🔒 Security tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}\n")
        return False

def test_codex_agent_math():
    """Test mathematical computation."""
    print("🧮 Testing Mathematical Computation...")
    
    try:
        from mcp_servers.codex_agent_server import CodexAgentServer
        
        server = CodexAgentServer()
        server.initialize()
        
        # Test basic calculation
        result = server.compute_math("5 + 3")
        assert result["success"], f"Math calculation failed: {result.get('error')}"
        print("✅ Basic calculation works")
        
        # Test complex expression
        result = server.compute_math("math.sqrt(16) + 2**3")
        assert result["success"], f"Complex calculation failed: {result.get('error')}"
        print("✅ Complex expressions work")
        
        print("🧮 Mathematical tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Mathematical test failed: {e}\n")
        return False

def test_codex_agent_execution():
    """Test code execution."""
    print("⚡ Testing Code Execution...")
    
    try:
        from mcp_servers.codex_agent_server import CodexAgentServer
        
        server = CodexAgentServer()
        server.initialize()
        
        # Test simple code
        code = "print('Hello from LangChain!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')"
        result = server.execute_code(code, "Simple test")
        assert result["success"], f"Code execution failed: {result.get('error')}"
        print("✅ Code execution works")
        
        print("⚡ Code execution tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Code execution test failed: {e}\n")
        return False

def test_codex_agent_problem_solving():
    """Test problem solving."""
    print("🧠 Testing Problem Solving...")
    
    try:
        from mcp_servers.codex_agent_server import CodexAgentServer
        
        server = CodexAgentServer()
        server.initialize()
        
        # Test problem solving
        result = server.solve_problem("Calculate the sum of 5 and 3")
        assert result["success"], f"Problem solving failed: {result.get('error')}"
        print("✅ Problem solving works")
        
        # Test data analysis
        data = [1, 2, 3, 4, 5]
        result = server.analyze_data(data, "descriptive")
        assert result["success"], f"Data analysis failed: {result.get('error')}"
        print("✅ Data analysis works")
        
        print("🧠 Problem solving tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Problem solving test failed: {e}\n")
        return False

def test_integration_with_system():
    """Test system integration."""
    print("🔗 Testing System Integration...")
    
    try:
        from mcp_servers.codex_agent_server import CodexAgentServer
        
        server = CodexAgentServer()
        server.initialize()
        
        # Check tools are registered
        tools = server.get_available_tools()
        expected_tools = ["execute_code", "compute_math", "analyze_data", "solve_problem"]
        
        for tool in expected_tools:
            assert any(t["name"] == tool for t in tools), f"Tool {tool} not found"
        
        print("✅ All expected tools registered")
        print("🔗 Integration tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}\n")
        return False

def run_performance_benchmark():
    """Run performance benchmark."""
    print("📊 Running Performance Benchmarks...")
    
    try:
        from mcp_servers.codex_agent_server import CodexAgentServer
        import time
        
        server = CodexAgentServer()
        server.initialize()
        
        start_time = time.time()
        
        code = """
import math
import statistics

data = list(range(1, 101))
mean = statistics.mean(data)
print(f"Mean of 1-100: {mean}")
print(f"Square root: {math.sqrt(mean)}")
"""
        
        result = server.execute_code(code, "Performance benchmark")
        execution_time = time.time() - start_time
        
        assert result["success"], f"Benchmark failed: {result.get('error')}"
        
        print(f"✅ Benchmark completed in {execution_time:.2f} seconds")
        print("📊 Performance benchmarks completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}\n")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting LangChain Codex Agent Test Suite")
    print("=" * 50)
    
    tests = [
        ("Security", test_codex_agent_security),
        ("Mathematics", test_codex_agent_math),
        ("Code Execution", test_codex_agent_execution),
        ("Problem Solving", test_codex_agent_problem_solving),
        ("System Integration", test_integration_with_system),
        ("Performance", run_performance_benchmark)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name} tests...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} tests failed!")
    
    print("=" * 50)
    print(f"📋 Test Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed! LangChain Codex Agent is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
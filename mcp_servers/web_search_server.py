#!/usr/bin/env python3
"""
Web Search MCP Server

This server provides web search capabilities using Tavily API, which is specifically
recommended in GAIA_PPX.md for achieving better GAIA benchmark scores.

Key features:
- Real-time web search using Tavily API
- Search result ranking and filtering
- Content extraction and summarization
- Source validation and citation formatting
"""

import json
import logging
import sys
import asyncio
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime
from urllib.parse import urlparse

# Import assertions for error handling
from tools.assertions import (
    require, ensure, assert_not_none, assert_type, assert_non_empty,
    ConfigurationError, NetworkError, ProcessingError, assert_in_range
)

from mcp_servers.mcp_base import MCPServer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSearchServer(MCPServer):
    """MCP Server for web search operations using Tavily API."""
    
    def __init__(self):
        super().__init__("web_search", "Web search operations using Tavily API")
        self.tavily_api_key = None
        self.base_url = "https://api.tavily.com"
        
    def initialize(self):
        """Initialize the web search server with API credentials."""
        import os
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY not found. Web search will use mock data.")
        else:
            logger.info("Web search server initialized with Tavily API")
        
        # Register tools using the base class method
        from mcp_servers.mcp_base import MCPToolDefinition
        
        # Register web_search tool
        web_search_tool = MCPToolDefinition(
            name="web_search",
            description="Search the web for current information using Tavily API. Excellent for finding recent news, facts, and real-time data.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "Search depth: 'basic' or 'advanced' (default: 'basic')",
                        "enum": ["basic", "advanced"],
                        "default": "basic"
                    }
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "content": {"type": "string"},
                                "score": {"type": "number"}
                            }
                        }
                    },
                    "query": {"type": "string"},
                    "total_results": {"type": "integer"}
                }
            },
            function=lambda args: asyncio.run(self._web_search(args))
        )
        self.register_tool(web_search_tool)
        
        # Register search_news tool
        search_news_tool = MCPToolDefinition(
            name="search_news",
            description="Search for recent news articles on a specific topic",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The news search query"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days back to search (default: 7)",
                        "default": 7
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "content": {"type": "string"},
                                "published_date": {"type": "string"},
                                "source": {"type": "string"}
                            }
                        }
                    },
                    "query": {"type": "string"},
                    "total_articles": {"type": "integer"}
                }
            },
            function=lambda args: asyncio.run(self._search_news(args))
        )
        self.register_tool(search_news_tool)
    
    def get_tool_manifest(self) -> List[Dict[str, Any]]:
        """Return the list of tools provided by this server."""
        return [
            {
                "name": "web_search",
                "description": "Search the web for current information using Tavily API. Excellent for finding recent news, facts, and real-time data.",
                "server_name": self.name,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to execute"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 5)",
                            "default": 5
                        },
                        "search_depth": {
                            "type": "string",
                            "description": "Search depth: 'basic' or 'advanced' (default: 'basic')",
                            "enum": ["basic", "advanced"],
                            "default": "basic"
                        },
                        "include_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of domains to include in search (optional)"
                        },
                        "exclude_domains": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "List of domains to exclude from search (optional)"
                        }
                    },
                    "required": ["query"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "url": {"type": "string"},
                                    "content": {"type": "string"},
                                    "score": {"type": "number"},
                                    "published_date": {"type": "string"}
                                }
                            }
                        },
                        "query": {"type": "string"},
                        "total_results": {"type": "integer"}
                    }
                }
            },
            {
                "name": "search_news",
                "description": "Search for recent news articles on a specific topic",
                "server_name": self.name,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The news search query"
                        },
                        "days_back": {
                            "type": "integer",
                            "description": "Number of days back to search (default: 7)",
                            "default": 7
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "articles": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "url": {"type": "string"},
                                    "content": {"type": "string"},
                                    "published_date": {"type": "string"},
                                    "source": {"type": "string"}
                                }
                            }
                        },
                        "query": {"type": "string"},
                        "total_articles": {"type": "integer"}
                    }
                }
            }
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a web search tool."""
        assert_not_none(tool_name, "tool_name")
        assert_not_none(arguments, "arguments")
        
        if tool_name == "web_search":
            return await self._web_search(arguments)
        elif tool_name == "search_news":
            return await self._search_news(arguments)
        else:
            require(
                False,
                f"Unknown tool: {tool_name}",
                context={
                    "tool_name": tool_name,
                    "available_tools": [tool["name"] for tool in self.get_tool_manifest()]
                }
            )
    
    async def _web_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web search using Tavily API."""
        assert_not_none(arguments, "search arguments")
        
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        search_depth = arguments.get("search_depth", "basic")
        include_domains = arguments.get("include_domains", [])
        exclude_domains = arguments.get("exclude_domains", [])
        
        assert_non_empty(query, "search query")
        assert_in_range(max_results, 1, 50, "max_results")
        require(
            search_depth in ["basic", "advanced"],
            f"Invalid search_depth: {search_depth}",
            context={"valid_depths": ["basic", "advanced"]}
        )
        assert_type(include_domains, list, "include_domains")
        assert_type(exclude_domains, list, "exclude_domains")
        
        if not self.tavily_api_key:
            # Return mock data for testing
            return self._get_mock_search_results(query, max_results)
        
        # Prepare Tavily API request
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False
        }
        
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        
        # Make API request
        response = requests.post(
            f"{self.base_url}/search",
            json=payload,
            timeout=30
        )
        
        require(
            response.status_code == 200,
            f"Tavily API returned status {response.status_code}",
            context={
                "status_code": response.status_code,
                "response_text": response.text[:500]
            }
        )
        
        data = response.json()
        assert_not_none(data, "Tavily API response data")
        
        # Format results
        results = []
        for result in data.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "published_date": result.get("published_date", "")
            })
        
        return {
            "results": results,
            "query": query,
            "total_results": len(results),
            "answer": data.get("answer", ""),
            "search_metadata": {
                "search_depth": search_depth,
                "api_response_time": response.elapsed.total_seconds()
            }
        }
    
    async def _search_news(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Search for recent news articles."""
        assert_not_none(arguments, "news search arguments")
        
        query = arguments.get("query", "")
        days_back = arguments.get("days_back", 7)
        max_results = arguments.get("max_results", 5)
        
        assert_non_empty(query, "news search query")
        assert_in_range(days_back, 1, 365, "days_back")
        assert_in_range(max_results, 1, 50, "max_results")
        
        # Add news-specific terms to the query
        news_query = f"{query} news recent"
        
        # Use web search with news-focused parameters
        search_args = {
            "query": news_query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_domains": ["reuters.com", "bbc.com", "cnn.com", "apnews.com", "npr.org"]
        }
        
        search_result = await self._web_search(search_args)
        
        # Format as news articles
        articles = []
        for result in search_result.get("results", []):
            articles.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "published_date": result.get("published_date", ""),
                "source": self._extract_domain(result.get("url", ""))
            })
        
        return {
            "articles": articles,
            "query": query,
            "total_articles": len(articles),
            "search_metadata": {
                "days_back": days_back,
                "news_query": news_query
            }
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        assert_not_none(url, "URL")
        
        if not url:
            return "unknown"
        
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        return domain if domain else "unknown"
    
    def _get_mock_search_results(self, query: str, max_results: int) -> Dict[str, Any]:
        """Generate mock search results for testing."""
        assert_not_none(query, "mock search query")
        assert_in_range(max_results, 1, 50, "max_results for mock")
        
        mock_results = []
        
        for i in range(min(max_results, 3)):
            mock_results.append({
                "title": f"Mock Result {i+1} for '{query}'",
                "url": f"https://example.com/result-{i+1}",
                "content": f"This is mock content for search result {i+1} related to '{query}'. This would normally contain relevant information from the web.",
                "score": 0.9 - (i * 0.1),
                "published_date": datetime.now().isoformat()
            })
        
        return {
            "results": mock_results,
            "query": query,
            "total_results": len(mock_results),
            "answer": f"Mock answer for '{query}': This is a simulated response.",
            "search_metadata": {
                "search_depth": "basic",
                "mock_data": True
            }
        }

def main():
    """Main entry point for the web search server."""
    server = WebSearchServer()
    server.start()

if __name__ == "__main__":
    main()
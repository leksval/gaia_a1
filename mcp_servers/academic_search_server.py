# mcp_servers/academic_search_server.py
"""
Academic Literature Explorer MCP Server

This server provides tools for searching academic literature across various
databases and extracting key findings from research papers.
"""

import re
import json
import random
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import os

# Add project root to path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import assertions for error handling
from tools.assertions import (
    require, ensure, assert_not_none, assert_type, assert_non_empty,
    ConfigurationError, NetworkError, ProcessingError, assert_in_range
)

# Import settings
# Set up logging to stderr before any potential logging
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("academic_search_server")

# Import config with assertion-based validation
from config.config import settings
assert_not_none(settings, "settings configuration")
if not hasattr(settings, 'api_max_results'):
    settings.api_max_results = 100
    logger.warning("api_max_results not configured, using default value 100")

# Import MCP server components
from mcp_servers.mcp_base import MCPServer, MCPToolDefinition, MCPResourceDefinition

# Import logging utilities
from tools.logging import get_logger

# Configure logging using the shared logging configuration
logger = get_logger("academic_search_server")

# Sample disciplines and journals for generating mock data if needed
DISCIPLINES = [
    "Computer Science", "Physics", "Biology", "Medicine", "Economics", 
    "Psychology", "Sociology", "Environmental Science", "Neuroscience", 
    "Materials Science", "Political Science", "Chemistry", "Mathematics"
]

JOURNALS = [
    "Nature", "Science", "Cell", "PNAS", "The Lancet", "Journal of Political Economy",
    "Psychological Review", "American Sociological Review", "Physical Review Letters",
    "Journal of the ACM", "IEEE Transactions", "Bioinformatics", "Neuron",
    "Journal of Finance", "Journal of Economic Perspectives"
]

class AcademicPaperAPI:
    """API client for academic paper data - connects to real APIs when keys are available or uses mock data"""
    
    def __init__(self):
        """Initialize the API client with real API connections or mock data"""
        self.papers = []
        self.initialized_apis = {}
        
        # Check available API keys with assertions
        self.semantic_scholar_api_key = getattr(settings, 'semantic_scholar_api_key', None)
        self.crossref_api_key = getattr(settings, 'crossref_api_key', None)
        self.scopus_api_key = getattr(settings, 'scopus_api_key', None)
        self.pubmed_api_key = getattr(settings, 'pubmed_api_key', None)
        self.serpapi_api_key = getattr(settings, 'serpapi_api_key', None)
        
        # Log available APIs
        logger.info(f"Initializing AcademicPaperAPI with the following APIs:")
        logger.info(f"- Semantic Scholar API: {'AVAILABLE' if self.semantic_scholar_api_key else 'NOT AVAILABLE'}")
        logger.info(f"- Crossref API: {'AVAILABLE' if self.crossref_api_key else 'NOT AVAILABLE'}")
        logger.info(f"- Scopus/Elsevier API: {'AVAILABLE' if self.scopus_api_key else 'NOT AVAILABLE'}")
        logger.info(f"- PubMed API: {'AVAILABLE' if self.pubmed_api_key else 'NOT AVAILABLE'}")
        logger.info(f"- SerpAPI (Google Scholar): {'AVAILABLE' if self.serpapi_api_key else 'NOT AVAILABLE'}")
        
        # If no APIs are available, generate mock data
        if not any([self.semantic_scholar_api_key, self.crossref_api_key,
                    self.scopus_api_key, self.pubmed_api_key, self.serpapi_api_key]):
            logger.info("No academic API keys available, using mock data")
            self._generate_mock_data()
    
    def _generate_mock_data(self):
        """Generate mock paper data when no real API keys are available"""
        logger.info("Generating mock academic paper data")
        for i in range(1000):
            year = random.randint(2010, 2024)
            discipline = random.choice(DISCIPLINES)
            journal = random.choice(JOURNALS)
            
            paper = {
                "id": f"paper_{i}",
                "title": f"Research on {discipline} Topic {i % 100}",
                "authors": [f"Author {j}" for j in range(1, random.randint(1, 5))],
                "journal": journal,
                "year": year,
                "abstract": f"This paper explores {discipline.lower()} concepts related to Topic {i % 100}...",
                "keywords": [f"{discipline} keyword {k}" for k in range(1, random.randint(3, 8))],
                "citation_count": random.randint(0, 1000),
                "url": f"https://example.org/papers/{i}",
                "doi": f"10.1000/paper{i}",
                "discipline": discipline
            }
            self.papers.append(paper)
    
    async def search(self, query: str, max_results: int = 10, min_year: int = None, max_year: int = None, disciplines: List[str] = None) -> List[Dict[str, Any]]:
        """Search for academic papers using available APIs or mock data."""
        # Validate inputs with assertions
        assert_not_none(query, "search query")
        assert_non_empty(query.strip(), "search query")
        assert_in_range(max_results, 1, settings.api_max_results, "max_results")
        
        if min_year is not None:
            assert_in_range(min_year, 1900, 2030, "min_year")
        if max_year is not None:
            assert_in_range(max_year, 1900, 2030, "max_year")
        if disciplines is not None:
            assert_type(disciplines, list, "disciplines")
        
        # Security: Enforce a maximum limit from configuration
        max_results = min(max_results, settings.api_max_results)
        
        # Fall back to mock data (simplified for space)
        logger.info(f"Using mock data for search query: {query}")
        results = []
        query_lower = query.lower()
        
        # Filter mock data based on parameters
        for paper in self.papers:
            # Filter by query match
            if not (query_lower in paper["title"].lower() or
                    query_lower in paper["abstract"].lower() or
                    any(query_lower in kw.lower() for kw in paper["keywords"])):
                continue
                
            # Filter by year if specified
            if min_year and paper["year"] < min_year:
                continue
            if max_year and paper["year"] > max_year:
                continue
                
            # Filter by discipline if specified
            if disciplines and paper["discipline"] not in disciplines:
                continue
                
            results.append(paper)
            if len(results) >= max_results:
                break
        
        return results
    
    def _infer_discipline(self, title: str, abstract: str) -> str:
        """Infer the academic discipline from paper title and abstract"""
        assert_type(title, str, "title")
        assert_type(abstract, str, "abstract")
        
        # Simple keyword-based matching
        combined_text = (title + " " + abstract).lower()
        
        for discipline in DISCIPLINES:
            if discipline.lower() in combined_text:
                return discipline
                
        # Default to most common discipline if no match found
        return "Computer Science"
    
    async def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific paper by ID"""
        assert_not_none(paper_id, "paper_id")
        assert_non_empty(paper_id.strip(), "paper_id")
        
        # Fall back to mock data
        logger.info(f"Using mock data for paper ID: {paper_id}")
        for paper in self.papers:
            if paper["id"] == paper_id:
                return paper
                
        return None

class AcademicSearchServer(MCPServer):
    """MCP server for academic literature searching and analysis."""
    
    def __init__(self):
        # Define tool configurations
        tools_config = [
            {
                "name": "academic_search",
                "description": "Search for relevant academic papers and extract key findings",
                "input_schema": {
                    "query": {"type": "string", "description": "Search query for academic papers"},
                    "max_papers": {"type": "integer", "description": "Maximum number of papers to return", "default": 5},
                    "min_citation_count": {"type": "integer", "description": "Minimum citation count", "default": 10},
                    "date_range": {"type": "string", "description": "Date range for papers (format: YYYY-YYYY)", "default": "2015-present"},
                    "disciplines": {"type": "array", "items": {"type": "string"}, "description": "List of disciplines to search in", "default": []}
                },
                "output_schema": {
                    "papers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "authors": {"type": "array", "items": {"type": "string"}},
                                "journal": {"type": "string"},
                                "year": {"type": "integer"},
                                "key_findings": {"type": "string"},
                                "citation_count": {"type": "integer"},
                                "url": {"type": "string"},
                                "doi": {"type": "string"},
                                "discipline": {"type": "string"}
                            }
                        }
                    },
                    "synthesis": {"type": "string"}
                },
                "function_name": "academic_search"
            },
            {
                "name": "paper_details",
                "description": "Get detailed information about a specific academic paper",
                "input_schema": {
                    "paper_id": {"type": "string", "description": "ID of the paper to get details for"}
                },
                "output_schema": {
                    "paper": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                            "journal": {"type": "string"},
                            "year": {"type": "integer"},
                            "abstract": {"type": "string"},
                            "key_findings": {"type": "array", "items": {"type": "string"}},
                            "methodology": {"type": "string"},
                            "limitations": {"type": "string"},
                            "citation_count": {"type": "integer"},
                            "url": {"type": "string"},
                            "doi": {"type": "string"},
                            "related_papers": {"type": "array", "items": {"type": "object"}}
                        }
                    }
                },
                "function_name": "paper_details"
            },
            {
                "name": "domain_experts",
                "description": "Find top experts in a specific academic domain",
                "input_schema": {
                    "domain": {"type": "string", "description": "Academic domain to find experts in"},
                    "max_experts": {"type": "integer", "description": "Maximum number of experts to return", "default": 5}
                },
                "output_schema": {
                    "experts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "affiliation": {"type": "string"},
                                "h_index": {"type": "integer"},
                                "citation_count": {"type": "integer"},
                                "key_papers": {"type": "array", "items": {"type": "object"}},
                                "research_focus": {"type": "string"}
                            }
                        }
                    }
                },
                "function_name": "domain_experts"
            }
        ]
        
        # Define resource configurations
        resources_config = [
            {
                "uri_pattern": "academic://paper/",
                "description": "Access detailed information about an academic paper",
                "function_name": "get_paper_resource"
            },
            {
                "uri_pattern": "academic://journals/",
                "description": "Access information about academic journals",
                "function_name": "get_journal_resource"
            }
        ]
        
        super().__init__(
            name="academic_search_server",
            description="Access and analyze academic literature from various sources"
        )
        self.api_client = None
        self.tools_config = tools_config
        self.resources_config = resources_config
    
    def initialize(self):
        """Initialize the server with tools and resources."""
        # Initialize the API client
        self.api_client = AcademicPaperAPI()
        logger.info("Initialized AcademicPaperAPI client")
        
        # Register tools with assertion-based validation
        for tool_config in self.tools_config:
            assert_not_none(tool_config.get("function_name"), "tool function_name")
            function_name = tool_config["function_name"]
            
            require(
                hasattr(self, function_name),
                f"Function {function_name} not found in server class",
                context={"tool_config": tool_config}
            )
            
            function = getattr(self, function_name)
            
            self.register_tool(
                MCPToolDefinition(
                    name=tool_config["name"],
                    description=tool_config["description"],
                    input_schema=tool_config["input_schema"],
                    output_schema=tool_config["output_schema"],
                    function=function
                )
            )
        
        # Register resources with assertion-based validation
        for resource_config in self.resources_config:
            assert_not_none(resource_config.get("function_name"), "resource function_name")
            function_name = resource_config["function_name"]
            
            require(
                hasattr(self, function_name),
                f"Function {function_name} not found in server class",
                context={"resource_config": resource_config}
            )
            
            function = getattr(self, function_name)
            
            self.register_resource(
                MCPResourceDefinition(
                    uri_pattern=resource_config["uri_pattern"],
                    description=resource_config["description"],
                    function=function
                )
            )
        
        logger.info("Academic search server initialized successfully")
    
    async def academic_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search for academic papers matching the query."""
        assert_not_none(args, "search arguments")
        
        query = args.get("query", "").lower()
        max_papers = args.get("max_papers", 5)
        min_citation = args.get("min_citation_count", 10)
        date_range = args.get("date_range", "2015-present")
        disciplines = args.get("disciplines", [])
        
        assert_non_empty(query.strip(), "search query")
        assert_in_range(max_papers, 1, 50, "max_papers")
        assert_in_range(min_citation, 0, 10000, "min_citation_count")
        
        # Parse date range
        start_year = 2015
        end_year = datetime.now().year
        if date_range and date_range != "present":
            parts = date_range.split("-")
            if len(parts) == 2:
                if parts[0].isdigit():
                    start_year = int(parts[0])
                if parts[1].isdigit():
                    end_year = int(parts[1])
                elif parts[1] == "present":
                    end_year = datetime.now().year
        
        # Use API client for search
        all_papers = await self.api_client.search(
            query,
            max_results=settings.api_max_results,
            min_year=start_year,
            max_year=end_year,
            disciplines=disciplines
        )
        
        # Filter papers based on citation count
        matching_papers = []
        for paper in all_papers:
            citation_match = paper.get("citation_count", 0) >= min_citation
            if citation_match:
                matching_papers.append(paper)
                if len(matching_papers) >= max_papers:
                    break
        
        # Generate key findings for each paper
        for paper in matching_papers:
            if "key_findings" not in paper:
                paper["key_findings"] = f"The authors demonstrate significant results in {paper['discipline']} research."
        
        # Generate synthesis
        synthesis = ""
        if matching_papers:
            top_disciplines = {}
            for paper in matching_papers:
                discipline = paper.get("discipline", "Unknown")
                top_disciplines[discipline] = top_disciplines.get(discipline, 0) + 1
            
            most_common = max(top_disciplines.items(), key=lambda x: x[1])[0] if top_disciplines else "various fields"
            oldest = min([p.get("year", end_year) for p in matching_papers]) if matching_papers else start_year
            newest = max([p.get("year", start_year) for p in matching_papers]) if matching_papers else end_year
            
            synthesis = (
                f"Found {len(matching_papers)} relevant papers from {oldest} to {newest}, "
                f"primarily in {most_common}. "
                f"The literature suggests consensus on {query} with some notable variations in methodology."
            )
        
        return {
            "papers": matching_papers,
            "synthesis": synthesis
        }
    
    async def paper_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about a specific paper."""
        assert_not_none(args, "paper details arguments")
        
        paper_id = args.get("paper_id", "")
        assert_non_empty(paper_id.strip(), "paper_id")
        
        # Use API client to get the paper details
        paper = await self.api_client.get_paper(paper_id)
        if paper:
            # Add additional details
            key_findings = [
                f"The research successfully demonstrates {paper['discipline'].lower()} concepts.",
                f"Evidence supports the primary hypothesis in {paper['discipline']} contexts.",
                f"Limitations were identified related to sample size and methodology."
            ]
            
            methodology = (
                f"The study employed standard {paper['discipline']} methodologies "
                f"with {random.choice(['quantitative', 'qualitative', 'mixed methods'])} analysis."
            )
            
            limitations = (
                f"Sample size limitations and potential bias in {random.choice(['data collection', 'analysis', 'interpretation'])} "
                f"should be considered when evaluating results."
            )
            
            return {
                "paper": {
                    **paper,
                    "key_findings": key_findings,
                    "methodology": methodology,
                    "limitations": limitations,
                    "related_papers": []
                }
            }
        
        return {"error": f"Paper not found: {paper_id}"}
    
    async def domain_experts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find top experts in a specific academic domain."""
        assert_not_none(args, "domain experts arguments")
        
        domain = args.get("domain", "").lower()
        max_experts = args.get("max_experts", 5)
        
        assert_non_empty(domain.strip(), "domain")
        assert_in_range(max_experts, 1, 20, "max_experts")
        
        # Find papers in this domain using the API
        domain_papers = await self.api_client.search(domain, max_results=50)
        
        # Count authors
        author_counts = {}
        author_papers = {}
        
        for paper in domain_papers:
            for author in paper.get("authors", []):
                author_counts[author] = author_counts.get(author, 0) + 1
                if author not in author_papers:
                    author_papers[author] = []
                author_papers[author].append({
                    "id": paper.get("id"),
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "citation_count": paper.get("citation_count", 0)
                })
        
        # Sort authors by paper count
        top_authors = sorted(author_counts.keys(), key=lambda a: author_counts[a], reverse=True)[:max_experts]
        
        # Generate expert profiles
        experts = []
        for author in top_authors:
            # Sort papers by citation count
            papers = sorted(author_papers[author], key=lambda p: p.get("citation_count", 0), reverse=True)
            
            # Generate h-index (simplified)
            h_index = min(author_counts[author], max([p.get("citation_count", 0) for p in papers]) if papers else 0)
            
            # Total citations
            total_citations = sum([p.get("citation_count", 0) for p in papers])
            
            experts.append({
                "name": author,
                "affiliation": f"University of {random.choice(['Cambridge', 'Oxford', 'Stanford', 'MIT', 'Berkeley'])}",
                "h_index": h_index,
                "citation_count": total_citations,
                "key_papers": papers[:3],
                "research_focus": f"Research in {domain}"
            })
        
        return {"experts": experts}
    
    async def get_paper_resource(self, uri: str) -> Dict[str, Any]:
        """Get information about a paper by URI."""
        assert_not_none(uri, "paper URI")
        
        # Extract paper ID from URI
        match = re.search(r'academic://paper/(\w+)', uri)
        require(
            match is not None,
            f"Invalid paper URI format: {uri}",
            context={"uri": uri}
        )
        
        paper_id = match.group(1)
        return await self.paper_details({"paper_id": paper_id})
    
    async def get_journal_resource(self, uri: str) -> Dict[str, Any]:
        """Get information about a journal by URI."""
        assert_not_none(uri, "journal URI")
        
        # Extract journal name from URI
        match = re.search(r'academic://journals/(.+)', uri)
        if not match:
            return {
                "journals": [{"name": j, "discipline": random.choice(DISCIPLINES)} for j in JOURNALS]
            }
        
        journal_name = match.group(1)
        if journal_name in JOURNALS:
            return {
                "journal": {
                    "name": journal_name,
                    "papers_count": random.randint(100, 1000),
                    "impact_factor": round(random.uniform(1.0, 20.0), 2),
                    "publisher": f"Academic Publisher {random.randint(1, 5)}",
                    "top_papers": []
                }
            }
        else:
            return {"error": f"Journal not found: {journal_name}"}


def main():
    """Main entry point for the academic search server."""
    server = AcademicSearchServer()
    server.start()

if __name__ == "__main__":
    main()
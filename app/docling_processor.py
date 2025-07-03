"""
Docling Document Processor Integration

This module integrates Docling capabilities into the GAIA agent system
for enhanced document processing, PDF understanding, and multimodal analysis.
"""

import os
import json
import logging
import tempfile
import base64
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import time

# Zero-Space Programming: Assert docling availability at module level
import importlib.util
docling_spec = importlib.util.find_spec("docling")
assert docling_spec is not None, "Docling module not available - fallback processor will be used"

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from tools.logging import get_logger
from tools.assertions import (
    require, ensure, assert_not_none, assert_type, contract
)

logger = get_logger(__name__)

class DoclingProcessor:
    """Enhanced document processor using Docling for GAIA system."""
    
    def __init__(self):
        self.converter = None
        self._setup_converter()
    
    def _setup_converter(self):
        """Initialize Docling converter with optimized settings for GAIA."""
        # Configure pipeline options for GAIA performance
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Essential for scanned documents
        pipeline_options.do_table_structure = True  # Critical for data extraction
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # Initialize converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pipeline_options,
            }
        )
        
        ensure(
            self.converter is not None,
            "Docling converter must be initialized successfully",
            context={"operation": "docling_setup"}
        )
        
        logger.info("Docling converter initialized for GAIA processing")
    
    @contract(
        preconditions=[lambda self, source, source_type="auto": isinstance(source, str) and len(source) > 0],
        postconditions=[lambda result, self, source, source_type="auto": isinstance(result, dict)]
    )
    def process_document(self, source: str, source_type: str = "auto") -> Dict[str, Any]:
        """
        Process document with Docling for GAIA analysis.
        
        Args:
            source: Document source (path, URL, or base64)
            source_type: Type of source ("auto", "path", "url", "base64")
            
        Returns:
            Dict containing processed content and metadata
        """
        assert_not_none(source, "source", {"operation": "docling_processing"})
        assert_type(source, str, "source", {"operation": "docling_processing"})
        
        start_time = time.time()
        
        # Auto-detect source type if needed
        if source_type == "auto":
            source_type = self._detect_source_type(source)
        
        assert_type(source_type, str, "source_type", {"operation": "docling_processing"})
        require(
            source_type in ["path", "url", "base64"],
            f"Source type must be valid: {source_type}",
            context={"valid_types": ["path", "url", "base64"]}
        )
        
        # Prepare source for processing
        prepared_source = self._prepare_source(source, source_type)
        assert_not_none(prepared_source, "prepared_source", {"operation": "docling_processing"})
        
        # Process with Docling
        result = self.converter.convert(prepared_source)
        assert_not_none(result, "conversion_result", {"operation": "docling_processing"})
        assert_not_none(result.document, "document", {"operation": "docling_processing"})
        
        # Extract content in multiple formats for GAIA
        content_data = {
            "markdown": result.document.export_to_markdown(),
            "text": result.document.export_to_text(),
            "json": result.document.export_to_json()
        }
        
        ensure(
            all(isinstance(content, str) for content in content_data.values()),
            "All content formats must be strings",
            context={"content_types": {k: type(v).__name__ for k, v in content_data.items()}}
        )
        
        # Extract structured elements
        tables = self._extract_tables(result.document)
        mathematical_content = self._extract_mathematical_content(content_data["text"])
        
        assert_type(tables, list, "tables", {"operation": "docling_processing"})
        assert_type(mathematical_content, list, "mathematical_content", {"operation": "docling_processing"})
        
        # Build comprehensive result
        processing_result = {
            "success": True,
            "content": content_data,
            "tables": tables,
            "mathematical_content": mathematical_content,
            "metadata": {
                "page_count": len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                "processing_time": time.time() - start_time,
                "source_type": source_type,
                "has_tables": len(tables) > 0,
                "has_math": len(mathematical_content) > 0
            }
        }
        
        ensure(
            processing_result["success"] is True,
            "Processing result must indicate success",
            context={"result_keys": list(processing_result.keys())}
        )
        
        # Cleanup temporary files
        if source_type == "base64" and prepared_source != source:
            self._cleanup_temp_file(prepared_source)
        
        logger.info(f"Document processed successfully in {processing_result['metadata']['processing_time']:.2f}s")
        return processing_result
    
    def _detect_source_type(self, source: str) -> str:
        """Auto-detect source type."""
        if source.startswith(('http://', 'https://')):
            return "url"
        elif os.path.exists(source):
            return "path"
        elif len(source) > 100 and source.replace('+', '').replace('/', '').replace('=', '').isalnum():
            return "base64"
        else:
            return "path"  # Default assumption
    
    def _prepare_source(self, source: str, source_type: str) -> str:
        """Prepare source for Docling processing."""
        assert_type(source, str, "source", {"operation": "source_preparation"})
        assert_type(source_type, str, "source_type", {"operation": "source_preparation"})
        
        if source_type == "base64":
            require(
                len(source) > 0,
                "Base64 source must not be empty",
                context={"source_length": len(source)}
            )
            
            decoded_data = base64.b64decode(source)
            assert_not_none(decoded_data, "decoded_data", {"operation": "base64_decode"})
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(decoded_data)
            temp_file.close()
            
            ensure(
                os.path.exists(temp_file.name),
                "Temporary file must be created successfully",
                context={"temp_file": temp_file.name}
            )
            
            return temp_file.name
        
        elif source_type == "url":
            require(
                source.startswith(('http://', 'https://')),
                "URL source must have valid protocol",
                context={"source": source[:50]}
            )
            return source  # Docling handles URLs directly
        
        elif source_type == "path":
            require(
                os.path.exists(source),
                f"Document file must exist: {source}",
                context={"path": source}
            )
            return source
        
        else:
            require(
                False,
                f"Source type must be supported: {source_type}",
                context={"supported_types": ["base64", "url", "path"]}
            )
    
    def _extract_tables(self, document) -> List[Dict[str, Any]]:
        """Extract table data from document."""
        assert_not_none(document, "document", {"operation": "table_extraction"})
        
        tables = []
        
        if hasattr(document, 'tables'):
            for i, table in enumerate(document.tables):
                assert_not_none(table, f"table_{i}", {"operation": "table_extraction"})
                
                table_data = {
                    "index": i,
                    "caption": getattr(table, 'caption', ''),
                    "page": getattr(table, 'page', 0)
                }
                
                # Extract table data with assertion validation
                if hasattr(table, 'export_to_dataframe'):
                    df = table.export_to_dataframe()
                    assert_not_none(df, f"dataframe_{i}", {"operation": "table_extraction"})
                    
                    table_data["data"] = df.to_dict()
                    table_data["rows"] = len(df)
                    table_data["columns"] = len(df.columns)
                    
                    ensure(
                        table_data["rows"] >= 0 and table_data["columns"] >= 0,
                        "Table dimensions must be non-negative",
                        context={"rows": table_data["rows"], "columns": table_data["columns"]}
                    )
                else:
                    table_data["data"] = str(table)
                
                tables.append(table_data)
        
        assert_type(tables, list, "tables", {"operation": "table_extraction"})
        return tables
    
    def _extract_mathematical_content(self, text: str) -> List[Dict[str, Any]]:
        """Extract mathematical formulas and equations from text."""
        assert_not_none(text, "text", {"operation": "math_extraction"})
        assert_type(text, str, "text", {"operation": "math_extraction"})
        
        mathematical_content = []
        
        import re
        
        # Patterns for mathematical content
        patterns = [
            (r'\$[^$]+\$', 'latex_inline'),
            (r'\$\$[^$]+\$\$', 'latex_display'),
            (r'\\begin\{equation\}.*?\\end\{equation\}', 'latex_equation'),
            (r'[a-zA-Z]\s*=\s*[0-9+\-*/().\s]+', 'simple_equation'),
            (r'\b\d+\.?\d*\s*[+\-*/]\s*\d+\.?\d*\s*=\s*\d+\.?\d*\b', 'arithmetic'),
        ]
        
        for pattern, content_type in patterns:
            assert_type(pattern, str, "pattern", {"operation": "math_extraction"})
            assert_type(content_type, str, "content_type", {"operation": "math_extraction"})
            
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                assert_type(match, str, "match", {"operation": "math_extraction"})
                
                math_item = {
                    "content": match.strip(),
                    "type": content_type,
                    "length": len(match)
                }
                
                ensure(
                    len(math_item["content"]) > 0,
                    "Mathematical content must not be empty after stripping",
                    context={"original_match": match, "type": content_type}
                )
                
                mathematical_content.append(math_item)
        
        assert_type(mathematical_content, list, "mathematical_content", {"operation": "math_extraction"})
        return mathematical_content
    
    def _cleanup_temp_file(self, file_path: str):
        """Clean up temporary files."""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    @contract(
        preconditions=[lambda question: isinstance(question, str) and len(question) > 0],
        postconditions=[lambda result, question: isinstance(result, dict)]
    )
    def analyze_for_gaia(self, question: str, document_source: str = None) -> Dict[str, Any]:
        """
        Analyze document content specifically for GAIA question answering.
        
        Args:
            question: GAIA question to answer
            document_source: Optional document to analyze
            
        Returns:
            Analysis results optimized for GAIA
        """
        assert_not_none(question, "question", {"operation": "gaia_analysis"})
        
        analysis_result = {
            "question": question,
            "document_analysis": None,
            "relevant_content": [],
            "mathematical_elements": [],
            "table_data": [],
            "confidence_score": 0.0
        }
        
        try:
            # Process document if provided
            if document_source:
                doc_result = self.process_document(document_source)
                
                if doc_result["success"]:
                    analysis_result["document_analysis"] = doc_result
                    
                    # Extract content relevant to the question
                    relevant_content = self._find_relevant_content(
                        question, 
                        doc_result["content"]["text"]
                    )
                    analysis_result["relevant_content"] = relevant_content
                    
                    # Extract mathematical elements if question involves math
                    if self._is_mathematical_question(question):
                        analysis_result["mathematical_elements"] = doc_result["mathematical_content"]
                    
                    # Extract table data if question involves data analysis
                    if self._involves_data_analysis(question):
                        analysis_result["table_data"] = doc_result["tables"]
                    
                    # Calculate confidence score
                    analysis_result["confidence_score"] = self._calculate_confidence(
                        question, doc_result
                    )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"GAIA analysis failed: {e}")
            analysis_result["error"] = str(e)
            return analysis_result
    
    def _find_relevant_content(self, question: str, text: str) -> List[str]:
        """Find content relevant to the question."""
        try:
            import re
            
            # Extract key terms from question
            question_terms = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
            
            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            
            # Find sentences containing question terms
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                relevance_score = sum(1 for term in question_terms if term in sentence_lower)
                
                if relevance_score > 0:
                    relevant_sentences.append({
                        "text": sentence.strip(),
                        "relevance_score": relevance_score
                    })
            
            # Sort by relevance and return top matches
            relevant_sentences.sort(key=lambda x: x["relevance_score"], reverse=True)
            return [s["text"] for s in relevant_sentences[:5]]
            
        except Exception as e:
            logger.warning(f"Relevant content extraction failed: {e}")
            return []
    
    def _is_mathematical_question(self, question: str) -> bool:
        """Check if question involves mathematical computation."""
        math_indicators = [
            'calculate', 'compute', 'sum', 'average', 'mean', 'total',
            'multiply', 'divide', 'add', 'subtract', 'equation', 'formula',
            'percentage', 'ratio', 'statistics', 'number', 'value'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in math_indicators)
    
    def _involves_data_analysis(self, question: str) -> bool:
        """Check if question involves data analysis."""
        data_indicators = [
            'table', 'data', 'chart', 'graph', 'column', 'row',
            'compare', 'analysis', 'trend', 'pattern', 'distribution'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in data_indicators)
    
    def _calculate_confidence(self, question: str, doc_result: Dict[str, Any]) -> float:
        """Calculate confidence score for GAIA answer."""
        confidence = 0.0
        
        try:
            # Base confidence from successful processing
            if doc_result["success"]:
                confidence += 0.3
            
            # Boost for mathematical content if math question
            if self._is_mathematical_question(question) and doc_result["mathematical_content"]:
                confidence += 0.3
            
            # Boost for table data if data question
            if self._involves_data_analysis(question) and doc_result["tables"]:
                confidence += 0.3
            
            # Boost for content length (more content = potentially more relevant)
            content_length = len(doc_result["content"]["text"])
            if content_length > 1000:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.0


# Global instance for use across the system
docling_processor = DoclingProcessor()
"""
Internal File Processor with Zero-Space Programming

This module provides fallback file processing capabilities when primary processors
(like Docling) fail, using assertion-based validation and contract programming.
"""

import base64
import io
import logging
import re
import warnings
import time
import traceback
from typing import Dict, Any, Optional
import PyPDF2

from tools.logging import get_logger
from tools.assertions import (
    require, ensure, assert_not_none, assert_type, contract,
    GaiaAssertionError, assert_execution_time, assert_memory_usage
)
from tools.langfuse_monitor import trace_span, log_assertion_failure, create_tracer

logger = get_logger(__name__)

class InternalFileProcessor:
    """Internal file processor with Zero-Space Programming validation."""
    
    def __init__(self, docling_processor=None):
        """
        Initialize internal file processor with dependency injection.
        
        Args:
            docling_processor: Optional docling processor instance for PDF processing
        """
        self.docling_processor = docling_processor
        self.supported_types = ['application/pdf', 'text/plain', 'application/json', 'text/csv']
        self.tracer = None
    
    def _create_assertion_context(self, operation: str, file_data=None, **kwargs) -> dict:
        """Create comprehensive assertion context for debugging and monitoring."""
        context = {
            "operation": operation,
            "timestamp": time.time(),
            "processor_state": {
                "docling_available": self.docling_processor is not None,
                "supported_types": self.supported_types
            }
        }
        
        if file_data:
            context["file_info"] = {
                "name": getattr(file_data, 'name', 'unknown'),
                "type": getattr(file_data, 'type', 'unknown'),
                "content_length": len(getattr(file_data, 'content', ''))
            }
        
        context.update(kwargs)
        return context
    
    def _enhanced_assert_with_langfuse(self, condition: bool, message: str, context: dict):
        """Enhanced assertion that logs failures to LangFuse"""
        if not condition:
            # Create detailed failure context
            failure_context = {
                "assertion_message": message,
                "assertion_context": context,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            }
            
            # Log to LangFuse if tracer is available
            if self.tracer:
                with self.tracer.trace(
                    name="assertion_failure",
                    input=failure_context
                ) as span:
                    span.update(
                        output={"assertion_failed": True},
                        metadata={
                            "severity": "error",
                            "assertion_type": context.get("operation", "unknown"),
                            "file_context": context.get("file_name", "unknown")
                        }
                    )
            
            # Log assertion failure
            log_assertion_failure(failure_context)
            
            # Raise the assertion error
            raise GaiaAssertionError(message, context)
    
    @contract(
        preconditions=[
            lambda self, content, file_name: isinstance(content, str) and len(content) > 0,
            lambda self, content, file_name: isinstance(file_name, str) and len(file_name) > 0
        ],
        postconditions=[lambda result, self, content, file_name: isinstance(result, dict)]
    )
    def process_pdf_fallback(self, content: str, file_name: str) -> Dict[str, Any]:
        """
        Process PDF using PyPDF2 as fallback with Zero-Space Programming.
        
        Args:
            content: Base64 encoded PDF content
            file_name: Name of the PDF file
            
        Returns:
            Dict containing processing results
        """
        assert_not_none(content, "content", {"operation": "pdf_fallback", "file_name": file_name})
        assert_type(content, str, "content", {"operation": "pdf_fallback"})
        assert_not_none(file_name, "file_name", {"operation": "pdf_fallback"})
        assert_type(file_name, str, "file_name", {"operation": "pdf_fallback"})
        
        require(
            len(content.strip()) > 0,
            "PDF content must not be empty",
            context={"file_name": file_name, "content_length": len(content)}
        )
        
        result = {
            "success": False,
            "content": {"text": "", "pages": 0},
            "error": None,
            "processor": "internal_pypdf2"
        }
        
        try:
            # Decode PDF content with assertion validation
            pdf_data = base64.b64decode(content)
            assert_not_none(pdf_data, "pdf_data", {"operation": "pdf_fallback", "file_name": file_name})
            
            require(
                len(pdf_data) > 0,
                "Decoded PDF data must not be empty",
                context={"file_name": file_name, "decoded_length": len(pdf_data)}
            )
            
            # Use PyPDF2 for text extraction with assertion validation
            # Global warning suppression is handled in main.py
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data), strict=False)
            assert_not_none(pdf_reader, "pdf_reader", {"operation": "pdf_fallback"})
            
            page_count = len(pdf_reader.pages)
            assert_type(page_count, int, "page_count", {"operation": "pdf_fallback"})
            
            require(
                page_count > 0,
                "PDF must contain at least one page",
                context={"file_name": file_name, "page_count": page_count}
            )
            
            extracted_text = ""
            successful_pages = 0
            
            for page_num, page in enumerate(pdf_reader.pages):
                assert_not_none(page, f"page_{page_num}", {"operation": "pdf_fallback"})
                
                try:
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 0:
                        extracted_text += f"Page {page_num + 1}:\n{page_text.strip()}\n\n"
                        successful_pages += 1
                    else:
                        # Zero-Space Programming: Alternative text extraction with assertions
                        if '/Contents' in page:
                            contents_obj = page['/Contents']
                            assert_not_none(contents_obj, "contents_obj", {"operation": "pdf_alternative_extraction"})
                            
                            contents = str(contents_obj)
                            assert_type(contents, str, "contents", {"operation": "pdf_alternative_extraction"})
                            
                            # Extract text from PDF content stream using Zero-Space Programming
                            text_matches = re.findall(r'\((.*?)\)\s*Tj', contents)
                            assert_type(text_matches, list, "text_matches", {"operation": "pdf_alternative_extraction"})
                            
                            if text_matches:
                                page_content = ' '.join(text_matches)
                                assert_type(page_content, str, "page_content", {"operation": "pdf_alternative_extraction"})
                                
                                if len(page_content.strip()) > 0:
                                    extracted_text += f"Page {page_num + 1}:\n{page_content.strip()}\n\n"
                                    successful_pages += 1
                                    logger.info(f"Extracted text from page {page_num + 1} using alternative method")
                except Exception as page_error:
                    logger.warning(f"Failed to extract text from page {page_num + 1} in {file_name}: {str(page_error)}")
            
            # Validate extraction results
            assert_type(extracted_text, str, "extracted_text", {"operation": "pdf_fallback"})
            assert_type(successful_pages, int, "successful_pages", {"operation": "pdf_fallback"})
            
            if len(extracted_text.strip()) > 0:
                result["success"] = True
                result["content"]["text"] = extracted_text.strip()
                result["content"]["pages"] = page_count
                result["content"]["successful_pages"] = successful_pages
                
                ensure(
                    result["success"] is True,
                    "PDF processing must be marked as successful when text is extracted",
                    context={"file_name": file_name, "text_length": len(extracted_text)}
                )
                
                logger.info(f"Successfully extracted text from {successful_pages}/{page_count} pages in {file_name}")
            else:
                # Handle no text case with proper validation
                error_msg = "No extractable text found in PDF"
                assert_type(error_msg, str, "error_msg", {"operation": "pdf_no_text"})
                
                result["error"] = error_msg
                # Reduce log level for test documents to avoid noise
                if "test_document" in file_name.lower():
                    logger.debug(f"No extractable text found in test PDF: {file_name}")
                else:
                    logger.warning(f"No extractable text found in PDF: {file_name}")
                
        except Exception as e:
            # Assert error handling with context
            error_msg = f"PDF processing failed: {str(e)}"
            assert_type(error_msg, str, "error_msg", {"operation": "pdf_fallback_error"})
            
            result["error"] = error_msg
            logger.error(f"Internal PDF processing error for {file_name}: {error_msg}")
            
            # Ensure error result structure is valid
            ensure(
                "error" in result and isinstance(result["error"], str),
                "Error result must contain valid error message",
                context={"file_name": file_name, "error": str(e)}
            )
        
        # Final validation of result structure
        assert_type(result, dict, "result", {"operation": "pdf_fallback"})
        require(
            "success" in result and "content" in result,
            "Result must contain success and content fields",
            context={"result_keys": list(result.keys()), "file_name": file_name}
        )
        
        ensure(
            isinstance(result["success"], bool),
            "Success field must be boolean",
            context={"success_type": type(result["success"]).__name__}
        )
        
        return result
    
    @contract(
        preconditions=[
            lambda self, content, file_name, file_type: isinstance(content, str) and len(content) > 0,
            lambda self, content, file_name, file_type: isinstance(file_name, str) and len(file_name) > 0,
            lambda self, content, file_name, file_type: isinstance(file_type, str) and len(file_type) > 0
        ],
        postconditions=[lambda result, self, content, file_name, file_type: isinstance(result, dict)]
    )
    def process_text_file(self, content: str, file_name: str, file_type: str) -> Dict[str, Any]:
        """
        Process text-based files with Zero-Space Programming validation.
        
        Args:
            content: Base64 encoded file content
            file_name: Name of the file
            file_type: MIME type of the file
            
        Returns:
            Dict containing processing results
        """
        assert_not_none(content, "content", {"operation": "text_processing", "file_name": file_name})
        assert_type(content, str, "content", {"operation": "text_processing"})
        assert_not_none(file_name, "file_name", {"operation": "text_processing"})
        assert_type(file_name, str, "file_name", {"operation": "text_processing"})
        assert_not_none(file_type, "file_type", {"operation": "text_processing"})
        assert_type(file_type, str, "file_type", {"operation": "text_processing"})
        
        require(
            file_type in ['text/plain', 'text/csv', 'application/json'],
            f"File type must be supported text type: {file_type}",
            context={"supported_types": ['text/plain', 'text/csv', 'application/json']}
        )
        
        result = {
            "success": False,
            "content": {"text": "", "type": file_type},
            "error": None,
            "processor": "internal_text"
        }
        
        try:
            # Decode text content with assertion validation
            decoded_content = base64.b64decode(content).decode('utf-8')
            assert_not_none(decoded_content, "decoded_content", {"operation": "text_processing", "file_name": file_name})
            assert_type(decoded_content, str, "decoded_content", {"operation": "text_processing"})
            
            require(
                len(decoded_content.strip()) > 0,
                "Decoded text content must not be empty",
                context={"file_name": file_name, "content_length": len(decoded_content)}
            )
            
            result["success"] = True
            result["content"]["text"] = decoded_content
            result["content"]["length"] = len(decoded_content)
            result["content"]["lines"] = len(decoded_content.splitlines())
            
            ensure(
                result["success"] is True,
                "Text processing must be marked as successful",
                context={"file_name": file_name, "content_length": len(decoded_content)}
            )
            
            logger.info(f"Successfully processed text file: {file_name} ({file_type})")
            
        except UnicodeDecodeError as decode_error:
            error_msg = f"Text decoding failed: {str(decode_error)}"
            result["error"] = error_msg
            logger.error(f"Text decoding error for {file_name}: {error_msg}")
            
        except Exception as e:
            error_msg = f"Text processing failed: {str(e)}"
            result["error"] = error_msg
            logger.error(f"Text processing error for {file_name}: {error_msg}")
        
        # Final validation of result structure
        assert_type(result, dict, "result", {"operation": "text_processing"})
        require(
            "success" in result and "content" in result,
            "Result must contain success and content fields",
            context={"result_keys": list(result.keys()), "file_name": file_name}
        )
        
        ensure(
            isinstance(result["success"], bool),
            "Success field must be boolean",
            context={"success_type": type(result["success"]).__name__}
        )
        
        return result
    
    @contract(
        preconditions=[lambda self, file_type: isinstance(file_type, str) and len(file_type) > 0],
        postconditions=[lambda result, self, file_type: isinstance(result, bool)]
    )
    def is_supported(self, file_type: str) -> bool:
        """
        Check if file type is supported by internal processor.
        
        Args:
            file_type: MIME type to check
            
        Returns:
            True if supported, False otherwise
        """
        assert_not_none(file_type, "file_type", {"operation": "support_check"})
        assert_type(file_type, str, "file_type", {"operation": "support_check"})
        
        supported = file_type in self.supported_types
        assert_type(supported, bool, "supported", {"operation": "support_check"})
        
        return supported

    # @contract(
    #     preconditions=[
    #         lambda self, files, base_question, tracer=None: isinstance(files, list) and len(files) > 0,
    #         lambda self, files, base_question, tracer=None: all(hasattr(f, 'name') and hasattr(f, 'type') and hasattr(f, 'content') for f in files),
    #         lambda self, files, base_question, tracer=None: isinstance(base_question, str),
    #         lambda self, files, base_question, tracer=None: tracer is None or hasattr(tracer, 'start_span')
    #     ],
    #     postconditions=[lambda result, self, files, base_question, tracer=None: isinstance(result, str)]
    # )
    def process_multimodal_files(self, files: list, base_question: str, tracer=None) -> str:
        """
        Process multiple files and append their content to the base question.
        
        Args:
            files: List of file data objects with name, type, and content attributes
            base_question: Base question to append file content to
            tracer: Optional LangFuse tracer for monitoring
            
        Returns:
            Enhanced question string with file content
        """
        # Initialize LangFuse tracer for this processing session
        self.tracer = tracer or create_tracer("multimodal_file_processing", "multimodal_processing")
        logger.info(f"Internal file processor tracer: {type(self.tracer)} - {self.tracer}")
        
        # Enhanced assertions with context
        logger.info("About to run assertions in process_multimodal_files")
        assert_not_none(files, "files", self._create_assertion_context("process_multimodal_files"))
        assert_type(files, list, "files", self._create_assertion_context("process_multimodal_files"))
        assert_not_none(base_question, "base_question", self._create_assertion_context("process_multimodal_files"))
        assert_type(base_question, str, "base_question", self._create_assertion_context("process_multimodal_files"))
        logger.info("Assertions completed successfully")
        
        require(
            len(files) > 0,
            "Files list cannot be empty",
            context=self._create_assertion_context("process_multimodal_files", files_count=len(files))
        )
        
        # Zero-Space Programming: Start LangFuse session tracking with comprehensive monitoring
        span_id = None
        processing_context = self._create_assertion_context("process_multimodal_files")
        
        if self.tracer:
            span_id = self.tracer.start_span(
                "multimodal_processing_session",
                {
                    "file_count": len(files),
                    "file_types": [f.type for f in files],
                    "question_length": len(base_question)
                }
            )
            assert_not_none(span_id, "span_id", processing_context)
        
        try:
            # Process files with comprehensive monitoring and validation
            user_message_content = self._process_files_with_monitoring(files, base_question, None)
            assert_not_none(user_message_content, "user_message_content", processing_context)
            assert_type(user_message_content, str, "user_message_content", processing_context)
            
            # Zero-Space Programming: Validate processing results
            require(
                len(user_message_content) >= len(base_question),
                "Enhanced content must be at least as long as base question",
                context={**processing_context, "base_length": len(base_question), "enhanced_length": len(user_message_content)}
            )
            
            # Log successful processing metrics with LangFuse
            if self.tracer and span_id:
                self.tracer.end_span(span_id, {
                    "enhanced_content_length": len(user_message_content),
                    "processing_success": True,
                    "files_processed": len(files),
                    "enhancement_ratio": len(user_message_content) / len(base_question)
                })
            
            ensure(
                isinstance(user_message_content, str),
                "Result must be string type",
                context=processing_context
            )
            
            return user_message_content
            
        except Exception as e:
            # Zero-Space Programming: Comprehensive error handling with context preservation
            error_context = {
                **processing_context,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "files_attempted": len(files),
                "processing_stage": getattr(e, 'processing_stage', 'unknown')
            }
            
            # Log processing failure with detailed context
            if self.tracer and span_id:
                self.tracer.end_span(span_id, {
                    "error": str(e),
                    "processing_success": False,
                    **error_context
                }, error=str(e))
            
            log_assertion_failure({
                "error_type": "multimodal_processing_failure",
                "message": str(e),
                **error_context
            })
            
            # Re-raise with enhanced context
            raise AssertionError(f"Multimodal processing failed: {str(e)}") from e
    
    def _process_files_with_monitoring(self, files: list, base_question: str, trace) -> str:
        """Process files with monitoring and assertions."""
        user_message_content = base_question
        logger.info(f"Processing {len(files)} multimodal files")
        for i, file_data in enumerate(files):
            # Enhanced file-level assertions with context
            file_context = self._create_assertion_context("file_processing", file_data, file_index=i)
            
            assert_not_none(file_data, "file_data", file_context)
            assert_not_none(file_data.name, "file_data.name", file_context)
            assert_not_none(file_data.type, "file_data.type", file_context)
            assert_not_none(file_data.content, "file_data.content", file_context)
            
            require(
                len(file_data.name.strip()) > 0,
                "File name cannot be empty",
                context=file_context
            )
            
            require(
                len(file_data.content.strip()) > 0,
                "File content cannot be empty",
                context=file_context
            )
            
            logger.info(f"Processing file: {file_data.name} (type: {file_data.type})")
            
            # Process file with type-specific handling and monitoring
            try:
                if file_data.type == 'application/pdf':
                    # ZPDF processing with performance monitoring
                    start_time = time.time()
                    user_message_content = self._process_pdf_file(file_data, user_message_content)
                    execution_time = time.time() - start_time
                    require(
                        execution_time <= 30.0,
                        f"PDF processing took {execution_time:.2f}s, exceeding limit of 30.0s",
                        context={**file_context, "execution_time": execution_time, "limit": 30.0}
                    )
                elif file_data.type in ['text/plain', 'text/csv', 'application/json']:
                    # Text processing with performance monitoring
                    start_time = time.time()
                    user_message_content = self._process_text_file_multimodal(file_data, user_message_content)
                    execution_time = time.time() - start_time
                    require(
                        execution_time <= 5.0,
                        f"Text processing took {execution_time:.2f}s, exceeding limit of 5.0s",
                        context={**file_context, "execution_time": execution_time, "limit": 5.0}
                    )
                else:
                    user_message_content = self._process_other_file(file_data, user_message_content)
                    
                # Validate processing result
                assert_type(user_message_content, str, "user_message_content", file_context)
                
            except Exception as e:
                # Enhanced error handling with context
                error_context = self._create_assertion_context(
                    "file_processing_error",
                    file_data,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                
                if self.tracer:
                    error_span_id = self.tracer.start_span(
                        "file_processing_error",
                        error_context
                    )
                    self.tracer.end_span(error_span_id, {
                        "processing_failed": True,
                        "file_name": file_data.name,
                        "error": str(e)
                    })
                
                logger.error(f"Failed to process file {file_data.name}: {str(e)}")
                # Continue processing other files instead of failing completely
                user_message_content += f"\n\nFile: {file_data.name} (Type: {file_data.type}) - Processing failed: {str(e)}"
        
        # Final validation
        ensure(
            len(user_message_content) >= len(base_question),
            "Processed content must not be shorter than original",
            context=self._create_assertion_context(
                "final_validation",
                original_length=len(base_question),
                enhanced_length=len(user_message_content)
            )
        )
        
        assert_type(user_message_content, str, "user_message_content", {"operation": "process_multimodal_files"})
        return user_message_content
    
    def _process_pdf_file(self, file_data, user_message_content: str) -> str:
        """Process PDF file with docling or internal processor fallback."""
        file_context = self._create_assertion_context("pdf_processing", file_data)
        
        assert_not_none(file_data, "file_data", file_context)
        assert_type(user_message_content, str, "user_message_content", file_context)
        
        # LangFuse monitoring for PDF processing
        if self.tracer:
            span_id = self.tracer.start_span(
                "pdf_file_processing",
                {
                    "file_name": file_data.name,
                    "file_size_mb": len(file_data.content) / (1024 * 1024),
                    "processor_available": self.docling_processor is not None
                }
            )
            try:
                result = self._execute_pdf_processing(file_data, user_message_content, None)
                
                # Log successful processing metrics
                self.tracer.end_span(span_id, {
                    "content_extracted": len(result) > len(user_message_content),
                    "processing_time_seconds": getattr(self, '_last_processing_time', 0),
                    "success": True,
                    "text_extracted_length": len(result) - len(user_message_content),
                    "processing_method_used": getattr(self, '_last_processing_method', 'unknown')
                })
                return result
                
            except Exception as e:
                # Log processing failure with context
                self.tracer.end_span(span_id, {
                    "error": str(e),
                    "success": False,
                    "error_type": type(e).__name__,
                    "processing_method": getattr(self, '_last_processing_method', 'unknown')
                }, error=str(e))
                
                # Log assertion failure if it's an assertion error
                if isinstance(e, GaiaAssertionError):
                    log_assertion_failure({
                        "error_type": "pdf_processing_assertion_failure",
                        "message": str(e),
                        "file_name": file_data.name,
                        "processing_method": getattr(self, '_last_processing_method', 'unknown'),
                        "assertion_context": getattr(e, 'context', {})
                    })
                
                raise
        else:
            # Fallback without LangFuse
            return self._execute_pdf_processing(file_data, user_message_content, None)
    
    def _execute_pdf_processing(self, file_data, user_message_content: str, span) -> str:
        """Execute PDF processing with timing and method tracking."""
        start_time = time.time()
        pdf_processed = False
        
        # Try docling processor first if available
        if self.docling_processor:
            self._last_processing_method = "docling"
            try:
                docling_result = self.docling_processor.process_pdf(file_data.content, file_data.name)
                file_context = self._create_assertion_context("docling_processing", file_data)
                assert_not_none(docling_result, "docling_result", file_context)
                
                if docling_result.get("success", False):
                    content = docling_result.get("content", {})
                    assert_type(content, dict, "content", file_context)
                    
                    text_content = content.get("text", "")
                    if text_content and len(text_content.strip()) > 0:
                        user_message_content += f"\n\nPDF file content from {file_data.name}:\n{text_content}"
                        pdf_processed = True
                        logger.info(f"✅ PDF processed successfully with docling: {file_data.name}")
            except Exception as e:
                logger.warning(f"Docling processing failed for {file_data.name}: {str(e)}")
        
        # Fallback to internal processor if docling failed or unavailable
        if not pdf_processed:
            self._last_processing_method = "pypdf2_fallback"
            internal_result = self.process_pdf_fallback(file_data.content, file_data.name)
            file_context = self._create_assertion_context("internal_pdf_processing", file_data)
            assert_not_none(internal_result, "internal_result", file_context)
            
            if internal_result.get("success", False):
                content = internal_result.get("content", {})
                assert_type(content, dict, "content", file_context)
                
                text_content = content.get("text", "")
                if text_content and len(text_content.strip()) > 0:
                    user_message_content += f"\n\nPDF file content from {file_data.name}:\n{text_content}"
                    logger.info(f"✅ PDF processed successfully with internal processor: {file_data.name}")
                else:
                    error_msg = internal_result.get("error", "Unknown processing error")
                    assert_type(error_msg, str, "error_msg", file_context)
                    user_message_content += f"\n\nPDF file provided: {file_data.name}. Note: {error_msg}"
                    logger.warning(f"PDF processing failed for {file_data.name}: {error_msg}")
            else:
                error_msg = internal_result.get("error", "Unknown processing error")
                assert_type(error_msg, str, "error_msg", file_context)
                user_message_content += f"\n\nPDF file provided: {file_data.name}. Note: {error_msg}"
                logger.warning(f"PDF processing failed for {file_data.name}: {error_msg}")
        
        # Record processing time
        self._last_processing_time = time.time() - start_time
        
        return user_message_content
    
    def _process_text_file_multimodal(self, file_data, user_message_content: str) -> str:
        """Process text-based files for multimodal processing."""
        assert_not_none(file_data, "file_data", {"operation": "text_processing"})
        assert_type(user_message_content, str, "user_message_content", {"operation": "text_processing"})
        
        text_result = self.process_text_file(file_data.content, file_data.name, file_data.type)
        assert_not_none(text_result, "text_result", {"operation": "text_processing", "file_name": file_data.name})
        
        if text_result.get("success", False):
            content = text_result.get("content", {})
            assert_type(content, dict, "content", {"operation": "text_processing"})
            
            text_content = content.get("text", "")
            if text_content and len(text_content.strip()) > 0:
                user_message_content += f"\n\n{file_data.type.upper()} file content from {file_data.name}:\n{text_content}"
                logger.info(f"✅ Text file processed successfully: {file_data.name}")
            else:
                error_msg = text_result.get("error", "No extractable content")
                assert_type(error_msg, str, "error_msg", {"operation": "text_processing"})
                user_message_content += f"\n\nText file provided: {file_data.name}. Note: {error_msg}"
                logger.warning(f"Text processing failed for {file_data.name}: {error_msg}")
        else:
            error_msg = text_result.get("error", "Unknown processing error")
            assert_type(error_msg, str, "error_msg", {"operation": "text_processing"})
            user_message_content += f"\n\nText file provided: {file_data.name}. Note: {error_msg}"
            logger.warning(f"Text processing failed for {file_data.name}: {error_msg}")
        
        return user_message_content
    
    def _process_other_file(self, file_data, user_message_content: str) -> str:
        """Process other file types (images, audio, video, Office documents)."""
        assert_not_none(file_data, "file_data", {"operation": "other_file_processing"})
        assert_type(user_message_content, str, "user_message_content", {"operation": "other_file_processing"})
        
        file_info = f"File: {file_data.name} (Type: {file_data.type})"
        assert_type(file_info, str, "file_info", {"operation": "multimodal_processing"})
        
        user_message_content += f"\n\n{file_info} - Content available for analysis"
        logger.info(f"✅ Multimodal file registered: {file_data.name}")
        
        return user_message_content

# Global instance for use across the system with dependency injection
import importlib.util

# Check docling availability and import if available
docling_available = importlib.util.find_spec("docling") is not None
if docling_available:
    try:
        from .docling_processor import docling_processor
    except ImportError:
        docling_processor = None
        logger.warning("Docling import failed, using fallback processor only")
else:
    docling_processor = None
    logger.info("Docling not available, using internal fallback processor only")

# Create global instance with injected dependencies
internal_file_processor = InternalFileProcessor(docling_processor)
"""
Test script for Gemini Vision chart extraction.

This script demonstrates how to use the chart extraction feature
and validates that it's working correctly.
"""
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from content_analyzer.document_parser import DocumentProcessor
from configuration.parameters import parameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_chart_extraction():
    """Test chart extraction on a sample PDF with charts."""
    
    logger.info("=" * 60)
    logger.info("Testing Gemini Vision Chart Extraction")
    logger.info("=" * 60)
    
    # Check if chart extraction is enabled
    if not parameters.ENABLE_CHART_EXTRACTION:
        logger.warning("?? Chart extraction is DISABLED")
        logger.info("Enable it by setting ENABLE_CHART_EXTRACTION=true in .env")
        return
    
    logger.info(f"? Chart extraction enabled")
    logger.info(f"?? Using model: {parameters.CHART_VISION_MODEL}")
    logger.info(f"?? Max tokens: {parameters.CHART_MAX_TOKENS}")
    
    # Initialize processor
    try:
        processor = DocumentProcessor()
        logger.info("? DocumentProcessor initialized")
        
        if processor.gemini_client:
            logger.info("? Gemini Vision client ready")
        else:
            logger.error("? Gemini Vision client not initialized")
            return
            
    except Exception as e:
        logger.error(f"? Failed to initialize processor: {e}")
        return
    
    # Test with example PDF (if exists)
    test_files = [
        "examples/google-2024-environmental-report.pdf",
        "examples/deppseek.pdf",
        "test/sample_with_charts.pdf"
    ]
    
    found_file = None
    for test_file in test_files:
        if os.path.exists(test_file):
            found_file = test_file
            break
    
    if not found_file:
        logger.warning("?? No test PDF files found")
        logger.info("Available test files:")
        for tf in test_files:
            logger.info(f"  - {tf}")
        logger.info("\nTo test manually:")
        logger.info("1. Place a PDF with charts in one of the above locations")
        logger.info("2. Run this script again")
        return
    
    logger.info(f"\n?? Processing test file: {found_file}")
    
    # Create mock file object
    class MockFile:
        def __init__(self, path):
            self.name = path
            self.size = os.path.getsize(path)
    
    try:
        # Process the file
        mock_file = MockFile(found_file)
        chunks = processor.process([mock_file])
        
        logger.info(f"\n? Processing complete!")
        logger.info(f"?? Total chunks extracted: {len(chunks)}")
        
        # Count chart chunks
        chart_chunks = [c for c in chunks if c.metadata.get("type") == "chart"]
        text_chunks = [c for c in chunks if c.metadata.get("type") != "chart"]
        
        logger.info(f"?? Chart chunks: {len(chart_chunks)}")
        logger.info(f"?? Text chunks: {len(text_chunks)}")
        
        # Display chart analyses
        if chart_chunks:
            logger.info(f"\n{'=' * 60}")
            logger.info("?? CHART ANALYSES EXTRACTED:")
            logger.info('=' * 60)
            
            for i, chunk in enumerate(chart_chunks, 1):
                logger.info(f"\n--- Chart {i} ---")
                logger.info(f"Page: {chunk.metadata.get('page')}")
                logger.info(f"Preview: {chunk.page_content[:200]}...")
                logger.info("")
        else:
            logger.info("\n?? No charts detected in this document")
            logger.info("This could mean:")
            logger.info("  - Document contains no charts")
            logger.info("  - Charts are embedded as tables (already extracted)")
            logger.info("  - Charts are too complex for detection")
        
        logger.info(f"\n{'=' * 60}")
        logger.info("? Test completed successfully!")
        logger.info('=' * 60)
        
    except Exception as e:
        logger.error(f"? Test failed: {e}", exc_info=True)


def test_api_connection():
    """Test Gemini API connection."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Gemini API Connection")
    logger.info("=" * 60)
    
    try:
        import google.generativeai as genai
        from PIL import Image
        import io
        
        genai.configure(api_key=parameters.GOOGLE_API_KEY)
        model = genai.GenerativeModel(parameters.CHART_VISION_MODEL)
        
        logger.info("? Gemini client initialized")
        
        # Test with a simple text prompt
        response = model.generate_content("Hello! Can you respond with 'API Working'?")
        logger.info(f"? API Response: {response.text}")
        
        logger.info("? Gemini API connection successful!")
        
    except ImportError as e:
        logger.error(f"? Missing dependency: {e}")
        logger.info("Install with: pip install google-generativeai Pillow")
    except Exception as e:
        logger.error(f"? API test failed: {e}")
        logger.info("Check your GOOGLE_API_KEY in .env file")


if __name__ == "__main__":
    print("\n?? SmartDoc AI - Chart Extraction Test Suite\n")
    
    # Test 1: API Connection
    test_api_connection()
    
    # Test 2: Chart Extraction
    test_chart_extraction()
    
    print("\n? All tests completed!\n")

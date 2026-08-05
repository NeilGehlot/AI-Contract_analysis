"""
PDF Text Extractor
Extracts text from PDF contracts
"""

import pdfplumber
import PyPDF2
from pathlib import Path
from typing import Dict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class PDFExtractor:
    """Extract text from PDF files"""
    
    def __init__(self):
        """Initialize PDF extractor"""
        logger.info("PDFExtractor initialized")
    
    def extract_text(self, pdf_path: str) -> Dict:
        """
        Extract text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Extracting text from: {pdf_path}")
        
        # Validate file
        if not Path(pdf_path).exists():
            logger.error(f"File not found: {pdf_path}")
            return {
                'success': False,
                'error': 'File not found',
                'text': '',
                'pages': 0
            }
        
        # Try pdfplumber first
        try:
            text, pages = self._extract_with_pdfplumber(pdf_path)
            logger.info(f"Extracted {len(text)} characters using pdfplumber")
            return {
                'success': True,
                'text': text,
                'pages': pages,
                'method': 'pdfplumber'
            }
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
        
        # Fallback to PyPDF2
        try:
            text, pages = self._extract_with_pypdf2(pdf_path)
            logger.info(f"Extracted {len(text)} characters using PyPDF2")
            return {
                'success': True,
                'text': text,
                'pages': pages,
                'method': 'PyPDF2'
            }
        except Exception as e:
            logger.error(f"All extraction methods failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'pages': 0
            }
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> tuple:
        """Extract using pdfplumber"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip(), num_pages
    
    def _extract_with_pypdf2(self, pdf_path: str) -> tuple:
        """Extract using PyPDF2"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip(), num_pages
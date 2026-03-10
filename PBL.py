

import sys
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# PDF Processing
try:
    import pdfplumber
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    print("Warning: PDF libraries not installed. Run: pip install pdfplumber PyPDF2")
    PDF_AVAILABLE = False

# NLP Processing
try:
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    print("Warning: spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")
    NLP_AVAILABLE = False


# ============================================================================
# SECTION 1: PDF TEXT EXTRACTION
# ============================================================================

class PDFExtractor:
    """Extracts text from PDF contracts"""
    
    def extract(self, pdf_path: str) -> Dict:
        """
        Extract text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not PDF_AVAILABLE:
            return {'success': False, 'error': 'PDF libraries not available'}
        
        if not Path(pdf_path).exists():
            return {'success': False, 'error': f'File not found: {pdf_path}'}
        
        print(f"📄 Extracting text from PDF: {Path(pdf_path).name}")
        
        # Try pdfplumber first (works best for modern PDFs)
        try:
            text, pages = self._extract_pdfplumber(pdf_path)
            return {
                'success': True,
                'text': text,
                'pages': pages,
                'method': 'pdfplumber'
            }
        except Exception as e:
            print(f"  ⚠ pdfplumber failed: {e}, trying PyPDF2...")
        
        # Fallback to PyPDF2
        try:
            text, pages = self._extract_pypdf2(pdf_path)
            return {
                'success': True,
                'text': text,
                'pages': pages,
                'method': 'PyPDF2'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'All extraction methods failed: {e}',
                'text': '',
                'pages': 0
            }
    
    def _extract_pdfplumber(self, pdf_path: str) -> Tuple[str, int]:
        """Extract using pdfplumber"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip(), num_pages
    
    def _extract_pypdf2(self, pdf_path: str) -> Tuple[str, int]:
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


# ============================================================================
# SECTION 2: TEXT CLEANING
# ============================================================================

class TextCleaner:
    """Cleans and preprocesses contract text"""
    
    def clean(self, text: str) -> str:
        """
        Clean contract text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        print("🧹 Cleaning text...")
        
        # Remove page numbers
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common headers/footers
        text = re.sub(r'CONFIDENTIAL.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'PROPRIETARY.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'DRAFT.*?\n', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove special characters but keep legal symbols
        text = re.sub(r'[^\w\s\.,;:()\-–—§©®™\'\"\[\]{}₹$]', '', text)
        
        # Standardize currency
        text = text.replace('Rs.', '₹')
        text = text.replace('INR', '₹')
        
        return text.strip()
    
    def get_statistics(self, text: str) -> Dict:
        """Get text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'characters': len(text),
            'words': len(words),
            'sentences': len([s for s in sentences if s.strip()]),
            'avg_word_length': round(sum(len(w) for w in words) / len(words), 2) if words else 0
        }


# ============================================================================
# SECTION 3: ENTITY EXTRACTION
# ============================================================================

class EntityExtractor:
    """Extracts key entities from contract text"""
    
    def __init__(self):
        """Initialize entity extractor with spaCy"""
        if NLP_AVAILABLE:
            print("🤖 Loading NLP model...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.nlp_available = True
            except:
                print("  ⚠ spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp_available = False
        else:
            self.nlp_available = False
    
    def extract_all(self, text: str) -> Dict:
        """
        Extract all entities from contract
        
        Args:
            text: Cleaned contract text
            
        Returns:
            Dictionary with all extracted entities
        """
        print("🔍 Extracting entities...")
        
        return {
            'parties': self.extract_parties(text),
            'dates': self.extract_dates(text),
            'amounts': self.extract_amounts(text),
            'emails': self.extract_emails(text),
            'terms': self.extract_terms(text)
        }
    
    def extract_parties(self, text: str) -> List[str]:
        """Extract party names"""
        parties = []
        
        # Method 1: Look for BETWEEN clause
        between_match = re.search(
            r'(?:BETWEEN|between)\s+(.*?)\s+(?:AND|and)\s+(.*?)(?:\.|,|\()',
            text,
            re.DOTALL
        )
        
        if between_match:
            party1 = between_match.group(1).strip()
            party2 = between_match.group(2).strip()
            
            # Clean up
            party1 = re.sub(r'\(.*?\)', '', party1).strip()
            party2 = re.sub(r'\(.*?\)', '', party2).strip()
            
            parties.extend([party1, party2])
        
        # Method 2: Use spaCy if available
        if self.nlp_available:
            doc = self.nlp(text[:5000])  # First 5000 chars
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON']:
                    if ent.text not in parties and len(ent.text) > 3:
                        parties.append(ent.text)
        
        # Remove duplicates and limit
        parties = list(dict.fromkeys(parties))[:10]
        return [p for p in parties if len(p) > 2]
    
    def extract_dates(self, text: str) -> List[Dict]:
        """Extract dates"""
        dates = []
        
        patterns = [
            (r'dated\s+(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})', 'signing_date'),
            (r'effective\s+(?:from|as of)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})', 'effective_date'),
            (r'expir(?:y|ing|es)\s+(?:on|date)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})', 'expiry_date'),
            (r'(\d{1,2}\s+\w+\s+\d{4})', 'general_date'),
            (r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})', 'general_date'),
        ]
        
        for pattern, date_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates.append({
                    'date': match.group(1),
                    'type': date_type
                })
        
        # Remove duplicates
        seen = set()
        unique_dates = []
        for d in dates:
            if d['date'] not in seen:
                seen.add(d['date'])
                unique_dates.append(d)
        
        return unique_dates[:10]
    
    def extract_amounts(self, text: str) -> List[Dict]:
        """Extract monetary amounts"""
        amounts = []
        
        patterns = [
            (r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', '₹'),
            (r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)', '₹'),
            (r'INR\s*(\d+(?:,\d+)*(?:\.\d+)?)', '₹'),
            (r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', '$'),
            (r'USD\s*(\d+(?:,\d+)*(?:\.\d+)?)', '$'),
        ]
        
        for pattern, currency in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount_value = float(amount_str)
                    amounts.append({
                        'amount': match.group(1),
                        'currency': currency,
                        'value': amount_value
                    })
                except ValueError:
                    pass
        
        # Sort by value
        amounts.sort(key=lambda x: x['value'], reverse=True)
        return amounts[:10]
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(pattern, text)
        return list(set(emails))[:5]
    
    def extract_terms(self, text: str) -> Dict:
        """Extract contract terms (duration, notice period, etc.)"""
        terms = {}
        
        # Duration/Term
        term_patterns = [
            r'term\s+of\s+(\d+)\s+(year|month|day)s?',
            r'period\s+of\s+(\d+)\s+(year|month|day)s?',
            r'duration\s+of\s+(\d+)\s+(year|month|day)s?',
        ]
        
        for pattern in term_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                terms['duration'] = f"{match.group(1)} {match.group(2)}(s)"
                break
        
        # Notice period
        notice_patterns = [
            r'notice\s+(?:period\s+)?of\s+(\d+)\s+(day|month)s?',
            r'(\d+)\s+(day|month)s?\s+(?:prior\s+)?notice',
        ]
        
        for pattern in notice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                terms['notice_period'] = f"{match.group(1)} {match.group(2)}(s)"
                break
        
        # Payment terms
        payment_match = re.search(r'payment.*?(\d+)\s+days?', text, re.IGNORECASE)
        if payment_match:
            terms['payment_terms'] = f"{payment_match.group(1)} days"
        
        return terms


# ============================================================================
# SECTION 4: RISK DETECTION
# ============================================================================

class RiskDetector:
    """Detects potential risks in contracts"""
    
    def detect(self, text: str, entities: Dict) -> List[Dict]:
        """
        Detect risks in contract
        
        Args:
            text: Cleaned contract text
            entities: Extracted entities
            
        Returns:
            List of detected risks
        """
        print("⚠️  Detecting risks...")
        
        risks = []
        text_lower = text.lower()
        
        # Risk 1: Unlimited liability
        if 'unlimited liability' in text_lower or 'unlimited damages' in text_lower:
            risks.append({
                'type': 'unlimited_liability',
                'severity': 'HIGH',
                'title': 'Unlimited Liability Clause',
                'description': 'Contract contains unlimited liability provision',
                'recommendation': 'Negotiate a liability cap (typically 1-2x contract value)',
                'indian_context': 'Consider Indian Contract Act Section 74 limitations'
            })
        
        # Risk 2: Long payment terms
        payment_terms = entities.get('terms', {}).get('payment_terms', '')
        if payment_terms:
            try:
                days = int(''.join(filter(str.isdigit, payment_terms)))
                if days > 60:
                    risks.append({
                        'type': 'long_payment_terms',
                        'severity': 'MEDIUM',
                        'title': f'{days}-Day Payment Terms',
                        'description': f'Payment terms of {days} days exceed industry standard',
                        'recommendation': f'Negotiate to 30-45 days. Current: {days} days (95th percentile)',
                        'indian_context': 'MSME Act mandates 45-day maximum for MSMEs'
                    })
            except:
                pass
        
        # Risk 3: Automatic renewal
        if 'automatic renewal' in text_lower or 'auto-renew' in text_lower:
            if 'notice' not in text_lower[text_lower.find('auto'):text_lower.find('auto')+200]:
                risks.append({
                    'type': 'automatic_renewal',
                    'severity': 'MEDIUM',
                    'title': 'Automatic Renewal Without Notice',
                    'description': 'Contract auto-renews without clear notice requirement',
                    'recommendation': 'Add 60-90 day notice period before renewal',
                    'indian_context': 'Ensure compliance with consumer protection laws'
                })
        
        # Risk 4: Long notice period
        notice_period = entities.get('terms', {}).get('notice_period', '')
        if notice_period:
            try:
                if 'month' in notice_period:
                    months = int(''.join(filter(str.isdigit, notice_period)))
                    if months > 3:
                        risks.append({
                            'type': 'long_notice_period',
                            'severity': 'LOW',
                            'title': f'{months}-Month Notice Period',
                            'description': f'Notice period of {months} months is longer than standard',
                            'recommendation': f'Negotiate to 30-90 days. Current: {months} months',
                            'indian_context': 'No legal maximum, but 90 days is market standard'
                        })
            except:
                pass
        
        # Risk 5: Missing force majeure
        if 'force majeure' not in text_lower:
            risks.append({
                'type': 'missing_force_majeure',
                'severity': 'MEDIUM',
                'title': 'Missing Force Majeure Clause',
                'description': 'No force majeure protection found',
                'recommendation': 'Add force majeure clause covering natural disasters, pandemics, govt actions',
                'indian_context': 'Critical post-COVID; not automatic under Indian Contract Act'
            })
        
        # Risk 6: Missing indemnification
        if 'indemnif' not in text_lower and 'indemnity' not in text_lower:
            risks.append({
                'type': 'missing_indemnification',
                'severity': 'MEDIUM',
                'title': 'Missing Indemnification Clause',
                'description': 'No indemnification protection identified',
                'recommendation': 'Add mutual indemnification clause with clear scope',
                'indian_context': 'Important for limiting liability under Indian law'
            })
        
        # Risk 7: Missing governing law
        if 'governing law' not in text_lower and 'jurisdiction' not in text_lower:
            risks.append({
                'type': 'missing_governing_law',
                'severity': 'HIGH',
                'title': 'Missing Governing Law Clause',
                'description': 'No governing law or jurisdiction specified',
                'recommendation': 'Specify Indian jurisdiction and applicable state laws',
                'indian_context': 'Critical for enforceability in Indian courts'
            })
        
        # Risk 8: No termination clause
        if 'termination' not in text_lower and 'terminate' not in text_lower:
            risks.append({
                'type': 'missing_termination',
                'severity': 'HIGH',
                'title': 'Missing Termination Clause',
                'description': 'No clear termination provisions found',
                'recommendation': 'Add termination for convenience and for cause clauses',
                'indian_context': 'Specify notice requirements per Indian Contract Act'
            })
        
        return risks



import re
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TextCleaner:
    """ 
    Clean contract text
    """
    
    def __init__(self):
        """
        Initialize text cleaner 
        """
        logger.info("TextCleaner initialized")
    
    def clean(self, text: str):
        """
        Clean contract text
        """
        logger.info(f"Cleaning text ({len(text)} characters)")
        
        
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'CONFIDENTIAL.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'PROPRIETARY.*?\n', '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        text = re.sub(r'[^\w\s\.,;:()\-–—§©®™\'\"\[\]{}₹$]', '', text)
        
        text = text.replace('Rs.', '₹').replace('INR', '₹')
        
        logger.info(f"Cleaned text ({len(text)} characters)")
        return text.strip()
    
    def get_statistics(self, text: str):
        """
        Get text statistics
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0
        }
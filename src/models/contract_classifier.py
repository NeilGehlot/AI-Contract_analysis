import yaml
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
CONTRACT_KEYWORDS=  {
        'NDA': ['non-disclosure', 'confidential', 'nda', 'secrecy'],
        'Employment Agreement': ['employment', 'employee', 'employer', 'job', 'position'],
        'Service Agreement': ['service', 'services', 'scope of work', 'deliverables'],
        'Vendor Agreement': ['vendor', 'supplier', 'purchase', 'procurement'],
        'Consulting Agreement': ['consulting', 'consultant', 'advisory', 'professional services'],
        'Lease Agreement': ['lease', 'rent', 'tenant', 'landlord', 'premises'],
        'Partnership Agreement': ['partnership', 'partner', 'joint venture'],
        'Sales Agreement': ['sale', 'purchase', 'buyer', 'seller']
    }


class ContractClassifier:
    """
    Classify contract type
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize classifier
        """
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        self.contract_types = config['contract_types']
        
        self.keywords =CONTRACT_KEYWORDS
        
        logger.info("ContractClassifier initialized")
    
    def classify(self, text: str):
        """
        Classify contract type
        """
        logger.info("Classifying contract type...")
        
        text_lower = text.lower()
        scores = {}
        
        for contract_type, keywords in CONTRACT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[contract_type] = score
        
        if scores:
            top_type = max(scores, key=scores.get)

            if scores[top_type] == 0:
                return {
                    "type": "Unknown",
                    "confidence": 0.0,
                    "all_scores": scores,
                }
            confidence = scores[top_type] / len(CONTRACT_KEYWORDS[top_type])
            
            logger.info(f"Classified as: {top_type} (confidence: {confidence:.2f})")
            
            return {
                'type': top_type,
                'confidence': confidence,
                'all_scores': scores
            }
        
        return {
            'type': 'Unknown',
            'confidence': 0.0,
            'all_scores': {}
        }
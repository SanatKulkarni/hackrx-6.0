"""
Configuration and Environment Setup
Handles environment variables and system paths
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
loader_dir = os.path.join(parent_dir, 'loader_&_extractor')
chunking_dir = os.path.join(parent_dir, 'chunking')
embeddings_dir = os.path.join(parent_dir, 'embeddings')

sys.path.extend([loader_dir, chunking_dir, embeddings_dir, current_dir, parent_dir])

class Config:
    """Configuration class for the Q&A system"""
    
    # API Keys
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # Pinecone Configuration
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackathon-policies")
    
    # Gemini AI Configuration
    # AI Model Configuration
    GEMINI_MODEL = "gemini-2.5-flash"  # Latest Gemini 2.5 Flash model
    
    # Document Processing Configuration
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    
    # Semantic Search Configuration  
    SEMANTIC_MODEL = 'all-MiniLM-L6-v2'  # Fast model (384 dimensions) - 5x faster than mpnet
    MAX_CHUNKS_TO_RETRIEVE = 1000
    MAX_TOPIC_QUERIES = 150
    MAX_SEARCH_QUERIES = 50
    
    @classmethod
    def validate_required_keys(cls):
        """Validate that required environment variables are set"""
        required_keys = {
            'GEMINI_API_KEY': cls.GEMINI_API_KEY,
            'PINECONE_API_KEY': cls.PINECONE_API_KEY
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
        
        return True

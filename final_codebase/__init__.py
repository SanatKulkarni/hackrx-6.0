"""
Final Codebase Package
Modular components for HackRX 6.0 Q&A System
"""

# Make all main components available at package level
from .hackathon_qa_system import HackathonQASystem
from .config import Config
from .gemini_ai_client import GeminiAIClient
from .semantic_search import SemanticSearchEngine
from .document_manager import DocumentManager
from .context_retriever import ContextRetriever
from .query_generator import QueryGenerator
from .module_importer import import_modules

__all__ = [
    'HackathonQASystem',
    'Config',
    'GeminiAIClient', 
    'SemanticSearchEngine',
    'DocumentManager',
    'ContextRetriever',
    'QueryGenerator',
    'import_modules'
]

"""
Hackathon Q&A System - Main System
Complete RAG Pipeline using modular components
Format: {"documents": "URL", "questions": [...]} -> {"answers": [...]}
"""

import time
import json
from typing import List, Dict, Any

# Import all components
from .config import Config
from .module_importer import import_modules
from .gemini_ai_client import GeminiAIClient
from .semantic_search import SemanticSearchEngine
from .document_manager import DocumentManager
from .context_retriever import ContextRetriever

class HackathonQASystem:
    """Complete Q&A system for hackathon requirements"""
    
    def __init__(self, index_name="hackathon-policies", namespace="documents"):
        """
        Initialize the Q&A system
        
        Args:
            index_name (str): Pinecone index name
            namespace (str): Pinecone namespace
        """
        print("🚀 Initializing Hackathon Q&A System with GitHub AI...")
        
        # Validate configuration
        Config.validate_required_keys()
        
        # Store configuration
        self.index_name = index_name
        self.namespace = namespace
        
        # Import required modules
        self.process_document, self.chunk_text, self.embed_and_store_chunks, self.query_embeddings = import_modules()
        
        # Initialize components
        self.gemini_ai_client = GeminiAIClient()
        self.semantic_engine = SemanticSearchEngine()
        self.document_manager = DocumentManager(
            self.process_document, self.chunk_text, 
            self.embed_and_store_chunks, self.query_embeddings
        )
        self.context_retriever = ContextRetriever()
        
        # State variables
        self.chunk_embeddings = None
        
        print("✅ Hackathon Q&A System initialized successfully!")
    
    def process_and_store_document(self, document_url: str) -> bool:
        """
        Process document from URL and store in Pinecone
        
        Args:
            document_url (str): URL of the document to process
            
        Returns:
            bool: Success status
        """
        return self.document_manager.process_and_store_document(
            document_url, self.index_name, self.namespace
        )
    
    def load_document_chunks(self) -> bool:
        """Load all document chunks into memory for fast retrieval"""
        success, chunk_texts = self.document_manager.load_document_chunks(
            self.index_name, self.namespace
        )
        
        if success and chunk_texts and self.semantic_engine.model:
            print("🔮 Computing Nomic semantic embeddings for chunks...")
            self.chunk_embeddings = self.semantic_engine.encode_texts(
                chunk_texts, show_progress=True
            )
            if self.chunk_embeddings is not None:
                print(f"✅ Computed Nomic embeddings for {len(chunk_texts)} chunks")
        
        return success
    
    def get_relevant_context(self, questions: List[str]) -> str:
        """
        Get relevant context for questions using advanced retrieval
        
        Args:
            questions (List[str]): Questions to find context for
            
        Returns:
            str: Relevant context text
        """
        # Load chunks if not already loaded
        if not self.document_manager.chunks_loaded:
            if not self.load_document_chunks():
                return ""
        
        # Get context using retriever
        return self.context_retriever.get_relevant_context(
            questions, 
            self.document_manager.get_document_chunks(),
            self.chunk_embeddings
        )
    
    def answer_questions_batch(self, questions: List[str]) -> List[str]:
        """
        Answer multiple questions using efficient semantic search + GitHub AI answering
        
        Args:
            questions (List[str]): List of questions to answer
            
        Returns:
            List[str]: List of answers
        """
        try:
            print(f"📝 Processing {len(questions)} questions with ultra-fast retrieval + GitHub AI answering...")
            
            # Step 1: Get relevant context using ultra-fast in-memory retrieval
            relevant_context = self.get_relevant_context(questions)
            
            if not relevant_context:
                print("⚠️  No relevant context available")
                return ["No relevant information found in the document to answer this question."] * len(questions)
            
            # Step 2: Generate answers using Gemini AI
            print(f"🤖 Processing with Gemini AI using ultra-fast retrieved context...")
            answers = self.gemini_ai_client.generate_answers(questions, relevant_context)
            
            print(f"✅ All {len(questions)} questions processed with ultra-fast retrieval + Gemini AI approach")
            return answers
            
        except Exception as e:
            print(f"❌ Error in ultra-fast processing: {e}")
            return ["An error occurred while processing this question."] * len(questions)
    
    def process_hackathon_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete hackathon request format
        
        Args:
            request_data (dict): {"documents": "URL", "questions": [...]}
            
        Returns:
            dict: {"answers": [...]}
        """
        try:
            start_time = time.time()
            
            print("🚀 Processing Hackathon Request with Ultra-Fast GitHub AI System")
            print("="*50)
            
            # Validate input format
            if 'documents' not in request_data or 'questions' not in request_data:
                return {
                    "error": "Invalid request format. Required: {'documents': 'URL', 'questions': [...]}"
                }
            
            document_url = request_data['documents']
            questions = request_data['questions']
            
            print(f"📄 Document URL: {document_url}")
            print(f"❓ Questions: {len(questions)}")
            
            # Step 1: Process and store document (with timing)
            doc_start_time = time.time()
            success = self.process_and_store_document(document_url)
            if not success:
                return {
                    "error": "Failed to process document from URL"
                }
            doc_processing_time = time.time() - doc_start_time
            
            # Step 2: Answer all questions (with timing)
            qa_start_time = time.time()
            print(f"\n🎯 Answering {len(questions)} questions with ultra-fast system...")
            answers = self.answer_questions_batch(questions)
            qa_processing_time = time.time() - qa_start_time
            
            # Step 3: Prepare response
            processing_time = time.time() - start_time
            
            response = {
                "answers": answers
            }
            
            print(f"\n🎉 REQUEST COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"⏱️  TOTAL PROCESSING TIME: {processing_time:.2f} seconds")
            print(f"📄 Document processing: {doc_processing_time:.2f}s ({doc_processing_time/processing_time*100:.1f}%)")
            print(f"🤖 Q&A processing: {qa_processing_time:.2f}s ({qa_processing_time/processing_time*100:.1f}%)")
            if answers:
                print(f"📊 Questions processed: {len(answers)}")
                print(f"⚡ Average time per question: {qa_processing_time/len(questions):.2f}s")
                print(f"🚀 Processing speed: {len(questions)/processing_time:.1f} questions/second")
                if doc_processing_time < 5:  # If using cache
                    print(f"🎯 Q&A-only speed: {len(questions)/qa_processing_time:.1f} questions/second")
            print("="*60)
            
            return response
            
        except Exception as e:
            print(f"❌ Error processing hackathon request: {e}")
            return {
                "error": f"Processing failed: {str(e)}"
            }


def test_hackathon_qa_system():
    """Test the Q&A system with hackathon sample data"""
    
    test_start_time = time.time()
    
    print("🧪 TESTING ULTRA-FAST HACKATHON Q&A SYSTEM WITH GITHUB AI")
    print("="*60)
    
    # Initialize Q&A system
    qa_system = HackathonQASystem(
        index_name="hackathon-qa-test",
        namespace="test-docs"
    )
    
    # Sample hackathon request
    hackathon_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    # Process request
    result = qa_system.process_hackathon_request(hackathon_request)
    
    # Calculate total test time
    total_test_time = time.time() - test_start_time
    
    # Display results
    if "error" in result:
        print(f"❌ Test failed: {result['error']}")
    else:
        print("\n🎉 TEST SUCCESSFUL!")
        print("="*60)
        print("📋 ANSWERS:")
        for i, answer in enumerate(result['answers'], 1):
            print(f"\n{i}. Q: {hackathon_request['questions'][i-1]}")
            print(f"   A: {answer}")
        
        # Validate JSON format
        try:
            json_output = json.dumps(result, indent=2)
            print(f"\n✅ Valid JSON output generated")
            print(f"📊 Response size: {len(json_output)} characters")
        except Exception as e:
            print(f"❌ JSON serialization error: {e}")
    
    # Display total test execution time
    print(f"\n🕒 TOTAL TEST EXECUTION TIME: {total_test_time:.2f} seconds")
    print("="*60)


if __name__ == "__main__":
    test_hackathon_qa_system()

#!/usr/bin/env python3
"""
Diagnostic script for HackRX 6.0 Q&A System
Debug context retrieval issues
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from final_codebase import HackathonQASystem

def diagnose_system():
    """Diagnose what's happening with context retrieval"""
    try:
        print("🔍 DIAGNOSTIC MODE: HackRX 6.0 Q&A System")
        print("=" * 60)
        
        # Initialize the system
        qa_system = HackathonQASystem(
            index_name="hackathon-qa-test",
            namespace="test-docs"
        )
        
        # Test document URL
        test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        
        print(f"📄 Processing document: {test_url}")
        
        # Step 1: Process document
        print("\n🔍 STEP 1: Document Processing")
        print("-" * 40)
        result = qa_system.document_manager.process_and_store_document(
            test_url, 
            qa_system.index_name, 
            qa_system.namespace
        )
        print(f"Document processing result: {result}")
        
        # Step 1.5: Load document chunks (this is what was missing!)
        print("\n🔍 STEP 1.5: Loading Document Chunks")
        print("-" * 40)
        load_result = qa_system.load_document_chunks()
        print(f"Chunk loading result: {load_result}")
        
        # Step 2: Check document chunks
        print("\n🔍 STEP 2: Document Chunks")
        print("-" * 40)
        chunks = qa_system.document_manager.get_document_chunks()
        print(f"Number of chunks: {len(chunks) if chunks else 0}")
        
        if chunks:
            print(f"First chunk preview: {chunks[0]['text'][:200]}...")
            print(f"Chunk structure: {list(chunks[0].keys())}")
        else:
            print("❌ NO CHUNKS FOUND - This is the problem!")
        
        # Step 3: Check embeddings
        print("\n🔍 STEP 3: Chunk Embeddings")
        print("-" * 40)
        embeddings = qa_system.chunk_embeddings
        print(f"Embeddings type: {type(embeddings)}")
        if embeddings is not None:
            print(f"Embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'No shape'}")
        else:
            print("❌ NO EMBEDDINGS FOUND - This is another problem!")
        
        # Step 4: Test semantic search directly
        print("\n🔍 STEP 4: Semantic Search Test")
        print("-" * 40)
        test_questions = ["What is the grace period for premium payment?"]
        
        if chunks and embeddings is not None:
            context = qa_system.semantic_engine.semantic_search(test_questions, chunks, embeddings)
            print(f"Context found: {len(context) if context else 0} characters")
            if context:
                print(f"Context preview: {context[:300]}...")
            else:
                print("❌ NO CONTEXT RETRIEVED from semantic search!")
        else:
            print("❌ Cannot test semantic search - missing chunks or embeddings")
        
        # Step 5: Test context retriever
        print("\n🔍 STEP 5: Context Retriever Test")
        print("-" * 40)
        if chunks:
            context = qa_system.context_retriever.get_relevant_context(test_questions, chunks, embeddings)
            print(f"Final context: {len(context) if context else 0} characters")
            if context:
                print(f"Final context preview: {context[:300]}...")
            else:
                print("❌ CONTEXT RETRIEVER RETURNED EMPTY!")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnostic error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = diagnose_system()
    print("\n" + "=" * 60)
    if success:
        print("🔍 Diagnostic completed!")
    else:
        print("💥 Diagnostic failed!")

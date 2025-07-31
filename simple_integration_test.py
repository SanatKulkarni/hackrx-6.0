#!/usr/bin/env python3
"""
Simple Integration Test - Document Processing + Chunking + Embeddings
"""

import os
import sys

def test_document_processing():
    """Test document processing"""
    print("🧪 Testing Document Processing...")
    
    # Change to loader directory and import
    loader_dir = os.path.join(os.path.dirname(__file__), 'loader_&_extractor')
    os.chdir(loader_dir)
    sys.path.insert(0, loader_dir)
    
    from url_document_loader import process_document
    
    # Test URL processing
    hackathon_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    doc_result = process_document(hackathon_url, 'url')
    
    if doc_result['success']:
        print(f"✅ Document processed: {len(doc_result['text'])} characters")
        return doc_result['text']
    else:
        print(f"❌ Document processing failed: {doc_result.get('error')}")
        return None

def test_chunking(text):
    """Test text chunking"""
    print("\n🔪 Testing Text Chunking...")
    
    # Change to chunking directory and import
    chunking_dir = os.path.join(os.path.dirname(__file__), 'chunking')
    os.chdir(chunking_dir)
    sys.path.insert(0, chunking_dir)
    
    from main_chunking import chunk_text
    
    chunks = chunk_text(text, chunk_size=1500, chunk_overlap=300)
    
    if chunks:
        print(f"✅ Text chunked: {len(chunks)} chunks")
        print(f"   Average chunk size: {sum(len(chunk) for chunk in chunks) // len(chunks)} characters")
        return chunks
    else:
        print("❌ Chunking failed")
        return None

def test_embeddings(chunks):
    """Test embedding generation and storage"""
    print("\n🔮 Testing Embeddings...")
    
    # Change to root directory for embeddings import
    root_dir = os.path.dirname(__file__)
    os.chdir(root_dir)
    sys.path.insert(0, root_dir)
    
    try:
        from embeddings import embed_and_store_chunks
        
        # Store first 5 chunks for testing (to avoid hitting API limits)
        test_chunks = chunks[:5] if len(chunks) > 5 else chunks
        
        print(f"Processing {len(test_chunks)} chunks for embedding...")
        
        index = embed_and_store_chunks(
            test_chunks, 
            index_name="hackathon-test", 
            namespace="integration_test"
        )
        
        if index:
            print(f"✅ Embeddings generated and stored successfully!")
            return True
        else:
            print("❌ Embedding storage failed")
            return False
            
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

def main():
    """Main integration test"""
    print("🚀 INTEGRATION PIPELINE TEST")
    print("="*50)
    
    # Save original directory
    original_dir = os.getcwd()
    
    try:
        # Test 1: Document Processing
        text = test_document_processing()
        if not text:
            print("❌ Integration test failed at document processing")
            return
        
        # Test 2: Chunking
        chunks = test_chunking(text)
        if not chunks:
            print("❌ Integration test failed at chunking")
            return
        
        # Test 3: Embeddings
        success = test_embeddings(chunks)
        if not success:
            print("❌ Integration test failed at embeddings")
            return
        
        print("\n🎉 INTEGRATION TEST SUCCESSFUL!")
        print("="*50)
        print("✅ Document Processing: Working")
        print("✅ Text Chunking: Working") 
        print("✅ Embeddings & Storage: Working")
        print("\n🚀 Ready for Q&A system integration!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
    
    finally:
        # Restore original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main()

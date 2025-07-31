"""
Integrated Document Processing, Chunking, and Embedding Pipeline
Combines document loading (URL/local), chunking, and Pinecone storage
"""

import os
import sys
from pathlib import Path

# Add paths for imports
current_dir = os.path.dirname(__file__)
loader_dir = os.path.join(current_dir, 'loader_&_extractor')
chunking_dir = os.path.join(current_dir, 'chunking')
embeddings_dir = os.path.join(current_dir, 'embeddings')

sys.path.extend([loader_dir, chunking_dir, embeddings_dir])

# Import our modules
try:
    # Import from loader directory
    sys.path.append(loader_dir)
    import url_document_loader
    from url_document_loader import process_document
    
    # Import from chunking directory  
    sys.path.append(chunking_dir)
    from main_chunking import chunk_text
    
    # Import from embeddings directory
    sys.path.append(embeddings_dir)
    import sys
    # For embeddings, we need to import from the parent package
    sys.path.append(current_dir)
    from embeddings import embed_and_store_chunks
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)

def process_and_store_document(source, source_type='auto', index_name="policies", namespace="default", chunk_size=1000, chunk_overlap=200):
    """
    Complete pipeline: Process document, chunk text, generate embeddings, and store in Pinecone
    
    Args:
        source (str): URL or local file path
        source_type (str): 'url', 'local', or 'auto' 
        index_name (str): Pinecone index name
        namespace (str): Pinecone namespace
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
    
    Returns:
        dict: Results of the complete processing pipeline
    """
    
    print("🚀 Starting Complete Document Processing Pipeline")
    print("="*60)
    
    try:
        # Step 1: Document Processing
        print("📄 Step 1: Processing Document...")
        doc_result = process_document(source, source_type)
        
        if not doc_result['success']:
            return {
                'success': False,
                'error': f"Document processing failed: {doc_result.get('error', 'Unknown error')}",
                'stage': 'document_processing'
            }
        
        extracted_text = doc_result['text']
        print(f"✅ Document processed successfully!")
        print(f"   📊 Text length: {len(extracted_text)} characters")
        print(f"   📄 File type: {doc_result['file_type']}")
        
        # Step 2: Text Chunking
        print(f"\n🔪 Step 2: Chunking Text...")
        print(f"   Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        
        chunks = chunk_text(extracted_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        if not chunks:
            return {
                'success': False,
                'error': "Text chunking produced no chunks",
                'stage': 'chunking'
            }
        
        print(f"✅ Text chunked successfully!")
        print(f"   📊 Number of chunks: {len(chunks)}")
        print(f"   📝 Average chunk length: {sum(len(chunk) for chunk in chunks) // len(chunks)} characters")
        
        # Step 3: Embedding and Storage
        print(f"\n🔮 Step 3: Generating Embeddings and Storing in Pinecone...")
        print(f"   Index: {index_name}, Namespace: {namespace}")
        
        index = embed_and_store_chunks(chunks, index_name, namespace)
        
        if not index:
            return {
                'success': False,
                'error': "Embedding and storage failed",
                'stage': 'embedding_storage'
            }
        
        print(f"✅ Embeddings generated and stored successfully!")
        
        # Step 4: Prepare Results
        result = {
            'success': True,
            'document_info': {
                'source': source,
                'source_type': source_type,
                'file_type': doc_result['file_type'],
                'text_length': len(extracted_text)
            },
            'chunking_info': {
                'num_chunks': len(chunks),
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'avg_chunk_length': sum(len(chunk) for chunk in chunks) // len(chunks)
            },
            'storage_info': {
                'index_name': index_name,
                'namespace': namespace,
                'pinecone_index': index
            },
            'chunks': chunks[:5]  # Store first 5 chunks for preview
        }
        
        print(f"\n🎉 Complete Pipeline Successful!")
        print("="*60)
        print("📋 SUMMARY:")
        print(f"   📄 Document: {Path(source).name if not source.startswith('http') else 'URL Document'}")
        print(f"   📊 Text: {len(extracted_text):,} characters → {len(chunks)} chunks")
        print(f"   🔮 Embeddings: Generated and stored in Pinecone")
        print(f"   🏪 Storage: Index '{index_name}', Namespace '{namespace}'")
        print("="*60)
        
        return result
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        return {
            'success': False,
            'error': str(e),
            'stage': 'unknown'
        }

def process_multiple_documents(sources, index_name="policies", namespace="default", chunk_size=1000, chunk_overlap=200):
    """
    Process multiple documents through the complete pipeline
    
    Args:
        sources (list): List of document sources (URLs or file paths)
        index_name (str): Pinecone index name
        namespace (str): Pinecone namespace
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
    
    Returns:
        dict: Results for all processed documents
    """
    
    print(f"🚀 Processing {len(sources)} Documents")
    print("="*60)
    
    results = {
        'successful': [],
        'failed': [],
        'total_chunks': 0,
        'total_characters': 0
    }
    
    for i, source in enumerate(sources, 1):
        print(f"\n📄 Processing Document {i}/{len(sources)}: {Path(source).name if not source.startswith('http') else 'URL Document'}")
        print("-" * 50)
        
        result = process_and_store_document(
            source=source,
            index_name=index_name,
            namespace=namespace,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if result['success']:
            results['successful'].append(result)
            results['total_chunks'] += result['chunking_info']['num_chunks']
            results['total_characters'] += result['document_info']['text_length']
        else:
            results['failed'].append({
                'source': source,
                'error': result['error'],
                'stage': result['stage']
            })
    
    # Summary
    print(f"\n🎉 BATCH PROCESSING COMPLETE!")
    print("="*60)
    print("📊 BATCH SUMMARY:")
    print(f"   ✅ Successful: {len(results['successful'])}")
    print(f"   ❌ Failed: {len(results['failed'])}")
    print(f"   📊 Total chunks: {results['total_chunks']:,}")
    print(f"   📄 Total characters: {results['total_characters']:,}")
    print("="*60)
    
    if results['failed']:
        print("\n❌ FAILED DOCUMENTS:")
        for failure in results['failed']:
            print(f"   - {Path(failure['source']).name}: {failure['error']}")
    
    return results

def test_integration_pipeline():
    """Test the complete integration pipeline"""
    
    print("🧪 TESTING COMPLETE INTEGRATION PIPELINE")
    print("="*80)
    
    # Test 1: Process hackathon sample URL
    hackathon_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    print("\n🔗 Test 1: Hackathon Sample URL")
    print("-" * 50)
    
    url_result = process_and_store_document(
        source=hackathon_url,
        source_type='url',
        index_name="hackathon_policies",
        namespace="test_documents",
        chunk_size=1500,  # Larger chunks for policy documents
        chunk_overlap=300
    )
    
    if url_result['success']:
        print("✅ URL processing pipeline successful!")
        print(f"   Stored {url_result['chunking_info']['num_chunks']} chunks in Pinecone")
    else:
        print(f"❌ URL processing failed: {url_result['error']}")
    
    # Test 2: Process local files if available
    local_test_files = [
        "dataset/pdf-format/1.pdf",
        "dataset/word-format/AI-helper-low-wage.docx"
    ]
    
    available_files = [f for f in local_test_files if os.path.exists(f)]
    
    if available_files:
        print(f"\n📁 Test 2: Local Files ({len(available_files)} available)")
        print("-" * 50)
        
        batch_result = process_multiple_documents(
            sources=available_files,
            index_name="hackathon_policies", 
            namespace="local_documents",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        print(f"✅ Batch processing complete!")
        print(f"   Successful: {len(batch_result['successful'])}")
        print(f"   Failed: {len(batch_result['failed'])}")
    else:
        print("\n📁 Test 2: No local files available for testing")
    
    print(f"\n🎉 INTEGRATION PIPELINE TESTING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    test_integration_pipeline()

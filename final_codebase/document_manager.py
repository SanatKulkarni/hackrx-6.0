"""
Document Manager
Handles document processing, storage, and chunk management
"""

import hashlib
from typing import List, Dict, Any
from .config import Config

class DocumentManager:
    """Manages document processing and chunk storage"""
    
    def __init__(self, process_document, chunk_text, embed_and_store_chunks, query_embeddings):
        self.process_document = process_document
        self.chunk_text = chunk_text
        self.embed_and_store_chunks = embed_and_store_chunks
        self.query_embeddings = query_embeddings
        
        self.document_chunks = []
        self.chunks_loaded = False
    
    def check_document_exists(self, document_url: str, index_name: str, namespace: str) -> bool:
        """
        Check if document already exists in the vector database
        
        Args:
            document_url (str): URL of the document to check
            index_name (str): Pinecone index name
            namespace (str): Pinecone namespace
            
        Returns:
            bool: True if document exists, False otherwise
        """
        try:
            # Create a unique identifier for the document based on URL
            doc_id = hashlib.md5(document_url.encode()).hexdigest()
            
            # Fast existence check: Use minimal query with very low n_results  
            test_results = self.query_embeddings(
                query_text="policy",  # Simple, short query
                index_name=index_name,
                namespace=namespace,
                n_results=1,  # Just need one result to confirm existence
                filter={"document_id": doc_id}  # Filter by document ID
            )
            
            # If we got any matches with this document_id, the document exists
            exists = test_results and 'matches' in test_results and len(test_results['matches']) > 0
            
            if exists:
                print(f"📋 Document ID {doc_id[:8]}... found in vector database")
            else:
                print(f"🆕 Document ID {doc_id[:8]}... not found - new document")
                
            return exists
            
        except Exception as e:
            print(f"⚠️  Warning: Could not check document existence: {e}")
            # If we can't check, assume it doesn't exist to be safe
            return False
    
    def process_and_store_document(self, document_url: str, index_name: str, namespace: str) -> bool:
        """
        Process document from URL and store in Pinecone (only if not already processed)
        
        Args:
            document_url (str): URL of the document to process
            index_name (str): Pinecone index name
            namespace (str): Pinecone namespace
            
        Returns:
            bool: Success status
        """
        try:
            print(f"📄 Processing document from URL...")
            
            # Check if document already exists
            if self.check_document_exists(document_url, index_name, namespace):
                print(f"✅ Document already exists in vector database - loading existing chunks")
                print(f"🚀 Using cached document embeddings for faster response")
                
                # Load existing chunks from Pinecone for in-memory operations
                success = self._load_existing_chunks_from_pinecone(document_url, index_name, namespace)
                if success:
                    print(f"📚 Loaded {len(self.document_chunks)} chunks from vector database")
                    return True
                else:
                    print("⚠️  Failed to load existing chunks - will reprocess document")
                    # Fall through to reprocess the document
            
            print(f"📥 New document detected - processing and storing...")
            
            # Step 1: Process document
            doc_result = self.process_document(document_url, 'url')
            if not doc_result['success']:
                print(f"❌ Document processing failed: {doc_result.get('error')}")
                return False
            
            text = doc_result['text']
            print(f"✅ Document processed: {len(text)} characters")
            
            # Step 2: Chunk text
            print(f"🔪 Chunking text...")
            chunks = self.chunk_text(text, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
            if not chunks:
                print("❌ Text chunking failed")
                return False
            
            print(f"✅ Text chunked: {len(chunks)} chunks")
            
            # Step 3: Add document metadata to chunks
            doc_id = hashlib.md5(document_url.encode()).hexdigest()
            
            # Add metadata to each chunk
            enriched_chunks = []
            for i, chunk in enumerate(chunks):
                enriched_chunk = {
                    'text': chunk,
                    'document_id': doc_id,
                    'document_url': document_url,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                enriched_chunks.append(enriched_chunk)
            
            # Step 4: Generate embeddings and store with metadata
            print(f"🔮 Generating embeddings and storing...")
            index = self.embed_and_store_chunks(enriched_chunks, index_name, namespace)
            if not index:
                print("❌ Embedding and storage failed")
                return False
            
            print(f"✅ Document processed and stored successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error processing document: {e}")
            return False
    
    def load_document_chunks(self, index_name: str, namespace: str) -> bool:
        """
        Load all document chunks into memory for fast retrieval
        This replaces slow Pinecone queries with in-memory search
        """
        try:
            if self.chunks_loaded:
                return True
                
            print("📚 Loading document chunks for fast retrieval...")
            
            # Get all chunks from Pinecone once
            results = self.query_embeddings(
                query_text="policy insurance coverage",  # Generic query to get all chunks
                index_name=index_name,
                namespace=namespace,
                n_results=Config.MAX_CHUNKS_TO_RETRIEVE
            )
            
            if not results or 'matches' not in results:
                print("⚠️  No document chunks found in database")
                return False
            
            # Store chunks with metadata for fast search
            self.document_chunks = []
            chunk_texts = []
            
            for match in results['matches']:
                if 'metadata' in match and 'text' in match['metadata']:
                    chunk_text = match['metadata']['text']
                    chunk_data = {
                        'text': chunk_text,
                        'text_lower': chunk_text.lower(),  # For fast searching
                        'score': match.get('score', 0.0)
                    }
                    self.document_chunks.append(chunk_data)
                    chunk_texts.append(chunk_text)
            
            self.chunks_loaded = True
            print(f"✅ Loaded {len(self.document_chunks)} document chunks into memory")
            return True, chunk_texts
            
        except Exception as e:
            print(f"❌ Error loading document chunks: {e}")
            return False, []
    
    def get_chunk_texts(self) -> List[str]:
        """Get list of chunk texts"""
        return [chunk['text'] for chunk in self.document_chunks]
    
    def get_document_chunks(self) -> List[Dict[str, Any]]:
        """Get document chunks"""
        return self.document_chunks
    
    def _load_existing_chunks_from_pinecone(self, document_url: str, index_name: str, namespace: str) -> bool:
        """
        Load existing document chunks from Pinecone vector database
        
        Args:
            document_url (str): URL of the document
            index_name (str): Pinecone index name  
            namespace (str): Pinecone namespace
            
        Returns:
            bool: Success status
        """
        try:
            # Generate document ID
            document_id = self._generate_document_id(document_url)
            
            # Query Pinecone for all chunks of this document using text query approach
            query_result = self.query_embeddings(
                query_text="policy insurance coverage",  # Generic query
                index_name=index_name,
                namespace=namespace, 
                n_results=Config.MAX_CHUNKS_TO_RETRIEVE,
                filter={"document_id": document_id}
            )
            
            if not query_result or 'matches' not in query_result:
                return False
            
            # Convert Pinecone results back to chunk format
            self.document_chunks = []
            for match in query_result['matches']:
                if 'metadata' in match and 'text' in match['metadata']:
                    chunk = {
                        'text': match['metadata']['text'],
                        'text_lower': match['metadata']['text'].lower(),
                        'chunk_id': match['id'],
                        'document_id': document_id
                    }
                    self.document_chunks.append(chunk)
            
            print(f"📥 Successfully loaded {len(self.document_chunks)} chunks from Pinecone")
            return True
            
        except Exception as e:
            print(f"❌ Error loading existing chunks: {e}")
            return False
    
    def _generate_document_id(self, document_url: str) -> str:
        """Generate a unique document ID from URL"""
        return hashlib.md5(document_url.encode()).hexdigest()

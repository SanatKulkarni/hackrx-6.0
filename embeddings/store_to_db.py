import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from multiple locations
load_dotenv()  # Load from current directory
root_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=root_env_path)  # Load from root directory

def store_embeddings_to_db(chunks, embeddings, index_name="policies", namespace="default"):
    """
    Store text chunks and their embeddings to Pinecone

    Args:
        chunks (list): List of enriched chunks (dicts with 'text' and metadata) or simple text chunks
        embeddings (list): List of embedding vectors corresponding to chunks
        index_name (str): Name of the Pinecone index
        namespace (str): Namespace within the index

    Returns:
        index: The Pinecone index object
    """
    
    # Initialize Pinecone client
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists, create if it doesn't
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")
        # Create index with dimension based on embeddings (assuming 768 for Nomic)
        dimension = len(embeddings[0]) if embeddings else 768
        
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created new index: {index_name}")
    else:
        print(f"Using existing index: {index_name}")
    
    # Get the index
    index = pc.Index(index_name)
    
    # Prepare vectors for upsert
    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Handle both enriched chunks (dicts) and simple text chunks
        if isinstance(chunk, dict):
            # Enriched chunk with metadata
            text = chunk['text']
            metadata = {key: value for key, value in chunk.items() if key != 'text'}
            metadata['text'] = text  # Ensure text is in metadata
            metadata['chunk_index'] = i  # Add index for compatibility
            
            # Create unique vector ID using document_id if available
            if 'document_id' in chunk:
                vector_id = f"{chunk['document_id']}_chunk_{i}"
            else:
                vector_id = f"chunk_{i}"
        else:
            # Simple text chunk
            text = chunk
            metadata = {
                "text": text,
                "chunk_index": i
            }
            vector_id = f"chunk_{i}"
        
        vectors_to_upsert.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })
    
    # Upsert vectors in batches (Pinecone recommends batch size of 100)
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
    
    print(f"Successfully stored {len(chunks)} chunks to Pinecone index '{index_name}' in namespace '{namespace}'")
    return index

def query_collection(query_text, query_embedding, index_name="policies", namespace="default", n_results=5, filter=None):
    """
    Query the Pinecone index

    Args:
        query_text (str): The query text (for metadata purposes)
        query_embedding (list): The embedding vector for the query
        index_name (str): Name of the index to query
        namespace (str): Namespace within the index
        n_results (int): Number of results to return
        filter (dict): Optional filter for metadata

    Returns:
        dict: Query results
    """
    # Initialize Pinecone client
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Query the index
    query_params = {
        "vector": query_embedding,
        "top_k": n_results,
        "namespace": namespace,
        "include_metadata": True,
        "include_values": False
    }
    
    if filter:
        query_params["filter"] = filter
    
    results = index.query(**query_params)
    
    return results
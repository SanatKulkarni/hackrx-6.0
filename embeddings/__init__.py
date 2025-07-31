from langchain_nomic import NomicEmbeddings
from dotenv import load_dotenv
import os
import getpass
from .store_to_db import store_embeddings_to_db, query_collection

# Load environment variables from the root .env file
root_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=root_env_path)

# Also try loading from current directory
load_dotenv()

if not os.getenv("NOMIC_API_KEY"):
    print("Nomic API key not found. Please get one from https://atlas.nomic.ai/")
    os.environ["NOMIC_API_KEY"] = getpass.getpass("Enter your Nomic API key: ")

embeddings = NomicEmbeddings(
    model="nomic-embed-text-v1.5",
    dimensionality=768,  
    inference_mode="remote",  
)

def embed_chunks(chunks):
    """
    Generate embeddings for a list of text chunks using Nomic embeddings.

    Args:
        chunks (list): List of text chunks to embed

    Returns:
        list: List of embedding vectors
    """
    embedded_vectors = embeddings.embed_documents(chunks)
    print(f"{len(embedded_vectors)} embeddings generated")
    return embedded_vectors

def embed_and_store_chunks(chunks, index_name="policies", namespace="default"):
    """
    Generate embeddings for text chunks and store them in Pinecone.
    Handles both simple text chunks and enriched chunks with metadata.

    Args:
        chunks (list): List of text chunks (strings) or enriched chunks (dicts) to embed and store
        index_name (str): Name of the Pinecone index
        namespace (str): Namespace within the index

    Returns:
        index: The Pinecone index object
    """
    # Handle both simple text chunks and enriched chunks with metadata
    if chunks and isinstance(chunks[0], dict):
        # Extract text for embedding generation
        text_chunks = [chunk['text'] for chunk in chunks]
        enriched_chunks = chunks
    else:
        # Simple text chunks - convert to enriched format
        text_chunks = chunks
        enriched_chunks = [{'text': chunk} for chunk in chunks]
    
    embedded_vectors = embed_chunks(text_chunks)
    index = store_embeddings_to_db(enriched_chunks, embedded_vectors, index_name, namespace)
    return index

def query_embeddings(query_text, index_name="policies", namespace="default", n_results=5, filter=None):
    """
    Query embeddings from Pinecone using a text query.

    Args:
        query_text (str): The query text
        index_name (str): Name of the Pinecone index
        namespace (str): Namespace within the index
        n_results (int): Number of results to return
        filter (dict): Optional filter for metadata

    Returns:
        dict: Query results from Pinecone
    """
    # Generate embedding for the query
    query_embedding = embeddings.embed_query(query_text)
    
    # Query Pinecone
    results = query_collection(query_text, query_embedding, index_name, namespace, n_results, filter)
    return results
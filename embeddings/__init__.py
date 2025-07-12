from langchain_nomic import NomicEmbeddings
from dotenv import load_dotenv
import os
import getpass

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

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
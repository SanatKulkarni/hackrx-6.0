import chromadb
import os

def store_embeddings_to_db(chunks, embeddings, collection_name="policies", db_path="./database"):
    """
    Store text chunks and their embeddings to ChromaDB

    Args:
        chunks (list): List of text chunks
        embeddings (list): List of embedding vectors corresponding to chunks
        collection_name (str): Name of the ChromaDB collection
        db_path (str): Path where to store the ChromaDB database

    Returns:
        collection: The ChromaDB collection object
    """

    os.makedirs(db_path, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=db_path)

    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except:
        collection = chroma_client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")

    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings
    )

    print(f"Successfully stored {len(chunks)} chunks to ChromaDB collection '{collection_name}' in {db_path}")
    return collection

def query_collection(query_text, collection_name="policies", n_results=5, db_path="./database"):
    """
    Query the ChromaDB collection

    Args:
        query_text (str): The query text
        collection_name (str): Name of the collection to query
        n_results (int): Number of results to return
        db_path (str): Path where the ChromaDB database is stored

    Returns:
        dict: Query results
    """
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(name=collection_name)

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results
import os
import sys
import subprocess


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import embed_chunks
from embeddings.store_to_db import store_embeddings_to_db, query_collection

def embedding_query(query: str):
    """
    Create embeddings for query and search in the database
    
    Args:
        query (str): The query text to embed and search
        
    Returns:
        dict: Query results from ChromaDB
    """
    try:

        print(f"Creating embedding for query: {query}")
        query_embeddings = embed_chunks([query])  
        
        print("Searching database for similar content...")
        db_path = "../database"
        collection_name = "policy_documents"
        
        results = query_collection(
            query_text=query,
            collection_name=collection_name,
            n_results=5,
            db_path=db_path
        )
        
        print(f"Found {len(results.get('documents', [[]])[0])} relevant documents")
        return results
        
    except Exception as e:
        print(f"Error in embedding_query: {e}")
        return embedding_query_subprocess(query)

def embedding_query_subprocess(query: str):
    """
    Fallback method using subprocess to call embedding functions
    
    Args:
        query (str): The query text to embed and search
        
    Returns:
        dict: Results from subprocess execution
    """
    try:
        script_content = f'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import embed_chunks
from embeddings.store_to_db import query_collection
import json

query = "{query}"
try:
    # Create embeddings
    embeddings = embed_chunks([query])
    
    # Query database
    results = query_collection(
        query_text=query,
        collection_name="policy_documents",
        n_results=5,
        db_path="../database"
    )
    
    print(json.dumps(results))
except Exception as e:
    print(f"Error: {{e}}")
'''
        
        temp_script = "temp_embedding_query.py"
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        result = subprocess.run(
            [sys.executable, temp_script],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if os.path.exists(temp_script):
            os.remove(temp_script)
        
        if result.returncode == 0:
            try:
                return eval(result.stdout.strip())  
            except:
                print(f"Subprocess output: {result.stdout}")
                return {"documents": [[]], "distances": [[]], "ids": [[]]}
        else:
            print(f"Subprocess error: {result.stderr}")
            return {"documents": [[]], "distances": [[]], "ids": [[]]}
            
    except Exception as e:
        print(f"Error in subprocess method: {e}")
        return {"documents": [[]], "distances": [[]], "ids": [[]]}

def store_query_embeddings(text_chunks: list, collection_name: str = "policy_documents"):
    """
    Store new text chunks with embeddings to database using function calls
    
    Args:
        text_chunks (list): List of text chunks to store
        collection_name (str): Name of the collection to store in
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Creating embeddings for {len(text_chunks)} chunks...")
        
        embeddings = embed_chunks(text_chunks)
        
        print("Storing embeddings to database...")
        collection = store_embeddings_to_db(
            chunks=text_chunks,
            embeddings=embeddings,
            collection_name=collection_name,
            db_path="../database"
        )
        
        print("Successfully stored embeddings to database!")
        return True
        
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return store_embeddings_subprocess(text_chunks, collection_name)

def store_embeddings_subprocess(text_chunks: list, collection_name: str):
    """
    Fallback method using subprocess to store embeddings
    
    Args:
        text_chunks (list): List of text chunks to store
        collection_name (str): Name of the collection
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        script_content = f'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import embed_chunks
from embeddings.store_to_db import store_embeddings_to_db

text_chunks = {text_chunks}
collection_name = "{collection_name}"

try:
    # Create embeddings
    embeddings = embed_chunks(text_chunks)
    
    # Store to database
    store_embeddings_to_db(
        chunks=text_chunks,
        embeddings=embeddings,
        collection_name=collection_name,
        db_path="../database"
    )
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
'''
        
        temp_script = "temp_store_embeddings.py"
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        result = subprocess.run(
            [sys.executable, temp_script],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        
        if os.path.exists(temp_script):
            os.remove(temp_script)
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            return True
        else:
            print(f"Subprocess error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error in subprocess store method: {e}")
        return False

def main():
    print("=== Embedding Query System ===")
    print("1. Query existing database")
    print("2. Store new content to database")
    
    choice = input("Choose option (1 or 2, press Enter for option 1): ").strip()
    
    if choice == "2":
        print("\n--- Store New Content ---")
        content = input("Enter text content to store (or press Enter for sample): ").strip()
        if not content:
            content = "Sample insurance policy for knee surgery coverage in Pune for 46-year-old male patients"
        
        chunks = [chunk.strip() for chunk in content.split('.') if chunk.strip()]
        
        success = store_query_embeddings(chunks, "policy_documents")
        if success:
            print("✅ Content successfully stored to database!")
        else:
            print("❌ Failed to store content to database")
    
    else:
        
        print("\n--- Query Database ---")
        default_query = "46M, knee surgery, Pune, 3-month policy"
        user_query = input("Enter your query (press Enter to use default): ").strip()
        if not user_query:
            user_query = default_query
        
        print(f"\nProcessing query: {user_query}")
        results = embedding_query(user_query)
        
        if results and results.get('documents') and results['documents'][0]:
            print("\n=== Search Results ===")
            for i, (doc, distance) in enumerate(zip(results['documents'][0], results.get('distances', [[]])[0] if results.get('distances') else [])):
                print(f"\nResult {i+1}:")
                print(f"Content: {doc[:200]}...")
                if distance is not None:
                    print(f"Similarity Score: {1 - distance:.3f}")
                print("-" * 50)
        else:
            print("No results found or database may be empty.")
            print("Try storing some content first using option 2.")

if __name__ == "__main__":
    main()
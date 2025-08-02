"""
Module Importer
Handles importing required modules with error handling
"""

import os
import sys

def import_modules():
    """Import required modules with error handling"""
    try:
        # Store original directory
        original_dir = os.getcwd()
        
        # Get parent directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Define module directories
        loader_dir = os.path.join(parent_dir, 'loader_&_extractor')
        chunking_dir = os.path.join(parent_dir, 'chunking')
        embeddings_dir = os.path.join(parent_dir, 'embeddings')
        
        # Change to loader directory for import
        os.chdir(loader_dir)
        sys.path.insert(0, loader_dir)
        from url_document_loader import process_document
        
        # Change to chunking directory
        os.chdir(chunking_dir) 
        sys.path.insert(0, chunking_dir)
        from main_chunking import chunk_text
        
        # Change back to root for embeddings
        os.chdir(original_dir)
        sys.path.insert(0, parent_dir)
        from embeddings import embed_and_store_chunks, query_embeddings
        
        print("✅ All modules imported successfully")
        
        return process_document, chunk_text, embed_and_store_chunks, query_embeddings
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_dir)

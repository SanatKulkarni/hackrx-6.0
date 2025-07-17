import subprocess
import sys
import os
from chunking.main_chunking import chunk_text
from embeddings import embed_chunks
from embeddings.store_to_db import store_embeddings_to_db

file_path = os.path.join("dataset", "pdf-format", "1.pdf")
main_extractor_path = os.path.join("loader_&_extractor", "main_extractor")

#Extract text from pdf/word/text files 
print("Step 1: Extracting text from file...")
result = subprocess.run(
    [sys.executable, main_extractor_path, file_path],
    capture_output=True, text=True, encoding='utf-8', errors='replace'
)

if result.returncode != 0:
    print("Failed to extract text from file.")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    sys.exit(1)

lines = result.stdout.splitlines()
content_lines = []
for line in lines:
    if line.strip() and not (line.lower().startswith("processing ") or line.lower().startswith("detected file type") or line.lower().startswith("successfully processed") or line.lower().startswith("error") or line.startswith("=")):
        content_lines.append(line)
extracted_text = "\n".join(content_lines)

print("Step 2: Chunking text...")
chunks = chunk_text(extracted_text)
print(f"Created {len(chunks)} chunks")

print("Step 3: Generating embeddings...")
embeddings = embed_chunks(chunks)
print(f"Generated {len(embeddings)} embeddings")

print("Step 4: Storing to ChromaDB...")
collection = store_embeddings_to_db(chunks, embeddings, collection_name="policy_documents", db_path="database")

print("\n=== Pipeline Complete ===")
print(f"Successfully processed {len(chunks)} chunks and stored in ChromaDB")
print("Sample chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n{chunk[:200]}...\n")

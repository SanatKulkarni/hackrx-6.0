# Document Processing Pipeline

A comprehensive document processing pipeline that extracts text from various file formats, chunks the content, generates embeddings, and stores them in a vector database for efficient semantic search and retrieval.

## 🚀 Features

- **Multi-format Document Support**
  - PDF documents
  - Microsoft Word documents (.docx)
  - Plain text files
  - Handwritten text extraction from images using OCR

- **Advanced Text Processing**
  - Intelligent text chunking with configurable overlap
  - High-quality embedding generation using Nomic embeddings
  - Vector storage in ChromaDB for fast semantic search

- **Scalable Architecture**
  - Modular design with separate components for each processing stage
  - Asynchronous processing capabilities
  - Comprehensive test coverage

## 📋 Requirements

- Python 3.8+
- Nomic API key (for embeddings)
- Google Cloud Vision API credentials (for handwriting extraction)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SanatKulkarni/hackrx-6.0.git
   cd hackrx-6.0
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   NOMIC_API_KEY=your_nomic_api_key_here
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google-credentials.json
   ```

## 📦 Dependencies

```txt
langchain
langchain-community
langchain-nomic
chromadb
PyMuPDF
Spire.Doc.Python
Pillow
python-dotenv
opencv-python
google-cloud-vision
numpy
```

## 🏗️ Project Structure

```
hackrx-6.0/
├── main.py                     # Main pipeline orchestrator
├── loader_&_extractor/         # Document loading and text extraction
│   ├── main_extractor.py       # Main extraction coordinator
│   ├── pdfLoading.py          # PDF text extraction
│   ├── wordLoading.py         # Word document processing
│   ├── textLoading.py         # Plain text file handling
│   ├── handwriting_text_extraction.py  # OCR for handwritten text
│   └── *_test.py              # Unit tests for each component
├── chunking/                   # Text chunking functionality
│   ├── main_chunking.py       # Text splitting logic
│   └── main_chunking_test.py  # Chunking tests
├── embeddings/                 # Embedding generation and storage
│   ├── __init__.py            # Nomic embeddings setup
│   └── store_to_db.py         # ChromaDB operations
└── database/                   # ChromaDB storage directory
```

## 🚀 Usage

### Basic Pipeline Execution

```bash
python main.py
```

This will process the default file located at `dataset/pdf-format/1.pdf` through the complete pipeline:

1. **Text Extraction** - Extracts text from the input document
2. **Text Chunking** - Splits text into manageable chunks
3. **Embedding Generation** - Creates vector embeddings using Nomic
4. **Database Storage** - Stores embeddings in ChromaDB

### Custom File Processing

```python
import subprocess
import sys
from chunking.main_chunking import chunk_text
from embeddings import embed_chunks
from embeddings.store_to_db import store_embeddings_to_db

# Extract text from your file
file_path = "path/to/your/document.pdf"
result = subprocess.run(
    [sys.executable, "loader_&_extractor/main_extractor.py", file_path],
    capture_output=True, text=True
)

# Process the extracted text
extracted_text = result.stdout
chunks = chunk_text(extracted_text, chunk_size=1000, chunk_overlap=200)
embeddings = embed_chunks(chunks)

# Store in database
collection = store_embeddings_to_db(
    chunks, 
    embeddings, 
    collection_name="my_documents",
    db_path="./my_database"
)
```

### Querying the Database

```python
from embeddings.store_to_db import query_collection

# Search for relevant documents
results = query_collection(
    query_text="What is the policy on remote work?",
    collection_name="policy_documents",
    n_results=5
)

for i, doc in enumerate(results['documents'][0]):
    print(f"Result {i+1}: {doc[:200]}...")
```

## 🔧 Configuration

### Chunking Parameters

Adjust text chunking behavior in `chunking/main_chunking.py`:

```python
chunks = chunk_text(
    text,
    chunk_size=1000,        # Maximum characters per chunk
    chunk_overlap=200,      # Overlap between chunks
    separators=None         # Custom separators (optional)
)
```

### Embedding Configuration

Modify embedding settings in `embeddings/__init__.py`:

```python
embeddings = NomicEmbeddings(
    model="nomic-embed-text-v1.5",
    dimensionality=768,
    inference_mode="remote"
)
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m unittest discover -s . -p "*_test.py" -v

# Run specific component tests
python -m unittest loader_&_extractor.pdfLoading_test
python -m unittest chunking.main_chunking_test
```

## 📝 API Keys Setup

### Nomic API Key
1. Visit [Nomic Atlas](https://atlas.nomic.ai/)
2. Create an account and generate an API key
3. Add to your `.env` file or set as environment variable

### Google Cloud Vision API (for handwriting extraction)
1. Create a Google Cloud project
2. Enable the Vision API
3. Create a service account and download credentials
4. Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of HackRX 6.0 hackathon submission.

## 🆘 Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

**API Key Errors**: Verify your environment variables are set correctly
```bash
echo $NOMIC_API_KEY
echo $GOOGLE_APPLICATION_CREDENTIALS
```

**Database Errors**: Check ChromaDB permissions and disk space
```bash
ls -la database/
```

### Performance Tips

- Use smaller chunk sizes for more granular search
- Increase chunk overlap for better context preservation
- Consider batch processing for large document collections

## 📊 Example Output

```
Step 1: Extracting text from file...
Step 2: Chunking text...
Created 45 chunks
Step 3: Generating embeddings...
Generated 45 embeddings
Step 4: Storing to ChromaDB...
Successfully stored 45 chunks to ChromaDB collection 'policy_documents' in database

=== Pipeline Complete ===
Successfully processed 45 chunks and stored in ChromaDB
Sample chunks:
--- Chunk 1 ---
This document outlines the company's remote work policy...

--- Chunk 2 ---
Employees are eligible for remote work arrangements...
```
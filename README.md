# HackRx 6.0

An intelligent document analysis system using Large Language Models (LLMs) for processing natural language queries against unstructured documents such as policies, contracts, and emails.

## Overview

**HackRx 6.0** enables natural language understanding over complex enterprise documents. The system accepts plain English queries, identifies relevant information from unstructured document sources, and returns structured, explainable outputs with semantic traceability. It is designed to support tasks such as claim adjudication, contract compliance, policy analysis, and HR query automation.

## Features

- **Natural Language Query Parsing**: Understands vague, incomplete, or informal user queries  
- **Multi-format Document Processing**: Supports PDF, Word, and plain text files  
- **Semantic Retrieval**: Retrieves relevant clauses using vector-based semantic search  
- **LLM-Powered Analysis**: Interprets and evaluates clauses using Google Gemini  
- **Structured, Explainable Output**: Returns decision, amount (if applicable), and justification tied to document sources  
- **OCR for Handwritten Text**: Recognizes content in scanned forms using Google Vision API  
- **Persistent Embedding Storage**: Uses ChromaDB for fast document retrieval  
- **Cross-domain Applicability**: Usable across insurance, legal, HR, and compliance workflows  

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hackrx-6.0
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   NOMIC_API_KEY=your_nomic_api_key
   GOOGLE_APPLICATION_CREDENTIALS=path/to/google-vision-credentials.json
   ```

## Configuration

### Chunking Parameters
```python
chunk_size = 1000
chunk_overlap = 200
```

### Embedding Configuration
```python
model = "nomic-embed-text-v1.5"
dimensionality = 768
inference_mode = "remote"
```

### Retrieval Settings
```python
k = 5
search_type = "similarity"
```

## Supported Inputs

| Format     | Extensions         | Handler                     |
|------------|--------------------|-----------------------------|
| PDF        | `.pdf`             | PyPDFLoader + OCR fallback |
| Word       | `.docx`, `.doc`    | python-docx                |
| Text       | `.txt`, `.text`    | Built-in file reader       |
| Handwritten | `.pdf` (scanned)  | Google Vision API          |

## System Capabilities

- Parses age, gender, procedure, location, and policy details from natural language  
- Retrieves semantically related clauses even for vague or loosely worded queries  
- Justifies each decision step with exact document clause references  
- Works with scanned handwritten documents via OCR  
- Can integrate with downstream claim or compliance workflows  

## Applications

- **Insurance**: Claim eligibility and benefit decisions  
- **Legal Compliance**: Clause matching and obligation detection  
- **Human Resources**: Querying policy handbooks and contracts  
- **Contract Management**: Clause analysis and deviation detection  

## System Requirements

- **Python**: 3.8+  
- **Memory**: 4GB minimum (8GB recommended)  
- **Storage**: 2GB+ for vector DB  
- **Network**: Required for LLM and OCR API usage  

## Acknowledgments

- **LangChain**: For RAG pipeline  
- **ChromaDB**: For vector storage  
- **Nomic**: For semantic embeddings  
- **Google Gemini & Vision APIs**: For LLM inference and OCR  
- **PyMuPDF**: For PDF extraction  

## Support

For issues and questions:
- Open an issue on the GitHub repository  
- Validate `.env` configuration and API key setup  
- Review supported file types and formatting requirements
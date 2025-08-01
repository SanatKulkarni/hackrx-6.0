# Hackathon Q&A System - Final Codebase

A modular, maintainable Question & Answer system using GitHub AI and advanced semantic search.

## Architecture

The system is organized into modular components for better maintainability:

### Core Components

1. **`config.py`** - Configuration and environment setup
2. **`module_importer.py`** - Handles importing required modules with error handling
3. **`github_ai_client.py`** - GitHub AI integration for answer generation
4. **`semantic_search.py`** - Semantic similarity search using sentence transformers
5. **`query_generator.py`** - Enhanced topic query generation with synonyms
6. **`document_manager.py`** - Document processing, storage, and chunk management
7. **`context_retriever.py`** - Context retrieval using semantic and keyword search
8. **`hackathon_qa_system.py`** - Main system that orchestrates all components

## Features

- **GitHub AI Integration**: Uses `openai/gpt-4o-mini` for high-quality answer generation
- **Advanced Semantic Search**: Sentence transformers with cosine similarity
- **Smart Context Retrieval**: Hybrid semantic + keyword search approach
- **Document Caching**: MD5-based document existence checking
- **Enhanced Query Generation**: Dynamic synonym expansion and domain-specific terms
- **Modular Architecture**: Clean separation of concerns for maintainability

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables** (create `.env` file):
   ```
   GITHUB_TOKEN=your_github_token_here
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=us-east-1
   PINECONE_INDEX_NAME=hackathon-policies
   NOMIC_API_KEY=your_nomic_api_key (optional)
   ```

3. **Run the System**:
   ```bash
   python hackathon_qa_system.py
   ```

## Usage

### Input Format
```json
{
  "documents": "URL_to_document",
  "questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?"
  ]
}
```

### Output Format
```json
{
  "answers": [
    "Answer to question 1 with specific details and numbers.",
    "Answer to question 2 with policy references.",
    "Answer to question 3 with exact conditions."
  ]
}
```

## Component Details

### Configuration (`config.py`)
- Centralizes all configuration settings
- Validates required environment variables
- Provides default values for system parameters

### GitHub AI Client (`github_ai_client.py`)
- Handles GitHub AI model communication
- Implements sophisticated answer parsing
- Provides consistent plain-text formatting

### Semantic Search (`semantic_search.py`)
- Uses `all-MiniLM-L6-v2` sentence transformer model
- Implements cosine similarity for semantic matching
- Includes keyword boosting for domain-specific terms

### Query Generator (`query_generator.py`)
- Generates enhanced topic queries from questions
- Includes comprehensive medical/insurance synonym dictionary
- Creates word combinations and context expansions

### Document Manager (`document_manager.py`)
- Handles document processing and storage
- Implements intelligent document caching
- Manages chunk loading and metadata

### Context Retriever (`context_retriever.py`)
- Combines semantic and keyword-based search
- Implements advanced scoring algorithms
- Provides fallback search mechanisms

## Performance

- **Document Caching**: Skips reprocessing of existing documents
- **In-Memory Search**: Fast chunk retrieval without repeated database queries
- **Semantic Embeddings**: Precomputed embeddings for instant similarity calculations
- **Hybrid Search**: Combines best of semantic and keyword approaches

## Testing

The system includes comprehensive testing with sample insurance policy questions:

- Grace period queries
- Waiting period questions
- Coverage condition inquiries
- Medical procedure waiting times
- Specific benefit coverage questions

## Maintainability

The modular architecture provides:

- **Clear Separation**: Each component has a single responsibility
- **Easy Testing**: Components can be tested independently
- **Simple Updates**: Changes to one component don't affect others
- **Readable Code**: Well-documented and organized structure

## Dependencies

Key dependencies include:
- `azure-ai-inference`: GitHub AI model access
- `sentence-transformers`: Semantic similarity
- `scikit-learn`: Cosine similarity calculations
- `pinecone`: Vector database for document storage
- `python-dotenv`: Environment variable management

## Error Handling

The system includes comprehensive error handling:
- Module import validation
- API key verification
- Graceful fallbacks for missing components
- Detailed error reporting and logging

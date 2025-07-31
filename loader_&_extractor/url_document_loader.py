import os
import sys
import requests
import tempfile
from pathlib import Path
from urllib.parse import urlparse
import subprocess

# Create wrapper functions for existing loaders
def get_pdf_text(file_path):
    """Extract text from PDF using existing pdfLoading.py logic"""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        full_text = []
        for page in pages:
            full_text.append(page.page_content)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise

def get_word_text(file_path):
    """Extract text from Word document using existing wordLoading.py logic"""
    try:
        # Import the existing function
        from wordLoading import getText
        return getText(file_path)
    except Exception as e:
        print(f"Error loading Word document: {e}")
        raise

def get_text_text(file_path):
    """Extract text from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading text file: {e}")
        raise

def download_document_from_url(url, temp_dir=None):
    """
    Download a document from URL to temporary file
    
    Args:
        url (str): URL of the document to download
        temp_dir (str): Temporary directory to save file (optional)
    
    Returns:
        str: Path to downloaded temporary file
    """
    try:
        print(f"Downloading document from URL: {url}")
        
        # Create temp directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        # Parse URL to get filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # If no filename in URL, create one based on content type
        if not filename or '.' not in filename:
            filename = "document"
        
        # Download the file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Try to get file extension from content-type if not in filename
        if '.' not in filename:
            content_type = response.headers.get('content-type', '')
            if 'pdf' in content_type:
                filename += '.pdf'
            elif 'word' in content_type or 'docx' in content_type:
                filename += '.docx'
            elif 'text' in content_type:
                filename += '.txt'
            else:
                filename += '.pdf'  # Default to PDF
        
        # Save to temporary file
        temp_file_path = os.path.join(temp_dir, filename)
        
        with open(temp_file_path, 'wb') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
        
        print(f"✅ Document downloaded successfully to: {temp_file_path}")
        return temp_file_path
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading document: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error downloading document: {e}")
        raise

def detect_file_type_from_path(file_path):
    """
    Detect file type from file path
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        str: File type ('pdf', 'docx', 'txt', 'unknown')
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        return 'pdf'
    elif file_extension in ['.docx', '.doc']:
        return 'docx'
    elif file_extension in ['.txt', '.text']:
        return 'txt'
    else:
        return 'unknown'

def extract_text_from_file(file_path, file_type=None):
    """
    Extract text from a file using appropriate loader
    
    Args:
        file_path (str): Path to the file
        file_type (str): File type override (optional)
    
    Returns:
        str: Extracted text content
    """
    try:
        # Auto-detect file type if not provided
        if file_type is None:
            file_type = detect_file_type_from_path(file_path)
        
        print(f"Extracting text from {file_type.upper()} file: {file_path}")
        
        # Extract text based on file type
        if file_type == 'pdf':
            return get_pdf_text(file_path)
        elif file_type == 'docx':
            return get_word_text(file_path)
        elif file_type == 'txt':
            return get_text_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        print(f"❌ Error extracting text from file: {e}")
        raise

def process_document_from_url(url, cleanup=True):
    """
    Complete pipeline: Download document from URL and extract text
    
    Args:
        url (str): URL of the document
        cleanup (bool): Whether to delete temporary file after processing
    
    Returns:
        dict: Dictionary containing extracted text and metadata
    """
    temp_file_path = None
    try:
        # Step 1: Download document
        temp_file_path = download_document_from_url(url)
        
        # Step 2: Detect file type
        file_type = detect_file_type_from_path(temp_file_path)
        
        # Step 3: Extract text
        extracted_text = extract_text_from_file(temp_file_path, file_type)
        
        # Step 4: Prepare result
        result = {
            'text': extracted_text,
            'source_url': url,
            'file_type': file_type,
            'temp_file_path': temp_file_path,
            'success': True
        }
        
        print(f"✅ Successfully processed document from URL")
        print(f"   File type: {file_type}")
        print(f"   Text length: {len(extracted_text)} characters")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing document from URL: {e}")
        result = {
            'text': '',
            'source_url': url,
            'file_type': 'unknown',
            'temp_file_path': temp_file_path,
            'success': False,
            'error': str(e)
        }
        return result
        
    finally:
        # Cleanup temporary file if requested
        if cleanup and temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"🗑️  Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up temporary file: {e}")

def process_document_from_local_path(file_path):
    """
    Process document from local file path
    
    Args:
        file_path (str): Local path to the document
    
    Returns:
        dict: Dictionary containing extracted text and metadata
    """
    try:
        # Step 1: Detect file type
        file_type = detect_file_type_from_path(file_path)
        
        # Step 2: Extract text
        extracted_text = extract_text_from_file(file_path, file_type)
        
        # Step 3: Prepare result
        result = {
            'text': extracted_text,
            'source_path': file_path,
            'file_type': file_type,
            'success': True
        }
        
        print(f"✅ Successfully processed local document")
        print(f"   File path: {file_path}")
        print(f"   File type: {file_type}")
        print(f"   Text length: {len(extracted_text)} characters")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing local document: {e}")
        result = {
            'text': '',
            'source_path': file_path,
            'file_type': 'unknown',
            'success': False,
            'error': str(e)
        }
        return result

def process_document(source, source_type='auto'):
    """
    Unified document processor - handles both URLs and local files
    
    Args:
        source (str): URL or local file path
        source_type (str): 'url', 'local', or 'auto' to detect automatically
    
    Returns:
        dict: Dictionary containing extracted text and metadata
    """
    try:
        # Auto-detect source type if not specified
        if source_type == 'auto':
            if source.startswith(('http://', 'https://')):
                source_type = 'url'
            else:
                source_type = 'local'
        
        print(f"Processing document from {source_type}: {source}")
        
        # Route to appropriate processor
        if source_type == 'url':
            return process_document_from_url(source)
        elif source_type == 'local':
            return process_document_from_local_path(source)
        else:
            raise ValueError(f"Invalid source_type: {source_type}")
            
    except Exception as e:
        print(f"❌ Error in unified document processor: {e}")
        return {
            'text': '',
            'source': source,
            'success': False,
            'error': str(e)
        }

# Test function for development
def test_document_processing():
    """Test function for document processing"""
    
    # Test with URL (from hackathon sample)
    test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    print("="*60)
    print("🧪 TESTING DOCUMENT PROCESSING PIPELINE")
    print("="*60)
    
    print("\n1. Testing URL processing...")
    result_url = process_document(test_url, 'url')
    if result_url['success']:
        print(f"✅ URL processing successful!")
        print(f"   Text preview: {result_url['text'][:200]}...")
    else:
        print(f"❌ URL processing failed: {result_url.get('error', 'Unknown error')}")
    
    # Test with local file if available
    local_test_files = [
        "../dataset/pdf-format/1.pdf",
        "../dataset/word-format/AI-helper-low-wage.docx",
        "../dataset/txt-format/sample.txt"
    ]
    
    print("\n2. Testing local file processing...")
    for test_file in local_test_files:
        if os.path.exists(test_file):
            print(f"\nTesting: {test_file}")
            result_local = process_document(test_file, 'local')
            if result_local['success']:
                print(f"✅ Local processing successful!")
                print(f"   Text preview: {result_local['text'][:200]}...")
            else:
                print(f"❌ Local processing failed: {result_local.get('error', 'Unknown error')}")
            break
    else:
        print("No local test files found in dataset directory")
    
    print("\n" + "="*60)
    print("🎉 TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_document_processing()

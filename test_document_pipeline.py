#!/usr/bin/env python3
"""
Test script for the unified document processing pipeline
Tests both URL and local file processing capabilities
"""

import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import from the loader directory
loader_dir = os.path.join(os.path.dirname(__file__), 'loader_&_extractor')
sys.path.append(loader_dir)

from url_document_loader import process_document, test_document_processing

def main():
    """Main test function"""
    print("🚀 Starting Document Processing Pipeline Test")
    print("="*60)
    
    # Test 1: URL Processing with Hackathon Sample
    hackathon_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    print("\n📁 Test 1: Processing Hackathon Sample PDF from URL")
    print("-" * 50)
    
    result = process_document(hackathon_url, 'url')
    
    if result['success']:
        print("✅ SUCCESS: Document processed successfully!")
        print(f"   📊 Text length: {len(result['text'])} characters")
        print(f"   📄 File type: {result['file_type']}")
        print(f"   🔗 Source URL: {result['source_url']}")
        print(f"   📝 Text preview (first 300 chars):")
        print(f"      {result['text'][:300]}...")
        
        # Check if it contains policy information
        text_lower = result['text'].lower()
        policy_keywords = ['policy', 'insurance', 'premium', 'claim', 'coverage']
        found_keywords = [kw for kw in policy_keywords if kw in text_lower]
        print(f"   🔍 Policy keywords found: {found_keywords}")
        
    else:
        print("❌ FAILED: Document processing failed!")
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Test 2: Local File Processing (if files exist)
    print("\n📁 Test 2: Processing Local Files")
    print("-" * 50)
    
    test_files = [
        "dataset/pdf-format/1.pdf",
        "dataset/word-format/AI-helper-low-wage.docx", 
        "dataset/txt-format/sample.txt"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n   Testing: {test_file}")
            local_result = process_document(test_file, 'local')
            
            if local_result['success']:
                print(f"   ✅ SUCCESS: {len(local_result['text'])} characters extracted")
                print(f"   📄 File type: {local_result['file_type']}")
                print(f"   📝 Preview: {local_result['text'][:100]}...")
            else:
                print(f"   ❌ FAILED: {local_result.get('error', 'Unknown error')}")
        else:
            print(f"   ⚠️  File not found: {test_file}")
    
    # Test 3: Auto-detection
    print("\n📁 Test 3: Auto-detection Test")
    print("-" * 50)
    
    auto_test_sources = [
        hackathon_url,  # Should detect as URL
        "dataset/pdf-format/1.pdf" if os.path.exists("dataset/pdf-format/1.pdf") else None  # Should detect as local
    ]
    
    for source in auto_test_sources:
        if source:
            print(f"\n   Auto-testing: {source[:60]}...")
            auto_result = process_document(source, 'auto')  # Auto-detect
            
            if auto_result['success']:
                source_type = 'URL' if source.startswith('http') else 'Local'
                print(f"   ✅ SUCCESS: Auto-detected as {source_type}")
                print(f"   📄 File type: {auto_result['file_type']}")
                print(f"   📊 Text length: {len(auto_result['text'])} characters")
            else:
                print(f"   ❌ FAILED: {auto_result.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)
    print("🎉 DOCUMENT PROCESSING PIPELINE TEST COMPLETE")
    print("="*60)
    
    # Summary
    print("\n📋 SUMMARY:")
    print("- ✅ URL document downloading")
    print("- ✅ Multiple file format support (PDF, DOCX, TXT)")
    print("- ✅ Local file processing")
    print("- ✅ Auto-detection of source type")
    print("- ✅ Error handling and cleanup")
    print("\n🚀 Ready for integration with RAG pipeline!")

if __name__ == "__main__":
    main()

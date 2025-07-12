from main_chunking import chunk_text

def test_chunk_text_basic():
    """Test basic text chunking functionality."""
    text = "This is a test document. " * 100  # Create a long text
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
    
    print(f"Original text length: {len(text)}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk length: {len(chunks[0])}")
    print(f"Last chunk length: {len(chunks[-1])}")
    
    # Verify chunks are not empty
    assert all(chunk.strip() for chunk in chunks), "All chunks should contain text"
    
    # Verify chunk sizes are reasonable
    for i, chunk in enumerate(chunks):
        assert len(chunk) <= 600, f"Chunk {i} is too long: {len(chunk)} characters"
    
    print("✓ Basic chunking test passed")

def test_chunk_text_small():
    """Test chunking with text smaller than chunk size."""
    text = "This is a small text."
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    
    print(f"Small text chunks: {len(chunks)}")
    assert len(chunks) == 1, "Small text should result in single chunk"
    assert chunks[0] == text, "Single chunk should contain original text"
    
    print("✓ Small text test passed")

def test_chunk_text_custom_separators():
    """Test chunking with custom separators."""
    text = "Section 1: Introduction.\nSection 2: Methods.\nSection 3: Results.\nSection 4: Conclusion."
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10, separators=["\n", ".", " "])
    
    print(f"Custom separator chunks: {len(chunks)}")
    print("Chunks with custom separators:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk[:50]}...")
    
    assert len(chunks) > 1, "Should create multiple chunks"
    
    print("✓ Custom separators test passed")

def test_chunk_text_real_document():
    """Test chunking with a realistic document structure."""
    text = """
    CHAPTER 1: INTRODUCTION
    
    This chapter provides an overview of the document structure and objectives.
    The main goal is to demonstrate text chunking capabilities.
    
    CHAPTER 2: METHODOLOGY
    
    The methodology section describes the approach used for text processing.
    Various techniques are employed to ensure optimal chunk sizes.
    
    CHAPTER 3: IMPLEMENTATION
    
    Implementation details are provided in this section.
    Code examples and best practices are included.
    
    CHAPTER 4: RESULTS
    
    Results demonstrate the effectiveness of the chunking algorithm.
    Performance metrics and analysis are presented.
    
    CHAPTER 5: CONCLUSION
    
    The conclusion summarizes key findings and future work.
    Recommendations for improvement are also discussed.
    """
    
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    
    print(f"Document chunks: {len(chunks)}")
    print("Sample chunks from realistic document:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk[:100].strip()}...")
    
    assert len(chunks) > 1, "Should create multiple chunks for long document"
    
    print("✓ Real document test passed")

def test_chunk_text_edge_cases():
    """Test edge cases."""
    
    # Empty text
    chunks = chunk_text("", chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0] == ""), "Empty text should result in empty or single empty chunk"
    print("✓ Empty text test passed")
    
    # Very small chunk size
    text = "This is a test with very small chunks."
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
    assert len(chunks) > 1, "Small chunk size should create multiple chunks"
    print("✓ Small chunk size test passed")
    
    print("✓ Edge cases test passed")

def main():
    """Run all tests."""
    print("Running main_chunking tests...\n")
    
    try:
        test_chunk_text_basic()
        print()
        
        test_chunk_text_small()
        print()
        
        test_chunk_text_custom_separators()
        print()
        
        test_chunk_text_real_document()
        print()
        
        test_chunk_text_edge_cases()
        print()
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()

import unittest
import os
import sys
import tempfile
import time
import io
from unittest.mock import patch, MagicMock
from PIL import Image
import fitz

# Add the main script to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handwriting_text_extraction import AsyncOCRProcessor

class TestHandwritingTextExtraction(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = AsyncOCRProcessor(max_concurrent=2)  # Lower concurrency for tests
        self.test_pdf_path = None
        
    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_pdf_path and os.path.exists(self.test_pdf_path):
            os.remove(self.test_pdf_path)
    
    def create_test_pdf(self, num_pages=3):
        """Create a simple test PDF with text"""
        # Create temporary PDF file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        os.close(temp_fd)
        
        doc = fitz.open()
        
        for i in range(num_pages):
            page = doc.new_page()
            text = f"This is test page {i + 1}\nSample handwritten text\nLine {i + 1}"
            page.insert_text((72, 72), text, fontsize=12)
        
        doc.save(temp_path)
        doc.close()
        
        self.test_pdf_path = temp_path
        return temp_path
    
    def test_extract_pages_as_images(self):
        """Test PDF to images conversion"""
        test_pdf = self.create_test_pdf(num_pages=2)
        
        pages_data = self.processor.extract_pages_as_images(test_pdf, dpi=100)
        
        self.assertEqual(len(pages_data), 2)
        self.assertIn('page_num', pages_data[0])
        self.assertIn('image_data', pages_data[0])
        self.assertEqual(pages_data[0]['page_num'], 0)
        self.assertEqual(pages_data[1]['page_num'], 1)
    
    def test_preprocess_image_data(self):
        """Test image preprocessing"""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        image_data = img_byte_arr.getvalue()
        
        processed_data = self.processor.preprocess_image_data(image_data)
        
        self.assertIsInstance(processed_data, bytes)
        self.assertGreater(len(processed_data), 0)
    
    @patch('handwriting_text_extraction.vision.ImageAnnotatorClient')
    def test_process_page_sync_success(self, mock_client_class):
        """Test successful page processing"""
        # Mock the Vision API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.error.message = ""
        mock_response.full_text_annotation.text = "Sample extracted text"
        mock_client.document_text_detection.return_value = mock_response
        
        # Create test page data
        img = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        
        page_data = {
            'page_num': 0,
            'image_data': img_byte_arr.getvalue()
        }
        
        result = self.processor._process_page_sync(page_data)
        
        self.assertEqual(result['page_num'], 0)
        self.assertEqual(result['text'], "Sample extracted text")
        self.assertFalse(result['error'])
    
    @patch('handwriting_text_extraction.vision.ImageAnnotatorClient')
    def test_process_page_sync_error(self, mock_client_class):
        """Test error handling in page processing"""
        # Mock the Vision API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock error response
        mock_response = MagicMock()
        mock_response.error.message = "API Error"
        mock_client.document_text_detection.return_value = mock_response
        
        # Create test page data
        img = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        
        page_data = {
            'page_num': 0,
            'image_data': img_byte_arr.getvalue()
        }
        
        result = self.processor._process_page_sync(page_data)
        
        self.assertEqual(result['page_num'], 0)
        self.assertEqual(result['text'], "")
        self.assertTrue(result['error'])
    
    def test_performance_benchmark(self):
        """Benchmark test to ensure performance targets"""
        test_pdf = self.create_test_pdf(num_pages=5)
        
        start_time = time.time()
        pages_data = self.processor.extract_pages_as_images(test_pdf, dpi=150)
        end_time = time.time()
        
        # Image extraction should be fast (< 2 seconds for 5 pages)
        extraction_time = end_time - start_time
        self.assertLess(extraction_time, 2.0, 
                       f"Image extraction took {extraction_time:.2f}s, should be < 2s")
        
        # Test that we got the expected number of pages
        self.assertEqual(len(pages_data), 5)
    
    def test_invalid_pdf_path(self):
        """Test handling of invalid PDF path"""
        with self.assertRaises(Exception):
            self.processor.extract_pages_as_images("nonexistent.pdf")
    
    def test_empty_pdf(self):
        """Test handling of PDF with blank page"""
        # Create PDF with one blank page (since PyMuPDF doesn't allow zero pages)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        os.close(temp_fd)
        
        try:
            doc = fitz.open()
            # Add a blank page (PyMuPDF requires at least one page)
            page = doc.new_page()
            doc.save(temp_path)
            doc.close()
            
            # Test extraction - should work with blank page
            pages_data = self.processor.extract_pages_as_images(temp_path)
            self.assertEqual(len(pages_data), 1)  # One blank page
            
            # Verify the page data structure
            self.assertIn('page_num', pages_data[0])
            self.assertIn('image_data', pages_data[0])
            self.assertEqual(pages_data[0]['page_num'], 0)
            self.assertIsInstance(pages_data[0]['image_data'], bytes)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_concurrency_limit(self):
        """Test that concurrency is properly limited"""
        processor = AsyncOCRProcessor(max_concurrent=3)
        self.assertEqual(processor.max_concurrent, 3)
        self.assertEqual(processor.semaphore._value, 3)

class TestIntegration(unittest.TestCase):
    """Integration tests that may require actual API calls"""
    
    def setUp(self):
        self.processor = AsyncOCRProcessor(max_concurrent=2)
        self.test_pdf_path = None
    
    def tearDown(self):
        if self.test_pdf_path and os.path.exists(self.test_pdf_path):
            os.remove(self.test_pdf_path)
    
    def create_test_pdf_with_text(self):
        """Create a PDF with actual text for integration testing"""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        os.close(temp_fd)
        
        doc = fitz.open()
        page = doc.new_page()
        
        # Add some text that looks like handwriting
        text = """Hello World
This is a test document
with multiple lines
for testing OCR"""
        
        page.insert_text((72, 72), text, fontsize=14)
        doc.save(temp_path)
        doc.close()
        
        self.test_pdf_path = temp_path
        return temp_path
    
    @unittest.skipUnless(
        os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'), 
        "Google credentials not configured"
    )
    def test_end_to_end_extraction(self):
        """End-to-end test with actual API calls (requires credentials)"""
        test_pdf = self.create_test_pdf_with_text()
        
        start_time = time.time()
        result = self.processor.extract_handwritten_text_from_pdf(test_pdf)
        end_time = time.time()
        
        # Should complete within reasonable time
        processing_time = end_time - start_time
        self.assertLess(processing_time, 10.0, 
                       f"Processing took {processing_time:.2f}s, should be < 10s")
        
        # Should return some text
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

class TestPerformance(unittest.TestCase):
    """Performance and load tests"""
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        processor = AsyncOCRProcessor(max_concurrent=5)
        
        # This would need actual memory monitoring
        # For now, just ensure it doesn't crash with multiple calls
        for i in range(3):
            try:
                # Create small test data
                img = Image.new('RGB', (50, 50), color='white')
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                
                processed = processor.preprocess_image_data(img_byte_arr.getvalue())
                self.assertIsInstance(processed, bytes)
            except Exception as e:
                self.fail(f"Memory test failed on iteration {i}: {e}")
    
    def test_concurrent_limit_respected(self):
        """Test that concurrency limits are respected"""
        processor = AsyncOCRProcessor(max_concurrent=2)
        
        # This is more of a behavioral test
        # In practice, you'd monitor actual concurrent requests
        self.assertEqual(processor.semaphore._value, 2)

if __name__ == '__main__':
    # Configure test output
    print("🧪 Running Handwriting Text Extraction Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestHandwritingTextExtraction))
    suite.addTest(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTest(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print(f"Tests run: {result.testsRun}")
    print("=" * 50)
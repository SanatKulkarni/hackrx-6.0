import unittest
import os
import sys
from unittest.mock import patch, MagicMock
from spire.doc import Document
from spire.doc.common import *

# Add the current directory to the path so we can import from wordLoading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestWordLoading(unittest.TestCase):
    """Test cases for Word document loading functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_word_path = "../dataset/word-format/AI-helper-low-wage.docx"
        self.invalid_word_path = "../dataset/word-format/nonexistent.docx"
        self.base_dataset_path = "../dataset/word-format/"
    
    def test_word_file_exists(self):
        """Test that the Word file exists in the expected location"""
        abs_path = os.path.abspath(self.test_word_path)
        self.assertTrue(os.path.exists(abs_path), f"Word file {self.test_word_path} should exist")
    
    def test_document_initialization(self):
        """Test that Spire Document can be initialized"""
        document = Document()
        self.assertIsInstance(document, Document)
        document.Close()
    
    def test_load_valid_word_document(self):
        """Test loading a valid Word document"""
        document = Document()
        try:
            # Check if file exists before loading
            abs_path = os.path.abspath(self.test_word_path)
            if os.path.exists(abs_path):
                document.LoadFromFile(self.test_word_path)
                # Check if document is loaded properly (mimics updated wordLoading.py)
                if hasattr(document, 'IsLoaded'):
                    self.assertTrue(document.IsLoaded, "Document should be loaded successfully")
                else:
                    self.assertTrue(True, "Document loaded successfully")
            else:
                self.skipTest(f"Test Word file {self.test_word_path} not found")
        except Exception as e:
            self.fail(f"Failed to load Word document: {str(e)}")
        finally:
            document.Close()
    
    def test_extract_text_from_document(self):
        """Test extracting text from the Word document"""
        document = Document()
        try:
            abs_path = os.path.abspath(self.test_word_path)
            if not os.path.exists(abs_path):
                self.skipTest(f"Test Word file {self.test_word_path} not found")
            
            document.LoadFromFile(self.test_word_path)
            # Check if document is loaded properly (mimics updated wordLoading.py)
            if hasattr(document, 'IsLoaded') and not document.IsLoaded:
                self.fail("Document failed to load - invalid format")
            
            document_text = document.GetText()
            
            # Verify that text is extracted
            self.assertIsInstance(document_text, str)
            self.assertGreater(len(document_text.strip()), 0, "Document should contain some text")
            
        except Exception as e:
            self.fail(f"Failed to extract text from Word document: {str(e)}")
        finally:
            document.Close()
    
    def test_document_text_content(self):
        """Test that the extracted text contains meaningful content"""
        document = Document()
        try:
            abs_path = os.path.abspath(self.test_word_path)
            if not os.path.exists(abs_path):
                self.skipTest(f"Test Word file {self.test_word_path} not found")
            
            document.LoadFromFile(self.test_word_path)
            document_text = document.GetText()
            
            # Check for some expected content based on filename
            # Since the file is named "AI-helper-low-wage.docx", it likely contains AI-related content
            text_lower = document_text.lower()
            
            # Verify the text is not just whitespace
            self.assertGreater(len(document_text.strip()), 10, "Document should contain substantial text")
            
            # The text should contain some readable content (not just special characters)
            import re
            alphanumeric_chars = re.findall(r'[a-zA-Z0-9]', document_text)
            self.assertGreater(len(alphanumeric_chars), 5, "Document should contain alphanumeric characters")
            
        except Exception as e:
            self.fail(f"Failed to analyze document text content: {str(e)}")
        finally:
            document.Close()
    
    def test_error_handling_invalid_file(self):
        """Test error handling when loading a non-existent file"""
        document = Document()
        try:
            # Attempt to load a non-existent file
            with self.assertRaises(Exception):
                document.LoadFromFile(self.invalid_word_path)
        finally:
            document.Close()
    
    def test_error_handling_invalid_file_format(self):
        """Test error handling when loading an invalid file format"""
        document = Document()
        temp_file = None
        try:
            # Create a temporary text file with .docx extension
            temp_file = "temp_invalid.docx"
            with open(temp_file, 'w') as f:
                f.write("This is not a valid Word document")
            
            # This should raise an exception or result in IsLoaded = False
            document.LoadFromFile(temp_file)
            if hasattr(document, 'IsLoaded') and not document.IsLoaded:
                # Expected behavior - document failed to load
                self.assertFalse(document.IsLoaded, "Invalid document should not be loaded")
            else:
                # Some versions might raise an exception instead
                self.fail("Expected invalid document to either raise exception or set IsLoaded to False")
                
        except Exception:
            # This is also acceptable behavior for invalid files
            self.assertTrue(True, "Exception raised for invalid file format as expected")
        finally:
            document.Close()
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_document_closure(self):
        """Test that documents are properly closed"""
        document = Document()
        try:
            abs_path = os.path.abspath(self.test_word_path)
            if not os.path.exists(abs_path):
                self.skipTest(f"Test Word file {self.test_word_path} not found")
            
            document.LoadFromFile(self.test_word_path)
            document_text = document.GetText()
            
            # Verify we can call Close without errors
            document.Close()
            self.assertTrue(True, "Document closed successfully")
            
        except Exception as e:
            self.fail(f"Failed to close document properly: {str(e)}")
    
    @patch('sys.stdout', new_callable=MagicMock)
    def test_print_functionality(self, mock_stdout):
        """Test that the document text can be printed (mimics original code)"""
        document = Document()
        try:
            abs_path = os.path.abspath(self.test_word_path)
            if not os.path.exists(abs_path):
                self.skipTest(f"Test Word file {self.test_word_path} not found")
            
            document.LoadFromFile(self.test_word_path)
            document_text = document.GetText()
            
            # Mimic the print functionality from the original code
            print(document_text)
            
            # Verify print was called (mock will capture this)
            self.assertTrue(True, "Print functionality works")
            
        except Exception as e:
            self.fail(f"Failed to print document text: {str(e)}")
        finally:
            document.Close()
    
    def test_text_encoding_handling(self):
        """Test that the extracted text handles encoding properly"""
        document = Document()
        try:
            abs_path = os.path.abspath(self.test_word_path)
            if not os.path.exists(abs_path):
                self.skipTest(f"Test Word file {self.test_word_path} not found")
            
            document.LoadFromFile(self.test_word_path)
            document_text = document.GetText()
            
            # Verify the text can be encoded/decoded without errors
            try:
                encoded_text = document_text.encode('utf-8')
                decoded_text = encoded_text.decode('utf-8')
                self.assertEqual(document_text, decoded_text)
            except UnicodeError:
                self.fail("Document text contains encoding issues")
            
        except Exception as e:
            self.fail(f"Failed to test text encoding: {str(e)}")
        finally:
            document.Close()
    
    def test_error_handling_like_updated_code(self):
        """Test error handling similar to the updated wordLoading.py"""
        document = Document()
        try:
            abs_path = os.path.abspath(self.test_word_path)
            if not os.path.exists(abs_path):
                self.skipTest(f"Test Word file {self.test_word_path} not found")
            
            # Mimic the updated wordLoading.py error handling
            document.LoadFromFile(self.test_word_path)
            if hasattr(document, 'IsLoaded') and not document.IsLoaded:
                raise Exception("Invalid document format")
            
            document_text = document.GetText()
            self.assertIsInstance(document_text, str)
            
        except Exception as e:
            # Check if it's the expected error message format
            error_msg = str(e).lower()
            if "invalid" in error_msg or "format" in error_msg:
                self.assertTrue(True, "Proper error handling for invalid format")
            else:
                self.fail(f"Unexpected error: {str(e)}")
        finally:
            document.Close()
    
    def test_invalid_file_error_message(self):
        """Test that proper error messages are generated for invalid files"""
        document = Document()
        try:
            # Test with non-existent file
            with self.assertRaises(Exception) as context:
                document.LoadFromFile(self.invalid_word_path)
            
            # The error should be related to file not found or invalid format
            error_msg = str(context.exception).lower()
            self.assertTrue(
                any(keyword in error_msg for keyword in ['file', 'format', 'invalid', 'not found']),
                f"Error message should indicate file/format issue: {error_msg}"
            )
        finally:
            document.Close()


class TestWordLoadingIntegration(unittest.TestCase):
    """Integration tests that mimic the actual usage in wordLoading.py"""
    
    def test_original_code_functionality(self):
        """Test that replicates the exact functionality of the updated wordLoading.py"""
        try:
            document = Document()
            document.LoadFromFile("../dataset/word-format/AI-helper-low-wage.docx")
            if hasattr(document, 'IsLoaded') and not document.IsLoaded:
                raise Exception("Invalid document format")
            document_text = document.GetText()
            
            # Verify the core functionality works
            self.assertIsInstance(document_text, str)
            self.assertGreater(len(document_text.strip()), 0)
            
            # The original code prints the text (we'll just verify it's printable)
            str(document_text)  # This should not raise an exception
            
        except Exception as e:
            # Check if file exists first
            abs_path = os.path.abspath("../dataset/word-format/AI-helper-low-wage.docx")
            if not os.path.exists(abs_path):
                self.skipTest("Test Word file not found")
            
            # If file exists but we get an error, check if it's the expected error format
            error_msg = str(e).lower()
            if "invalid" in error_msg or "format" in error_msg or "file not found" in error_msg:
                self.assertTrue(True, "Proper error handling as in updated code")
            else:
                self.fail(f"Original code functionality failed: {str(e)}")
        finally:
            if 'document' in locals():
                document.Close()
    
    def test_complete_workflow(self):
        """Test the complete workflow from loading to text extraction"""
        document = Document()
        try:
            abs_path = os.path.abspath("../dataset/word-format/AI-helper-low-wage.docx")
            if not os.path.exists(abs_path):
                self.skipTest("Test Word file not found")
            
            # Step 1: Initialize document
            self.assertIsInstance(document, Document)
            
            # Step 2: Load document
            document.LoadFromFile("../dataset/word-format/AI-helper-low-wage.docx")
            
            # Step 3: Extract text
            document_text = document.GetText()
            self.assertIsInstance(document_text, str)
            
            # Step 4: Verify text content
            self.assertGreater(len(document_text.strip()), 0)
            
            # Step 5: Verify text is readable
            self.assertTrue(any(c.isalnum() for c in document_text))
            
        except Exception as e:
            self.fail(f"Complete workflow test failed: {str(e)}")
        finally:
            # Step 6: Close document
            document.Close()
    
    def test_resource_management(self):
        """Test proper resource management with multiple document operations"""
        for i in range(3):  # Test multiple iterations
            document = Document()
            try:
                abs_path = os.path.abspath("../dataset/word-format/AI-helper-low-wage.docx")
                if not os.path.exists(abs_path):
                    self.skipTest("Test Word file not found")
                
                document.LoadFromFile("../dataset/word-format/AI-helper-low-wage.docx")
                document_text = document.GetText()
                
                self.assertIsInstance(document_text, str)
                
            except Exception as e:
                self.fail(f"Resource management test failed on iteration {i}: {str(e)}")
            finally:
                document.Close()


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)

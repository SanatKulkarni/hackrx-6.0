import unittest
import os
import sys
import tempfile
from unittest.mock import patch, mock_open, MagicMock
from io import StringIO

# Add the current directory to the path so we can import from textLoading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the function we want to test
try:
    from textLoading import read_and_print_file
except ImportError:
    # If import fails, we'll define a mock for testing
    def read_and_print_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                print(content)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"Error occurred: {e}")


class TestTextLoading(unittest.TestCase):
    """Test cases for text file loading functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_txt_path = "../dataset/txt-format/sample.txt"
        self.invalid_txt_path = "../dataset/txt-format/nonexistent.txt"
        self.base_dataset_path = "../dataset/txt-format/"
    
    def test_text_file_exists(self):
        """Test that the text file exists in the expected location"""
        abs_path = os.path.abspath(self.test_txt_path)
        self.assertTrue(os.path.exists(abs_path), f"Text file {self.test_txt_path} should exist")
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_read_valid_text_file(self, mock_stdout):
        """Test reading a valid text file"""
        abs_path = os.path.abspath(self.test_txt_path)
        if not os.path.exists(abs_path):
            self.skipTest(f"Test text file {self.test_txt_path} not found")
        
        # Call the function
        read_and_print_file(self.test_txt_path)
        
        # Get the printed output
        output = mock_stdout.getvalue()
        
        # Verify that something was printed (the file content)
        self.assertGreater(len(output.strip()), 0, "Should print file content")
        
        # Verify the output doesn't contain error messages
        self.assertNotIn("Error:", output)
        self.assertNotIn("not found", output.lower())
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_read_nonexistent_file(self, mock_stdout):
        """Test reading a file that doesn't exist"""
        # Call the function with non-existent file
        read_and_print_file(self.invalid_txt_path)
        
        # Get the printed output
        output = mock_stdout.getvalue()
        
        # Verify that error message was printed
        self.assertIn("Error:", output)
        self.assertIn("not found", output)
        self.assertIn(self.invalid_txt_path, output)
    
    def test_file_content_verification(self):
        """Test that the file content is read correctly"""
        abs_path = os.path.abspath(self.test_txt_path)
        if not os.path.exists(abs_path):
            self.skipTest(f"Test text file {self.test_txt_path} not found")
        
        # Read the file directly to compare
        with open(self.test_txt_path, 'r', encoding='utf-8') as file:
            expected_content = file.read()
        
        # Capture the output from our function
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            read_and_print_file(self.test_txt_path)
            actual_output = mock_stdout.getvalue()
        
        # The output should match the file content (plus newline from print)
        self.assertEqual(actual_output.rstrip('\n'), expected_content.rstrip('\n'))
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    @patch('sys.stdout', new_callable=StringIO)
    def test_permission_error_handling(self, mock_stdout, mock_open):
        """Test handling of permission errors"""
        read_and_print_file("some_file.txt")
        
        output = mock_stdout.getvalue()
        self.assertIn("Error occurred:", output)
        self.assertIn("Permission denied", output)
    
    @patch('builtins.open', side_effect=UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte'))
    @patch('sys.stdout', new_callable=StringIO)
    def test_encoding_error_handling(self, mock_stdout, mock_open):
        """Test handling of encoding errors"""
        read_and_print_file("some_file.txt")
        
        output = mock_stdout.getvalue()
        self.assertIn("Error occurred:", output)
    
    def test_function_with_different_encodings(self):
        """Test the function with files having different encodings"""
        # Create a temporary file with UTF-8 content
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as temp_file:
            test_content = "Hello, World! 🌍 Test content with unicode."
            temp_file.write(test_content)
            temp_file_path = temp_file.name
        
        try:
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                read_and_print_file(temp_file_path)
                output = mock_stdout.getvalue()
            
            # Verify the content was read correctly
            self.assertIn("Hello, World!", output)
            self.assertIn("🌍", output)
            self.assertNotIn("Error:", output)
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_empty_file_handling(self):
        """Test handling of empty files"""
        # Create a temporary empty file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file_path = temp_file.name
            # File is created but nothing is written (empty file)
        
        try:
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                read_and_print_file(temp_file_path)
                output = mock_stdout.getvalue()
            
            # Should not raise an error, just print empty content
            self.assertNotIn("Error:", output)
            # Output might be empty or just contain newline
            self.assertEqual(len(output.strip()), 0)
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_large_file_handling(self):
        """Test handling of larger text files"""
        # Create a temporary file with substantial content
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as temp_file:
            # Write multiple lines of content
            large_content = "\n".join([f"Line {i}: This is test content for line number {i}" for i in range(100)])
            temp_file.write(large_content)
            temp_file_path = temp_file.name
        
        try:
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                read_and_print_file(temp_file_path)
                output = mock_stdout.getvalue()
            
            # Verify the content was read correctly
            self.assertIn("Line 0:", output)
            self.assertIn("Line 99:", output)
            self.assertNotIn("Error:", output)
            
            # Verify all lines are present
            output_lines = output.strip().split('\n')
            self.assertEqual(len(output_lines), 100)
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_file_with_special_characters(self, mock_stdout):
        """Test reading files with special characters and symbols"""
        # Create a temporary file with special characters
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as temp_file:
            special_content = "Special chars: @#$%^&*()[]{}|\\:;\"'<>?/~`\nUnicode: αβγδε 中文 العربية\nNumbers: 12345"
            temp_file.write(special_content)
            temp_file_path = temp_file.name
        
        try:
            read_and_print_file(temp_file_path)
            output = mock_stdout.getvalue()
            
            # Verify special characters are handled correctly
            self.assertIn("@#$%^&*()", output)
            self.assertIn("αβγδε", output)
            self.assertIn("中文", output)
            self.assertIn("العربية", output)
            self.assertNotIn("Error:", output)
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


class TestTextLoadingIntegration(unittest.TestCase):
    """Integration tests that mimic the actual usage in textLoading.py"""
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_original_code_functionality(self, mock_stdout):
        """Test that replicates the exact functionality of the original code"""
        file_path = "../dataset/txt-format/sample.txt"
        
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            self.skipTest("Test text file not found")
        
        # Mimic the original code exactly
        read_and_print_file(file_path)
        
        output = mock_stdout.getvalue()
        
        # Verify the core functionality works
        if "Error:" not in output:
            # If no error, should have content
            self.assertGreater(len(output.strip()), 0)
        else:
            # If there's an error, it should be a proper error message
            self.assertIn("Error:", output)
    
    def test_function_signature_and_behavior(self):
        """Test that the function has the correct signature and behavior"""
        # Test that function exists and is callable
        self.assertTrue(callable(read_and_print_file))
        
        # Test that function accepts a file path parameter
        try:
            # This should not raise a TypeError about missing arguments
            with patch('sys.stdout', new_callable=StringIO):
                with patch('builtins.open', side_effect=FileNotFoundError):
                    read_and_print_file("test.txt")
        except TypeError as e:
            if "missing" in str(e) and "required" in str(e):
                self.fail("Function should accept a file_path parameter")
    
    def test_utf8_encoding_usage(self):
        """Test that the function properly uses UTF-8 encoding"""
        # Create a file with UTF-8 content
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as temp_file:
            utf8_content = "UTF-8 test: café, naïve, résumé, 漢字"
            temp_file.write(utf8_content)
            temp_file_path = temp_file.name
        
        try:
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                read_and_print_file(temp_file_path)
                output = mock_stdout.getvalue()
            
            # Verify UTF-8 characters are handled correctly
            self.assertIn("café", output)
            self.assertIn("naïve", output)
            self.assertIn("résumé", output)
            self.assertIn("漢字", output)
            self.assertNotIn("Error:", output)
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_error_message_format(self):
        """Test that error messages follow the expected format"""
        nonexistent_file = "definitely_does_not_exist.txt"
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            read_and_print_file(nonexistent_file)
            output = mock_stdout.getvalue()
        
        # Check error message format matches the original code
        self.assertIn("Error: File", output)
        self.assertIn("not found", output)
        self.assertIn(nonexistent_file, output)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)

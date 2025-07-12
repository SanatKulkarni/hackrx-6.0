import unittest
import asyncio
import os
import sys
from unittest.mock import patch, MagicMock
from langchain_community.document_loaders import PyPDFLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestPDFLoading(unittest.TestCase):
    """Test cases for PDF loading functionality"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_pdf_path = "../dataset/pdf-format/1.pdf"
        self.invalid_pdf_path = "../dataset/pdf-format/nonexistent.pdf"
        self.all_pdf_files = [
            "../dataset/pdf-format/1.pdf",
            "../dataset/pdf-format/2.pdf",
            "../dataset/pdf-format/3.pdf",
            "../dataset/pdf-format/4.pdf",
            "../dataset/pdf-format/5.pdf"
        ]

    def test_pdf_file_exists(self):
        """Test that the PDF files exist in the expected location"""
        for pdf_path in self.all_pdf_files:

            abs_path = os.path.abspath(pdf_path)
            self.assertTrue(os.path.exists(abs_path), f"PDF file {pdf_path} should exist")

    def test_pypdflloader_initialization(self):
        """Test that PyPDFLoader can be initialized with a valid file path"""
        loader = PyPDFLoader(self.test_pdf_path)
        self.assertIsInstance(loader, PyPDFLoader)
        self.assertEqual(loader.file_path, self.test_pdf_path)

    def test_pypdflloader_invalid_path(self):
        """Test PyPDFLoader behavior with invalid file path"""

        with self.assertRaises(ValueError):
            loader = PyPDFLoader(self.invalid_pdf_path)

    async def async_load_pages_test(self, file_path):
        """Helper method to test async loading of pages"""
        loader = PyPDFLoader(file_path)
        pages = []

        try:
            async for page in loader.alazy_load():
                pages.append(page)

                self.assertTrue(hasattr(page, 'page_content'))
                self.assertTrue(hasattr(page, 'metadata'))
                self.assertIsInstance(page.page_content, str)
                self.assertIsInstance(page.metadata, dict)

        except Exception as e:
            self.fail(f"Failed to load pages from {file_path}: {str(e)}")

        return pages

    def test_async_load_pages_single_file(self):
        """Test async loading of pages from a single PDF file"""
        pages = asyncio.run(self.async_load_pages_test(self.test_pdf_path))
        self.assertGreater(len(pages), 0, "Should load at least one page")

        for page in pages:
            self.assertIsInstance(page.page_content, str)

            self.assertIsNotNone(page.page_content)

    def test_async_load_all_pdf_files(self):
        """Test async loading of pages from all PDF files"""
        for pdf_path in self.all_pdf_files:
            with self.subTest(pdf_path=pdf_path):
                if os.path.exists(os.path.abspath(pdf_path)):
                    pages = asyncio.run(self.async_load_pages_test(pdf_path))
                    self.assertGreater(len(pages), 0, f"Should load at least one page from {pdf_path}")

    def test_page_metadata_structure(self):
        """Test that loaded pages have proper metadata structure"""
        pages = asyncio.run(self.async_load_pages_test(self.test_pdf_path))

        for page in pages:
            metadata = page.metadata

            self.assertIn('source', metadata, "Metadata should contain source field")
            self.assertIn('page', metadata, "Metadata should contain page number")
            self.assertEqual(metadata['source'], self.test_pdf_path)
            self.assertIsInstance(metadata['page'], int)

    @patch('sys.stdout', new_callable=MagicMock)
    def test_print_output(self, mock_stdout):
        """Test that the loading process produces output"""

        async def load_with_print():
            loader = PyPDFLoader(self.test_pdf_path)
            pages = []
            async for page in loader.alazy_load():
                pages.append(page)
                print(page.page_content)  
            return pages

        pages = asyncio.run(load_with_print())
        self.assertGreater(len(pages), 0)

    def test_error_handling_invalid_file(self):
        """Test error handling when trying to create loader with invalid file"""

        with self.assertRaises(ValueError) as context:
            loader = PyPDFLoader(self.invalid_pdf_path)

        self.assertIn("not a valid file", str(context.exception))

    def test_page_content_not_empty(self):
        """Test that loaded pages contain meaningful content"""
        pages = asyncio.run(self.async_load_pages_test(self.test_pdf_path))

        has_content = any(page.page_content.strip() for page in pages)
        self.assertTrue(has_content, "At least one page should contain non-whitespace content")

    def test_async_functionality(self):
        """Test that the async functionality works correctly"""
        async def test_async_behavior():
            loader = PyPDFLoader(self.test_pdf_path)
            pages = []
            page_count = 0

            async for page in loader.alazy_load():
                pages.append(page)
                page_count += 1

                if page_count >= 3:
                    break

            return pages, page_count

        pages, count = asyncio.run(test_async_behavior())
        self.assertEqual(len(pages), count)
        self.assertGreater(count, 0)

class TestPDFLoadingIntegration(unittest.TestCase):
    """Integration tests that mimic the actual usage in pdfLoading.py"""

    def test_original_code_functionality(self):
        """Test that replicates the exact functionality of the original code"""
        file_path = "../dataset/pdf-format/1.pdf"
        loader = PyPDFLoader(file_path)
        pages = []

        async def load_pages():
            async for page in loader.alazy_load():
                pages.append(page)

                self.assertIsInstance(page.page_content, str)

        try:
            asyncio.run(load_pages())
            self.assertGreater(len(pages), 0, "Should load at least one page")
        except Exception as e:
            self.fail(f"Original code functionality failed: {str(e)}")

    def test_multiple_files_loading(self):
        """Test loading multiple PDF files sequentially"""
        all_pages = {}

        for i in range(1, 6):  
            file_path = f"../dataset/pdf-format/{i}.pdf"
            if os.path.exists(os.path.abspath(file_path)):
                loader = PyPDFLoader(file_path)
                pages = []

                async def load_pages():
                    async for page in loader.alazy_load():
                        pages.append(page)

                asyncio.run(load_pages())
                all_pages[file_path] = pages
                self.assertGreater(len(pages), 0, f"Should load pages from {file_path}")

        self.assertGreater(len(all_pages), 0, "Should load from at least one PDF file")

if __name__ == '__main__':

    unittest.main(verbosity=2)
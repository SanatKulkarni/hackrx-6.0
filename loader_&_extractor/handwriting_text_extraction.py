import os
import sys
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import cv2
from google.cloud import vision
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class AsyncOCRProcessor:
    def __init__(self, max_concurrent=10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    def create_vision_client(self):
        """Create a new client for each thread"""
        return vision.ImageAnnotatorClient()
    
    def extract_pages_as_images(self, pdf_path, dpi=150):
        """Convert PDF pages to images with page numbers for ordering"""
        pages_data = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            # Convert page to image
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            pages_data.append({
                'page_num': page_num,
                'image_data': img_data
            })
        
        doc.close()
        return pages_data
    
    def preprocess_image_data(self, image_data):
        """Minimal preprocessing on image data"""
        # Convert to PIL Image
        pil_img = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale
        img = np.array(pil_img.convert('L'))
        
        # Simple threshold for speed
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        
        # Convert back to bytes
        processed_img = Image.fromarray(img)
        img_byte_arr = io.BytesIO()
        processed_img.save(img_byte_arr, format='PNG', optimize=True)
        
        return img_byte_arr.getvalue()
    
    async def process_single_page_async(self, page_data, executor):
        """Process a single page asynchronously"""
        async with self.semaphore:  # Limit concurrent requests
            loop = asyncio.get_event_loop()
            
            # Run the actual OCR in a thread pool to avoid blocking
            result = await loop.run_in_executor(
                executor, 
                self._process_page_sync, 
                page_data
            )
            
            return result
    
    def _process_page_sync(self, page_data):
        """Synchronous page processing (runs in thread pool)"""
        try:
            # Create client for this thread
            client = self.create_vision_client()
            
            # Preprocess image
            processed_data = self.preprocess_image_data(page_data['image_data'])
            
            # Create vision image
            image = vision.Image(content=processed_data)
            
            # Make API call
            response = client.document_text_detection(image=image)
            
            if response.error.message:
                print(f"API Error on page {page_data['page_num'] + 1}: {response.error.message}")
                return {
                    'page_num': page_data['page_num'],
                    'text': '',
                    'error': True
                }
            
            text = response.full_text_annotation.text if response.full_text_annotation else ""
            
            return {
                'page_num': page_data['page_num'],
                'text': text,
                'error': False
            }
            
        except Exception as e:
            print(f"Exception processing page {page_data['page_num'] + 1}: {e}")
            return {
                'page_num': page_data['page_num'],
                'text': '',
                'error': True
            }
    
    async def process_all_pages_async(self, pages_data):
        """Process all pages concurrently"""
        # Create thread pool for blocking operations
        max_workers = min(self.max_concurrent, len(pages_data))
        executor = ThreadPoolExecutor(max_workers=max_workers)
        
        try:
            print(f"Processing {len(pages_data)} pages with {max_workers} concurrent workers...")
            
            # Create tasks for all pages
            tasks = [
                self.process_single_page_async(page_data, executor)
                for page_data in pages_data
            ]
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Exception for page {i + 1}: {result}")
                    processed_results.append({
                        'page_num': i,
                        'text': '',
                        'error': True
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        finally:
            executor.shutdown(wait=True)
    
    def extract_handwritten_text_from_pdf(self, pdf_path):
        """Main extraction function with async processing"""
        start_time = time.time()
        
        print("Converting PDF pages to images...")
        pages_data = self.extract_pages_as_images(pdf_path, dpi=150)
        
        if not pages_data:
            return "No pages found in PDF."
        
        print(f"Starting async processing of {len(pages_data)} pages...")
        
        # Run async processing
        results = asyncio.run(self.process_all_pages_async(pages_data))
        
        # Sort results by page number to maintain order
        results.sort(key=lambda x: x['page_num'])
        
        # Combine text from all pages
        text_parts = []
        successful_pages = 0
        
        for result in results:
            if not result['error'] and result['text'].strip():
                text_parts.append(result['text'].strip())
                successful_pages += 1
        
        print(f"Successfully processed {successful_pages}/{len(pages_data)} pages")
        
        full_text = '\n\n'.join(text_parts)  # Double newline between pages
        
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        
        return full_text

def main():
    if len(sys.argv) < 2:
        print("Usage: python handwriting_text_extraction.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Extracting handwritten text from: {pdf_path}")
    
    # Initialize async processor
    processor = AsyncOCRProcessor(max_concurrent=10)
    text = processor.extract_handwritten_text_from_pdf(pdf_path)
    
    print("\n--- Extracted Handwritten Text ---")
    print(text)

if __name__ == "__main__":
    main()
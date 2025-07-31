from langchain_community.document_loaders import PyPDFLoader
import os
import asyncio
import sys

if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file_path = os.environ.get('FILE_PATH', '../dataset/pdf-format/2.pdf')

loader = PyPDFLoader(file_path)
pages = []

async def load_pages():
    async for page in loader.alazy_load():
        pages.append(page)
        try:
            print(page.page_content.encode('utf-8', errors='replace').decode('utf-8'))
        except UnicodeEncodeError:
            print(page.page_content.encode('utf-8', errors='replace').decode('utf-8'))

asyncio.run(load_pages())
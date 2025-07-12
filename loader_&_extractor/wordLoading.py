from spire.doc import *
from spire.doc.common import *
import os


file_path = os.environ.get('FILE_PATH', "../dataset/word-format/AI-helper-low-wage.docx")

if not file_path.lower().endswith('.docx'):
    raise Exception("Invalid file type: Only .docx files are supported by wordLoading.py")

try:
    document = Document()
    document.LoadFromFile(file_path)
    document_text = document.GetText()
    print(document_text)

except Exception as e:
    print(f"Error: Unable to process document. Invalid format or file not found.")
    print(f"Details: {str(e)}")
    raise

finally:
    if 'document' in locals():
        document.Close()
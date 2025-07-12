import os
import sys
import subprocess
from pathlib import Path

if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def detect_file_type(file_path):
    """
    Detect the file type based on file extension.

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

def run_pdf_loader(file_path):
    """
    Run the PDF loader by executing pdfLoading.py with the specified file path.

    Args:
        file_path (str): Path to the PDF file
    """
    print(f"Processing PDF file: {file_path}")

    try:

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['FILE_PATH'] = os.path.abspath(file_path)

        result = subprocess.run([sys.executable, "pdfLoading.py"], 
                              capture_output=True, text=True, 
                              cwd=os.path.dirname(__file__),
                              env=env, encoding='utf-8', errors='replace')


        if result.returncode == 0:
            print("PDF processing output:")
            clean_text = result.stdout.encode('utf-8', errors='replace').decode('utf-8')
            print(clean_text)
            print("Successfully processed PDF file.")
            return clean_text
        else:
            print(f"Error processing PDF file:")
            print(result.stderr)
            raise Exception(f"PDF processing failed with return code {result.returncode}")

    except Exception as e:
        print(f"Error running PDF loader: {str(e)}")
        raise

def run_word_loader(file_path):
    """
    Run the Word document loader by executing wordLoading.py with the specified file path.

    Args:
        file_path (str): Path to the Word document
    """
    print(f"Processing Word document: {file_path}")

    try:

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['FILE_PATH'] = os.path.abspath(file_path)

        result = subprocess.run([sys.executable, "wordLoading.py"], 
                              capture_output=True, text=True, 
                              cwd=os.path.dirname(__file__),
                              env=env, encoding='utf-8', errors='replace')


        if result.returncode == 0:
            print("Word document processing output:")
            print(result.stdout)
            print("Successfully processed Word document.")
            return result.stdout
        else:
            print(f"Error processing Word document:")
            print(result.stderr)
            raise Exception(f"Word processing failed with return code {result.returncode}")

    except Exception as e:
        print(f"Error running Word loader: {str(e)}")
        raise

def run_text_loader(file_path):
    """
    Run the text file loader by executing textLoading.py with the specified file path.

    Args:
        file_path (str): Path to the text file
    """
    print(f"Processing text file: {file_path}")

    try:

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['FILE_PATH'] = os.path.abspath(file_path)

        result = subprocess.run([sys.executable, "textLoading.py"], 
                              capture_output=True, text=True, 
                              cwd=os.path.dirname(__file__),
                              env=env, encoding='utf-8', errors='replace')


        if result.returncode == 0:
            print("Text file processing output:")
            print(result.stdout)
            print("Successfully processed text file.")
            return result.stdout
        else:
            print(f"Error processing text file:")
            print(result.stderr)
            raise Exception(f"Text processing failed with return code {result.returncode}")

    except Exception as e:
        print(f"Error running text loader: {str(e)}")
        raise

def process_file(file_path):
    """
    Main function to process a file based on its type.

    Args:
        file_path (str): Path to the file to process
    """
    try:

        file_type = detect_file_type(file_path)
        print(f"Detected file type: {file_type}")


        if file_type == 'pdf':
            return run_pdf_loader(file_path)
        elif file_type == 'docx':
            return run_word_loader(file_path)
        elif file_type == 'txt':
            return run_text_loader(file_path)
        else:
            print(f"Unsupported file type: {file_type}")
            print(f"Supported types: PDF (.pdf), Word (.docx, .doc), Text (.txt, .text)")
            return None


    except Exception as e:
        print(f"Error processing file '{file_path}': {str(e)}")
        return False

def main():
    """
    Main function to handle command line arguments or default file processing.
    """


    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        default_files = [
            "../dataset/pdf-format/1.pdf",
            "../dataset/word-format/AI-helper-low-wage.docx",
            "../dataset/txt-format/sample.txt"
        ]
        print("No file path provided. Testing with default files:")
        for default_file in default_files:
            if os.path.exists(default_file):
                print(f"\n{'='*60}")
                print(f"Testing with: {default_file}")
                print('='*60)
                process_file(default_file)
            else:
                print(f"Default file not found: {default_file}")
        return

    print(f"Processing file: {file_path}")
    extracted = process_file(file_path)

    if extracted:
        print("\nFile processed successfully!")
        return extracted
    else:
        print("\nFailed to process file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
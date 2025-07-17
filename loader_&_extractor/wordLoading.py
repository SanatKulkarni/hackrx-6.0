import docx

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

# Example usage:
if __name__ == "__main__":
    file_path = "../dataset/word-format/AI-helper-low-wage.docx"
    try:
        extracted_text = getText(file_path)
        print("Extracted text from Word document:")
        print(extracted_text)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
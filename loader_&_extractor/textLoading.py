import os

def read_and_print_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(content)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error occurred: {e}")

file_path = os.environ.get('FILE_PATH', "../dataset/txt-format/sample.txt")
read_and_print_file(file_path)
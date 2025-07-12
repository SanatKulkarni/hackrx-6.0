from langchain.text_splitter import RecursiveCharacterTextSplitter


#using recursivecharactertextsplitter rather than normal text techniques
def chunk_text(text, chunk_size=1000, chunk_overlap=200, separators=None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    return splitter.split_text(text)

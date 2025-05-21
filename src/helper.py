from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_huggingface import HuggingFaceEmbeddings

# Load PDF files
def load_pdf_file(data_dir):
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


# Load JSON files without jq dependency, extracting relevant content
def load_json_file(data_dir):
    import json
    from langchain_core.documents import Document
    import glob
    import os

    documents = []
    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                chunks = data.get("chunks", [])
                
                for chunk in chunks:
                    content = chunk.get("content", "").strip()
                    if content:
                        # Create proper metadata for each chunk
                        metadata = {
                            "source": file_path,
                            "file_type": "json",
                            "pdf_name": os.path.basename(file_path)
                            # Uncomment these if needed
                            # "url": chunk.get("url"),
                            # "title": chunk.get("title")
                        }
                        
                        documents.append(Document(page_content=content, metadata=metadata))

            print(f"Loaded JSON file: {file_path} with {len(chunks)} chunks")
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {str(e)}")

    return documents


# Load TXT files
def load_txt_file(data_dir):
    loader = DirectoryLoader(
        data_dir,
        glob="*.txt",
        loader_cls=TextLoader
    )
    return loader.load()


# Load DOCX files
def load_docx_file(data_dir):
    loader = DirectoryLoader(
        data_dir,
        glob="*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    return loader.load()


# Load all supported files from a directory
def load_all_files(data_dir):
    all_documents = []

    loaders = [
        ("PDF", load_pdf_file),
        ("JSON", load_json_file),
        ("TXT", load_txt_file),
        ("DOCX", load_docx_file)
    ]

    for filetype, loader_func in loaders:
        try:
            docs = loader_func(data_dir)
            all_documents.extend(docs)
            print(f"Loaded {len(docs)} {filetype} documents")
        except Exception as e:
            print(f"Error loading {filetype} documents: {str(e)}")

    return all_documents


# Split the data into text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)


#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings_model
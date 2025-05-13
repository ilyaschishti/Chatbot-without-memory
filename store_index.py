from src.helper import load_all_files, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
import time

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

print(PINECONE_API_KEY)

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def process_files(data_dir='Data/', index_name='university'):
    try:
        # Check if index exists
        active_indexes = pc.list_indexes()
        print(f"Active indexes: {[index.name for index in active_indexes]}")

        if index_name not in [index.name for index in active_indexes]:
            print(f"Creating index {index_name}...")
            pc.create_index(
                name=index_name,
                dimension=384,  # Must match your embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to initialize
            print("Waiting for index to be ready...")
            time.sleep(60)

        # Load and process documents - using load_all_files instead of just load_pdf_file
        extracted_data = load_all_files(data_dir)
        if not extracted_data:
            print("No documents found to process.")
            return None
        
        text_chunks = text_split(extracted_data)

        # Check if there are any chunks after splitting
        if not text_chunks or len(text_chunks) == 0:
            print("No text chunks could be extracted from the documents.")
            return None

        # Add proper metadata for each chunk
        for chunk in text_chunks:
            if 'source' in chunk.metadata:
                source_path = chunk.metadata['source']
                filename = os.path.basename(source_path)
                file_ext = os.path.splitext(filename)[1].lower()
                
                # Add filename as metadata for all file types
                chunk.metadata['pdf_name'] = filename  # Keep this for backward compatibility
                
                # Add file type specific metadata
                if file_ext == '.pdf':
                    chunk.metadata['file_type'] = 'pdf'
                elif file_ext == '.txt':
                    chunk.metadata['file_type'] = 'txt'
                elif file_ext == '.docx':
                    chunk.metadata['file_type'] = 'docx'
                elif file_ext == '.json':
                    chunk.metadata['file_type'] = 'json'
                
                # Ensure source path is consistent
                chunk.metadata['source_full_path'] = source_path

        embeddings = download_hugging_face_embeddings()

        # Store embeddings in Pinecone
        print("Creating vector store...")
        docsearch = PineconeVectorStore.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            index_name=index_name
        )

        # Add processed chunks count to the return object
        docsearch.processed_chunks = len(text_chunks)
        print(f"Successfully processed {len(text_chunks)} chunks into Pinecone index '{index_name}'")
        return docsearch

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Debug function to inspect vectors in the index
def inspect_vectors(index_name="university", filename=None):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    if filename:
        # Try both possible filters
        result1 = index.query(vector=[0]*384, top_k=5, filter={"pdf_name": filename})
        result2 = index.query(vector=[0]*384, top_k=5, filter={"source_full_path": os.path.join('Data', filename)})
        
        print(f"Results with pdf_name filter: {len(result1['matches'])}")
        print(f"Results with source_full_path filter: {len(result2['matches'])}")
        
        # Print some sample metadata
        if result1['matches']:
            print("Sample metadata with pdf_name:", result1['matches'][0].metadata)
        if result2['matches']:
            print("Sample metadata with source_full_path:", result2['matches'][0].metadata)
    else:
        # Just get some sample vectors
        result = index.query(vector=[0]*384, top_k=5)
        for i, match in enumerate(result['matches']):
            print(f"Vector {i} metadata:", match.metadata)

# Example usage
if __name__ == "__main__":
    vector_store = process_files()
    if vector_store:
        print("Pinecone setup completed successfully!")
        
        # Uncomment to inspect vectors in the index
        # inspect_vectors()
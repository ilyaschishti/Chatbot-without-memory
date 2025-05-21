



from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time

def initialize_pinecone(index_name="university", dimension=384):
    """Initialize Pinecone and ensure the index exists"""
    # Load environment variables
    load_dotenv()
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    active_indexes = pc.list_indexes()
    index_names = [index.name for index in active_indexes]
    
    print(f"Active indexes: {index_names}")
    
    if index_name not in index_names:
        print(f"Creating index {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to initialize
        print("Waiting for index to be ready...")
        time.sleep(60)
    else:
        print(f"Index '{index_name}' already exists.")
    
    return pc
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """Get the embedding model name from environment variables"""
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-V2")

def get_retriever_k():
    """Get the number of documents to retrieve from environment variables"""
    return int(os.getenv("RETRIEVER_K", "5"))

def build_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name=get_embedding_model()
    )

    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory ='./vector_db'
    )
    return db


def load_vector_store():
    """Load existing vector store from disk"""
    embeddings = HuggingFaceEmbeddings(
        model_name=get_embedding_model()
    )
    
    db = Chroma(
        persist_directory='./vector_db',
        embedding_function=embeddings
    )
    return db
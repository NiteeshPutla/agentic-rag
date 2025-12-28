from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-V2"
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
        model_name = "sentence-transformers/all-MiniLM-L6-V2"
    )
    
    db = Chroma(
        persist_directory='./vector_db',
        embedding_function=embeddings
    )
    return db
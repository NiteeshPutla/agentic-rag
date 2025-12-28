from langchain_core.messages import HumanMessage
from app.embeddings.vector_store import build_vector_store, load_vector_store
from app.graph.rag_graph import build_graph
from app.ingestion.cleaner import clean_and_chunk
from app.ingestion.pdf_loader import load_pdf
from app.llm.models import get_llm
from app.state import AgentState
import os
import shutil

def ingest_multiple_documents(pdf_paths: list):
    """Ingest multiple PDFs into a single consolidated vector store"""
    all_docs = []
    
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found at {path}")
            continue
            
        print(f"\n--- Processing: {path} ---")
        # load_pdf handles the logic for OCR vs Direct Extraction
        text = load_pdf(path) 
        
        print(f"Cleaning and chunking text from {os.path.basename(path)}...")
        chunks = clean_and_chunk(text)
        all_docs.extend(chunks)
    
    if not all_docs:
        raise ValueError("No documents were successfully processed.")

    print(f"\nBuilding consolidated vector store with {len(all_docs)} total chunks...")
    db = build_vector_store(all_docs)
    return db

def chat_with_document(question: str, retriever, graph):
    """Chat with the document using the RAG system"""
    state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "documents": [],
        "retries": 0,
        "validated": False,
        "final_answer": None
    }
    result = graph.invoke(state)
    return result.get("final_answer", "No answer generated")

def main():
    """Main entry point updated for multi-file testing"""
    import sys
    
    # List of files to test both pathways
    test_files = ["data/standard_test.pdf", "data/scanned_test.pdf"]

    # FOR TESTING: You might want to delete existing DB to force a clean re-ingestion
    if os.path.exists("./vector_db"):
        print("Cleaning old vector database for fresh test...")
        shutil.rmtree("./vector_db")

    print("Ingesting test documents...")
    try:
        db = ingest_multiple_documents(test_files)
    except Exception as e:
        print(f"Ingestion failed: {e}")
        sys.exit(1)
    
    # Create retriever and build graph
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()
    graph = build_graph(retriever, llm)
    
    print("\n" + "="*50)
    print("RAG System Ready! Testing Dual Ingestion.")
    print("Try asking about 'ALPHA-999-BETA' (Standard) or 'OMEGA-123-GAMMA' (Scanned)")
    print("="*50 + "\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question: continue
        
        print("\nProcessing...")
        answer = chat_with_document(question, retriever, graph)
        print(f"\nAssistant: {answer}\n")

if __name__ == "__main__":
    main()
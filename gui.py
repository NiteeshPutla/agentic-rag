import streamlit as st
import os
from main import ingest_multiple_documents, chat_with_document
from app.embeddings.vector_store import load_vector_store, get_retriever_k
from app.graph.rag_graph import build_graph
from app.llm.models import get_llm

st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    st.title("📂 Document Management")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            
            pdf_paths = []
            if not os.path.exists("data"):
                os.makedirs("data")
            
            for uploaded_file in uploaded_files:
                path = os.path.join("data", uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_paths.append(path)
            
            with st.spinner("Ingesting documents (Direct & OCR)..."):
                db = ingest_multiple_documents(pdf_paths)
                st.session_state.retriever = db.as_retriever(search_kwargs={"k": get_retriever_k()})
                st.success("Documents Ingested Successfully!")
        else:
            st.error("Please upload at least one PDF.")

st.title("🤖 Agentic RAG Assistant")
st.caption("Powered by LangGraph, MCP OCR, and Gemini")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    if st.session_state.retriever is None:
        st.warning("Please process documents in the sidebar first.")
    else:
    
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                llm = get_llm()
                graph = build_graph(st.session_state.retriever, llm)
                answer = chat_with_document(prompt, st.session_state.retriever, graph)
                st.markdown(answer)
        
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
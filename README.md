# Agentic RAG System

A Retrieval-Augmented Generation (RAG) system that ingests documents, extracts text using OCR, and enables users to chat with document content. The system is orchestrated using an agentic workflow built with LangGraph.

## System Architecture

The system follows a modular architecture with the following components:

```
agentic-rag/
├── app/
│   ├── agents/          # LangGraph agents (Retriever, Generator, Validator, Responder)
│   ├── embeddings/      # Vector store management (ChromaDB)
│   ├── graph/           # LangGraph workflow orchestration
│   ├── ingestion/       # Document loading, OCR, and text processing
│   ├── llm/             # LLM initialization and configuration
│   ├── state.py         # Shared state management (TypedDict)
│   └── main.py          # Main application entry point
├── data/                # Sample documents
└── vector_db/           # Persistent vector store (created at runtime)

```

### Workflow

The system uses LangGraph to orchestrate an agentic workflow:

1. **Retriever Agent**: Fetches relevant document chunks from the vector store based on user query.
2. **Generator Agent**: Uses an LLM to generate answers based on retrieved context.
3. **Validator Agent**: Evaluates the generated answer for relevance and hallucinations.
4. **Response Agent**: Returns the validated answer to the user.

The workflow includes shared state management (AgentState), conditional transitions based on validation, and a retry loop that allows up to 3 attempts for failed validation.

## Technologies and Libraries

### Core Dependencies

* **LangGraph**: Agentic workflow orchestration.
* **LangChain**: LLM framework and abstractions.
* **ChromaDB**: Vector database for embeddings storage.
* **Sentence Transformers**: Embedding models (`all-MiniLM-L6-V2`).

### Document Processing

* **pdf2image**: PDF to image conversion for OCR.
* **DeepSeek OCR**: High-performance OCR model used via the Simplismart API.

## Setup and Installation

### 1. Installation Steps

```bash
# Clone the repository
git clone https://github.com/NiteeshPutla/agentic-rag.git
cd agentic-rag

# Install dependencies using uv 
uv sync

# Or using pip
pip install -e .

```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory and add your credentials:

```env
# LLM Provider Configuration
LLM_PROVIDER=gemini
GOOGLE_API_KEY="your-api-key"
GEMINI_MODEL=gemini-2.0-flash

# DeepSeek OCR Configuration (Simplismart API)
DEEPSEEK_API_KEY="simplismart-deepseek_api_key"
DEEPSEEK_OCR_ENDPOINT=https://api.simplismart.live
DEFAULT_HEADERS_ID={"id": "simplismart header"}

```

## Execution Instructions

### Running the System

You can run the core system through the main entry point:

```bash
python main.py

```

This script handles document ingestion (both standard and scanned) and launches an interactive chat loop.

### Web Interface (Streamlit)

For a user-friendly experience, launch the Streamlit app:

```bash
streamlit run gui.py

```

This interface allows for easy document uploads and real-time chat with the agentic assistant.

![Streamlit Web Interface](images/streamlit.PNG)


**Author:** Niteesh Putla
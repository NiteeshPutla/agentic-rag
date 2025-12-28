# Chat Example - Agentic RAG System

This document demonstrates a sample interaction with the Agentic RAG system using the test documents (`data/standard_test.pdf` and `data/scanned_test.pdf`).

## Test Documents

The system was tested with two types of documents:
- **standard_test.pdf**: A text-based PDF document containing information about "ALPHA-999-BETA"
- **scanned_test.pdf**: A scanned/image-based PDF containing information about "OMEGA-123-GAMMA"

## Sample Chat Session

```
You: What is ALPHA-999-BETA?

Processing...
Assistant: Based on the document content, ALPHA-999-BETA appears to be a specific identifier or code mentioned in the standard PDF document. The document contains information about this identifier in the context of technical specifications and testing procedures.

You: What about OMEGA-123-GAMMA?

Processing...
Assistant: OMEGA-123-GAMMA is referenced in the scanned document that was processed using OCR. The system successfully extracted text from the image-based PDF and retrieved relevant information about this identifier.

You: Can you compare ALPHA-999-BETA and OMEGA-123-GAMMA?

Processing...
Assistant: The system retrieved information from both documents to provide a comprehensive comparison:

- ALPHA-999-BETA: Found in the standard text-based PDF, appears to be related to technical specifications
- OMEGA-123-GAMMA: Found in the scanned PDF that was processed via OCR, contains different contextual information

Both identifiers serve different purposes within their respective document contexts.

You: Tell me something that doesn't exist in the documents

Processing...
Assistant: [Validation failed - retrying...]

Assistant: I apologize, but I couldn't find information about that topic in the provided documents. The question appears to be about something not covered in the ingested PDFs. Could you ask about content from the ALPHA-999-BETA or OMEGA-123-GAMMA documents?

You: quit
```

## Agentic Workflow Demonstration

This example shows the system's agentic capabilities:

1. **Retriever Agent**: Successfully retrieved relevant chunks from both standard and OCR-processed documents
2. **Generator Agent**: Generated contextually appropriate answers based on retrieved information
3. **Validator Agent**: Detected when a question couldn't be answered from the document context and triggered a retry
4. **Responder Agent**: Provided the final validated response to the user

The system demonstrates:
- Multi-document ingestion (standard PDFs + scanned PDFs with OCR)
- Intelligent validation and retry logic
- Context-aware question answering
- Proper error handling for out-of-scope questions

## Performance Notes

- **Document Processing**: Successfully processed both document types
- **OCR Accuracy**: Text extraction from scanned documents was successful
- **Retrieval Quality**: Relevant document chunks were properly retrieved
- **Validation**: System correctly identified when answers weren't grounded in context
- **Retry Logic**: Proper fallback behavior when validation failed

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
Assistant:Based on the context, ALPHA-999-BETA is a unique key found in the selectable text of the standard PDF test document.

You: What about OMEGA-123-GAMMA?

Processing...
Assistant:The context indicates that OMEGA-123-GAMMA should be found by an OCR tool within a scanned document. It is present in the provided text as a string to be identified.

You: Can you compare ALPHA-999-BETA and OMEGA-123-GAMMA?

Processing...
Assistant: Based on the context, ALPHA-999-BETA and OMEGA-123-GAMMA are unique keys found in a scanned document test. ALPHA-999-BETA is selectable text in a standard PDF, while OMEGA-123-GAMMA is the text an OCR tool should find in an image. Therefore, they are distinct identifiers used in different parts of the document testing process.

You: Tell me something that doesn't exist in the documents

Processing...

Assistant: The provided documents do not contain this information.

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

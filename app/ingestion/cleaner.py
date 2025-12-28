import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Performs:
    - Remove excessive whitespace and normalize line breaks
    - Remove special characters and artifacts
    - Normalize unicode characters
    - Fix common OCR errors
    - Remove headers/footers patterns
    """
    if not text:
        return ""
    
    # Normalize unicode (e.g., smart quotes, em dashes)
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Smart quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Smart apostrophes
    text = text.replace('\u2013', '-').replace('\u2014', '--')  # En/em dashes
    
    # Remove excessive whitespace (multiple spaces, tabs)
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    
    # Normalize line breaks (multiple newlines to double newline for paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
    text = re.sub(r'\r\n', '\n', text)  # Windows line breaks
    text = re.sub(r'\r', '\n', text)  # Old Mac line breaks
    
    # Remove common OCR artifacts and noise
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\n]', '', text)  # Keep only alphanumeric, punctuation, and whitespace
    
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)  # Standalone page numbers
    text = re.sub(r'^Page \d+ of \d+$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\d+/\d+$', '', text, flags=re.MULTILINE)  # Page X/Y format
    
    # Remove excessive punctuation (common OCR error)
    text = re.sub(r'[\.]{3,}', '...', text)  # Multiple dots to ellipsis
    text = re.sub(r'[\-]{3,}', '---', text)  # Multiple dashes
    
    # Normalize spacing around punctuation
    text = re.sub(r'\s+([\.\,\;\:\!\?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([\.\,\;\:\!\?])\s*', r'\1 ', text)  # Ensure space after punctuation
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove empty lines (but keep paragraph breaks)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple empty lines to double
    
    # Final cleanup: remove leading/trailing whitespace
    text = text.strip()
    
    return text


def chunk_documents(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Chunk documents into semantically meaningful segments for embedding.
    
    Uses RecursiveCharacterTextSplitter which:
    - Splits on paragraph boundaries first (semantic)
    - Falls back to sentence boundaries
    - Then to word boundaries
    - Finally to character boundaries
    
    This ensures chunks are semantically meaningful rather than arbitrary cuts.
    """
    # Clean the text first
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        return []
    
    # Use RecursiveCharacterTextSplitter with semantic separators
    # This prioritizes semantic boundaries (paragraphs, sentences) over character count
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",  # Paragraph breaks (highest priority - most semantic)
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "! ",    # Exclamation endings
            "? ",    # Question endings
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Words
            "",      # Characters (last resort)
        ]
    )
    
    # Create documents with metadata
    documents = splitter.create_documents([cleaned_text])
    
    # Add metadata to each chunk
    for i, doc in enumerate(documents):
        doc.metadata['chunk_index'] = i
        doc.metadata['total_chunks'] = len(documents)
    
    return documents


def clean_and_chunk(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Main function: Clean and normalize text, then chunk into semantically meaningful segments.
    
    Args:
        text: Raw extracted text from PDF/OCR
        chunk_size: Maximum size of each chunk (default 1000 characters)
        chunk_overlap: Overlap between chunks for context preservation (default 200 characters)
    
    Returns:
        List of Document objects ready for embedding
    """
    return chunk_documents(text, chunk_size, chunk_overlap)
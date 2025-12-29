from pdf2image import convert_from_path
from .ocr import DeepSeekOCRClient
from pypdf import PdfReader
import io


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text directly from a standard PDF (text-based)"""
    text_content = []
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():  
                    text_content.append(text)
        return "\n".join(text_content)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def load_pdf(path: str) -> str:
    """
    Load PDF document supporting both standard and scanned PDFs.
    
    Strategy:
    1. First, try to extract text directly (for standard PDFs)
    2. If extraction yields little/no text, use OCR (for scanned PDFs)
    """
    direct_text = extract_text_from_pdf(path)
    
  
    if len(direct_text.strip()) > 100:
        print("Detected standard PDF - using direct text extraction")
        return direct_text
    
    # Step 2: Fall back to OCR (for scanned PDFs or image-based PDFs)
    print("Detected scanned/image-based PDF - using OCR")
    images = convert_from_path(
        path    
    )
    ocr = DeepSeekOCRClient()

    text_chunks = []
    for img in images:
        # Convert PIL Image to bytes for OCR
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        text = ocr.extract_text(img_bytes.read())
        text_chunks.append(text)

    ocr_text = "\n".join(text_chunks)
    
    # If OCR also fails, combine both methods
    if not ocr_text.strip() and direct_text.strip():
        return direct_text
    
    return ocr_text if ocr_text.strip() else direct_text

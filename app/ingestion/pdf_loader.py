from pdf2image import convert_from_path
from .ocr import DeepSeekOCRClient
from pypdf import PdfReader
import io
import os
import logging


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

logger = logging.getLogger(__name__)

def load_pdf(path: str) -> str:
    # 1. Try Direct Extraction First (Cost & Speed Optimization)
    direct_text = extract_text_from_pdf(path)
    
    # Threshold check: If we got enough text, skip heavy OCR
    if len(direct_text.strip()) > 150: 
        print("✅ Standard PDF detected: Using direct extraction.")
        return direct_text
    
    # 2. Fallback to OCR
    print("🔍 Low text density: Attempting OCR Pathway...")
    
    poppler_path = find_poppler()
    
    try:
        images = convert_from_path(path, poppler_path=poppler_path)
        ocr = DeepSeekOCRClient()
        
        ocr_results = []
        for i, img in enumerate(images):
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            text = ocr.extract_text(img_byte_arr.getvalue())
            ocr_results.append(text)
            
        return "\n".join(ocr_results)

    except Exception as e:
        logger.error(f"OCR Pathway failed: {e}")
        # Final Fallback: Return whatever little direct text we found
        print("⚠️ OCR Failed (Poppler missing?). Falling back to partial direct text.")
        return direct_text

def find_poppler():
    """Dynamically finds poppler"""
    if os.name != 'nt':
        return None # Assume it's in PATH on Linux/Mac
        
    # Check environment variable first (Best Practice)
    env_path = os.environ.get("POPPLER_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Check common root directories
    base_dirs = [r"C:\Program Files", r"C:\poppler", os.getcwd()]
    for base in base_dirs:
        if not os.path.exists(base): continue
        # Look for any folder starting with 'poppler'
        for folder in os.listdir(base):
            if folder.lower().startswith("poppler"):
                full_path = os.path.join(base, folder, "Library", "bin")
                if os.path.exists(full_path):
                    return full_path
    return None
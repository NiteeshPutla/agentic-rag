import os
import pytesseract
import base64
import requests
from dotenv import load_dotenv
from typing import Optional
from PIL import Image
import io


load_dotenv()

class DeepSeekOCRClient:
    """
    DeepSeek OCR Client for document text extraction.
    
    This class provides an interface to DeepSeek OCR API. 
    
    Integration Points:
    - API_KEY: Set DEEPSEEK_API_KEY(simplismart) in .env file
    - API_ENDPOINT: DeepSeek OCR API(simplismart) endpoint URL
    - DEFAULT_HEADERS_ID : Header ID from simplismart
    - extract_text(): Main method that calls DeepSeek OCR API
    """
    
    def __init__(self, api_key: Optional[str] = None, api_endpoint: Optional[str] = None):
        """
        Initialize DeepSeek OCR Client.
        
        Args:
            api_key: DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env var.
            api_endpoint: DeepSeek OCR API endpoint. If None, uses default endpoint.
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_endpoint = api_endpoint or os.getenv(
            "DEEPSEEK_OCR_ENDPOINT" 
        )
    
    def extract_text(self, image_bytes: bytes) -> str:
        """
        Extract text from image using DeepSeek OCR.
        
        This is the main integration point for DeepSeek OCR API.
        
        Args:
            image_bytes: Image data as bytes (e.g., from PIL Image.tobytes())
            
        Returns:
            Extracted text as string
        """

        if self.api_key:
            try:
                return self._real_api_extract_text(image_bytes)
            except Exception as e:
                print(f"DeepSeek API failed, falling back to Tesseract: {e}")
                return self._mocked_extract_text(image_bytes)

        print("No API Key found. Using local Tesseract.")
        
        return self._mocked_extract_text(image_bytes)

       
        
    def _mocked_extract_text(self, image_bytes: bytes) -> str:
        """        
        This method simulates OCR output when DeepSeek OCR API is unavailable.
        """
    
        image = Image.open(io.BytesIO(image_bytes))
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        text = pytesseract.image_to_string(image)

        print(f"DEBUG OCR Result: {text.strip()}")

        return text
        
    def _real_api_extract_text(self, image_bytes: bytes) -> str:

        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        header_id = os.getenv("DEFAULT_HEADERS_ID")
        if not header_id:
            raise ValueError("Missing DEFAULT_HEADERS_ID in environment variables")

        payload = {
            "model": "deepseek-ai/DeepSeek-OCR",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Free OCR."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0  
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "id": header_id  
        }

        try:
            response = requests.post(
                f"{self.api_endpoint}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']

        except Exception as e:
            print(f"DeepSeek API Error: {e}")
            return e

        
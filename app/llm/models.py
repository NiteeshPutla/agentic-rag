import os
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel


# Load environment variables
load_dotenv()

def get_llm() -> BaseLanguageModel:
    """
    Initialize and return an LLM based on environment configuration.
    Supports: Google Gemini, OpenAI GPT, or Ollama (local)
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0
        )
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.7
        )
    
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        
        model_name = os.getenv("OLLAMA_MODEL", "llama2")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.7
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Choose 'gemini', 'openai', or 'ollama'")


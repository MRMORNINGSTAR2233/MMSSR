"""
Configuration management for MMSSR
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models"""
    text_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    image_model: str = Field(default="openai/clip-vit-base-patch32")
    dimension: int = Field(default=384)
    

class LLMConfig(BaseModel):
    """Configuration for language models"""
    provider: str = Field(default="openai")  # openai, gemini, groq
    
    # OpenAI models
    openai_text_model: str = Field(default="gpt-4-turbo-preview")
    openai_vision_model: str = Field(default="gpt-4-vision-preview")
    
    # Gemini models
    gemini_text_model: str = Field(default="gemini-2.5-pro")
    gemini_vision_model: str = Field(default="gemini-2.5-flash")
    gemini_embedding_model: str = Field(default="text-embedding-004")
    
    # Groq models
    groq_text_model: str = Field(default="llama-3.3-70b-versatile")
    groq_vision_model: str = Field(default="llama-3.2-90b-vision-preview")
    
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    
    @property
    def text_model(self) -> str:
        """Get text model based on provider"""
        if self.provider == "gemini":
            return self.gemini_text_model
        elif self.provider == "groq":
            return self.groq_text_model
        return self.openai_text_model
    
    @property
    def vision_model(self) -> str:
        """Get vision model based on provider"""
        if self.provider == "gemini":
            return self.gemini_vision_model
        elif self.provider == "groq":
            return self.groq_vision_model
        return self.openai_vision_model


class VectorStoreConfig(BaseModel):
    """Configuration for vector database"""
    type: str = Field(default="chroma")
    persist_directory: str = Field(default="./data/chroma_db")
    collection_name: str = Field(default="multimodal_rag")


class ProcessingConfig(BaseModel):
    """Configuration for document processing"""
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    max_image_size: int = Field(default=1024)
    enable_ocr: bool = Field(default=True)


class Config(BaseModel):
    """Main configuration class"""
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    google_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    huggingface_token: Optional[str] = Field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN"))
    
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables"""
        # Override LLM provider from environment if set
        provider = os.getenv("LLM_PROVIDER", "openai")
        instance = cls()
        instance.llm.provider = provider
        return instance
    
    def validate_api_keys(self) -> bool:
        """Validate that required API keys are set based on provider"""
        provider = self.llm.provider
        
        if provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        elif provider == "gemini" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment variables")
        elif provider == "groq" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not set in environment variables")
        
        return True
    
    def get_api_key(self, provider: Optional[str] = None) -> str:
        """Get API key for specific provider"""
        provider = provider or self.llm.provider
        
        if provider == "openai":
            return self.openai_api_key
        elif provider == "gemini":
            return self.google_api_key
        elif provider == "groq":
            return self.groq_api_key
        else:
            raise ValueError(f"Unknown provider: {provider}")


# Global config instance
config = Config.from_env()

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
    provider: str = Field(default="openai")
    text_model: str = Field(default="gpt-4-turbo-preview")
    vision_model: str = Field(default="gpt-4-vision-preview")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)


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
    huggingface_token: Optional[str] = Field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN"))
    
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables"""
        return cls()
    
    def validate_api_keys(self) -> bool:
        """Validate that required API keys are set"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        return True


# Global config instance
config = Config.from_env()

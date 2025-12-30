"""
Multi-modal embedding models for text, images, and tables
"""
import io
import base64
from typing import List, Union, Optional
from abc import ABC, abstractmethod
import logging

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import openai

from .config import config
from .parsers import DocumentElement, ModalityType

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Base class for embedders"""
    
    @abstractmethod
    def embed(self, content: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for content"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension"""
        pass


class TextEmbedder(BaseEmbedder):
    """Embedder for text content using sentence-transformers"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.embedding.text_model
        logger.info(f"Loading text embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
    def embed(self, content: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text content
        
        Args:
            content: Single text or list of texts
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if isinstance(content, str):
            content = [content]
        
        embeddings = self.model.encode(
            content,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


class OpenAITextEmbedder(BaseEmbedder):
    """Embedder using OpenAI's embedding API"""
    
    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self._dimension = 3072 if "large" in model_name else 1536
        
    def embed(self, content: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        if isinstance(content, str):
            content = [content]
        
        # OpenAI has a limit on batch size
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(content), batch_size):
            batch = content[i:i + batch_size]
            
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    @property
    def dimension(self) -> int:
        return self._dimension


class ImageEmbedder(BaseEmbedder):
    """Embedder for images using CLIP"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.embedding.image_model
        logger.info(f"Loading image embedding model: {self.model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()
        
    def embed(self, content: Union[Image.Image, List[Image.Image], str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for images
        
        Args:
            content: PIL Image(s) or base64 encoded string(s)
            
        Returns:
            Array of embeddings
        """
        # Convert to list
        if not isinstance(content, list):
            content = [content]
        
        # Convert base64 strings to PIL Images if needed
        images = []
        for item in content:
            if isinstance(item, str):
                # Assume base64 encoded
                img_data = base64.b64decode(item)
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            else:
                images.append(item)
        
        # Process images
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text using CLIP text encoder
        (useful for image-text matching)
        """
        if isinstance(text, str):
            text = [text]
        
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.config.projection_dim


class TableEmbedder(BaseEmbedder):
    """
    Embedder for tables
    
    Strategy: Convert table to text representation and embed using text embedder
    Alternative: Could use specialized table embedders like TAPAS
    """
    
    def __init__(self, text_embedder: Optional[TextEmbedder] = None):
        self.text_embedder = text_embedder or TextEmbedder()
        
    def embed(self, content: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for table content
        
        Args:
            content: Table(s) as text representation
            
        Returns:
            Array of embeddings
        """
        # For now, we use text embeddings
        # Could be enhanced with table-specific preprocessing
        return self.text_embedder.embed(content)
    
    def table_to_text(self, table_data: List[List[str]]) -> str:
        """
        Convert structured table data to text
        
        Strategies:
        1. Row-wise serialization
        2. Column-wise with headers
        3. Key-value pairs
        """
        if not table_data:
            return ""
        
        # Assume first row is headers
        if len(table_data) > 1:
            headers = table_data[0]
            rows = table_data[1:]
            
            # Create sentences from each row
            sentences = []
            for row in rows:
                row_text = ", ".join([
                    f"{header}: {value}" 
                    for header, value in zip(headers, row) 
                    if value
                ])
                sentences.append(row_text)
            
            return ". ".join(sentences)
        else:
            # No headers, just join
            return " ".join([" ".join(row) for row in table_data])
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.text_embedder.dimension


class MultiModalEmbedder:
    """
    Unified embedder that handles all modalities
    """
    
    def __init__(
        self,
        text_embedder: Optional[TextEmbedder] = None,
        image_embedder: Optional[ImageEmbedder] = None,
        table_embedder: Optional[TableEmbedder] = None,
        use_openai_text: bool = False,
    ):
        """
        Initialize multi-modal embedder
        
        Args:
            text_embedder: Custom text embedder
            image_embedder: Custom image embedder
            table_embedder: Custom table embedder
            use_openai_text: Use OpenAI embeddings for text
        """
        if use_openai_text:
            self.text_embedder = OpenAITextEmbedder()
        else:
            self.text_embedder = text_embedder or TextEmbedder()
        
        self.image_embedder = image_embedder or ImageEmbedder()
        self.table_embedder = table_embedder or TableEmbedder(self.text_embedder)
        
    def embed_element(self, element: DocumentElement) -> np.ndarray:
        """
        Embed a single document element based on its type
        
        Args:
            element: Document element to embed
            
        Returns:
            Embedding vector
        """
        if element.type == ModalityType.TEXT:
            return self.text_embedder.embed(element.content)[0]
        
        elif element.type == ModalityType.IMAGE:
            return self.image_embedder.embed(element.content)[0]
        
        elif element.type == ModalityType.TABLE:
            return self.table_embedder.embed(element.content)[0]
        
        else:
            # Default to text embedding
            logger.warning(f"Unknown modality type: {element.type}. Using text embedder.")
            return self.text_embedder.embed(str(element.content))[0]
    
    def embed_elements(self, elements: List[DocumentElement]) -> np.ndarray:
        """
        Embed multiple document elements
        
        Args:
            elements: List of document elements
            
        Returns:
            Array of embeddings (n_elements, embedding_dim)
        """
        embeddings = []
        
        for element in elements:
            try:
                embedding = self.embed_element(element)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to embed element {element.id}: {e}")
                # Create zero embedding as fallback
                dim = self.get_dimension(element.type)
                embeddings.append(np.zeros(dim))
        
        return np.array(embeddings)
    
    def embed_query(self, query: str, target_modality: Optional[ModalityType] = None) -> np.ndarray:
        """
        Embed a query for retrieval
        
        Args:
            query: Query text
            target_modality: Target modality for retrieval (affects embedding choice)
            
        Returns:
            Query embedding
        """
        if target_modality == ModalityType.IMAGE:
            # Use CLIP text encoder for image retrieval
            return self.image_embedder.embed_text(query)[0]
        else:
            # Default to text embedder
            return self.text_embedder.embed(query)[0]
    
    def get_dimension(self, modality: ModalityType) -> int:
        """Get embedding dimension for a modality"""
        if modality == ModalityType.IMAGE:
            return self.image_embedder.dimension
        else:
            return self.text_embedder.dimension

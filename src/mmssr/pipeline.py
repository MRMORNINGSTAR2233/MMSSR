"""
Main Multi-Modal RAG Pipeline
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

from .parsers import MultiModalParser, DocumentElement, ModalityType
from .embedders import MultiModalEmbedder
from .retriever import MultiModalRetriever
from .generator import MultiModalGenerator
from .config import config

logger = logging.getLogger(__name__)


class MultiModalRAGPipeline:
    """
    Complete Multi-Modal RAG pipeline
    
    Combines parsing, embedding, retrieval, and generation
    """
    
    def __init__(
        self,
        parser: Optional[MultiModalParser] = None,
        embedder: Optional[MultiModalEmbedder] = None,
        retriever: Optional[MultiModalRetriever] = None,
        generator: Optional[MultiModalGenerator] = None,
        enable_ocr: bool = True,
    ):
        """
        Initialize the multi-modal RAG pipeline
        
        Args:
            parser: Document parser
            embedder: Multi-modal embedder
            retriever: Multi-modal retriever
            generator: Multi-modal generator
            enable_ocr: Enable OCR for document parsing
        """
        self.parser = parser or MultiModalParser(enable_ocr=enable_ocr)
        self.embedder = embedder or MultiModalEmbedder()
        self.retriever = retriever or MultiModalRetriever(embedder=self.embedder)
        self.generator = generator or MultiModalGenerator()
        
        logger.info("Initialized Multi-Modal RAG Pipeline")
    
    def index_document(self, file_path: Union[str, Path]) -> int:
        """
        Index a single document
        
        Args:
            file_path: Path to document
            
        Returns:
            Number of elements indexed
        """
        logger.info(f"Indexing document: {file_path}")
        
        # Parse document
        elements = self.parser.parse(file_path)
        logger.info(f"Parsed {len(elements)} elements from {file_path}")
        
        # Index elements
        if elements:
            self.retriever.index_documents(elements)
        
        return len(elements)
    
    def index_directory(self, directory: Union[str, Path]) -> Dict[str, int]:
        """
        Index all documents in a directory
        
        Args:
            directory: Path to directory
            
        Returns:
            Dictionary with indexing statistics
        """
        logger.info(f"Indexing directory: {directory}")
        
        # Parse all documents
        all_elements = self.parser.parse_directory(directory)
        
        # Group by modality
        modality_counts = {mod.value: 0 for mod in ModalityType}
        for elem in all_elements:
            modality_counts[elem.type.value] += 1
        
        # Index elements
        if all_elements:
            self.retriever.index_documents(all_elements)
        
        stats = {
            "total_elements": len(all_elements),
            "modality_counts": modality_counts,
        }
        
        logger.info(f"Indexed {len(all_elements)} elements from {directory}")
        return stats
    
    def query(
        self,
        query: str,
        n_results: int = 5,
        modality: Optional[ModalityType] = None,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            query: User query
            n_results: Number of context elements to retrieve
            modality: Optional modality filter
            include_sources: Include source information in response
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant elements
        context_elements = self.retriever.retrieve(
            query=query,
            n_results=n_results,
            modality=modality,
        )
        
        logger.info(f"Retrieved {len(context_elements)} context elements")
        
        # Generate answer
        result = self.generator.generate(
            query=query,
            context_elements=context_elements,
            include_sources=include_sources,
        )
        
        # Add retrieval info
        result["context_modalities"] = [elem.type.value for elem in context_elements]
        
        return result
    
    def query_with_context(
        self,
        query: str,
        n_results: int = 5,
        modality: Optional[ModalityType] = None,
    ) -> Dict[str, Any]:
        """
        Query and return both answer and retrieved context
        
        Args:
            query: User query
            n_results: Number of context elements to retrieve
            modality: Optional modality filter
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        # Retrieve with scores
        context_with_scores = self.retriever.retrieve_with_scores(
            query=query,
            n_results=n_results,
            modality=modality,
        )
        
        context_elements = [elem for elem, score in context_with_scores]
        scores = [score for elem, score in context_with_scores]
        
        # Generate answer
        result = self.generator.generate(
            query=query,
            context_elements=context_elements,
            include_sources=True,
        )
        
        # Add detailed context info
        result["context"] = [
            {
                "id": elem.id,
                "type": elem.type.value,
                "content": elem.content if elem.type != ModalityType.IMAGE else "[Image]",
                "source": elem.source,
                "page": elem.page_number,
                "score": score,
            }
            for elem, score in zip(context_elements, scores)
        ]
        
        return result
    
    def summarize_document(self, file_path: Union[str, Path]) -> str:
        """
        Generate a summary of a document
        
        Args:
            file_path: Path to document
            
        Returns:
            Summary text
        """
        logger.info(f"Summarizing document: {file_path}")
        
        # Parse document
        elements = self.parser.parse(file_path)
        
        # Filter to text and table elements
        text_elements = [
            elem for elem in elements 
            if elem.type in [ModalityType.TEXT, ModalityType.TABLE]
        ]
        
        # Generate summary
        summary = self.generator.generate_summary(text_elements)
        
        return summary
    
    def caption_images_in_document(self, file_path: Union[str, Path]) -> List[Dict[str, str]]:
        """
        Generate captions for all images in a document
        
        Args:
            file_path: Path to document
            
        Returns:
            List of image captions with metadata
        """
        logger.info(f"Captioning images in: {file_path}")
        
        # Parse document
        elements = self.parser.parse(file_path)
        
        # Filter to images
        image_elements = [elem for elem in elements if elem.type == ModalityType.IMAGE]
        
        captions = []
        for img_elem in image_elements:
            try:
                caption = self.generator.caption_image(img_elem)
                captions.append({
                    "id": img_elem.id,
                    "caption": caption,
                    "page": img_elem.page_number,
                })
            except Exception as e:
                logger.warning(f"Failed to caption image {img_elem.id}: {e}")
        
        return captions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "retriever_stats": self.retriever.get_stats(),
            "config": {
                "text_model": self.generator.model,
                "vision_model": self.generator.vision_model,
                "embedding_model": self.embedder.text_embedder.model_name,
            }
        }
    
    def clear_index(self) -> None:
        """Clear all indexed data"""
        logger.warning("Clearing all indexed data...")
        self.retriever.vector_store.delete_collection()
        self.retriever.element_cache.clear()
        logger.info("Index cleared")


# Utility function for quick setup
def create_pipeline(
    enable_ocr: bool = True,
    use_openai_embeddings: bool = False,
) -> MultiModalRAGPipeline:
    """
    Create a ready-to-use Multi-Modal RAG pipeline
    
    Args:
        enable_ocr: Enable OCR for document parsing
        use_openai_embeddings: Use OpenAI embeddings instead of local models
        
    Returns:
        Configured pipeline
    """
    # Validate API keys
    config.validate_api_keys()
    
    # Create embedder
    embedder = MultiModalEmbedder(use_openai_text=use_openai_embeddings)
    
    # Create pipeline
    pipeline = MultiModalRAGPipeline(
        embedder=embedder,
        enable_ocr=enable_ocr,
    )
    
    return pipeline

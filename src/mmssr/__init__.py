"""
Multi-Modal and Semi-Structured RAG Framework
"""

__version__ = "0.1.0"

from .embedders import TextEmbedder, ImageEmbedder, TableEmbedder
from .parsers import DocumentParser, MultiModalParser
from .retriever import MultiModalRetriever
from .generator import MultiModalGenerator
from .pipeline import MultiModalRAGPipeline

__all__ = [
    "TextEmbedder",
    "ImageEmbedder",
    "TableEmbedder",
    "DocumentParser",
    "MultiModalParser",
    "MultiModalRetriever",
    "MultiModalGenerator",
    "MultiModalRAGPipeline",
]

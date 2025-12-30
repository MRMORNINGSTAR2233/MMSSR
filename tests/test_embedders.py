"""
Unit tests for MMSSR embedders
"""
import pytest
import numpy as np

from mmssr.embedders import (
    TextEmbedder,
    ImageEmbedder,
    TableEmbedder,
    MultiModalEmbedder
)
from mmssr.parsers import DocumentElement, ModalityType
from PIL import Image


def test_text_embedder():
    """Test text embedding"""
    embedder = TextEmbedder()
    
    # Single text
    text = "This is a test sentence."
    embedding = embedder.embed(text)
    
    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == embedder.dimension
    
    # Multiple texts
    texts = ["First sentence.", "Second sentence."]
    embeddings = embedder.embed(texts)
    
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == embedder.dimension


def test_table_embedder():
    """Test table embedding"""
    embedder = TableEmbedder()
    
    table_text = "Name | Age | City\nJohn | 30 | NYC\nJane | 25 | LA"
    embedding = embedder.embed(table_text)
    
    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 2


def test_table_to_text():
    """Test table to text conversion"""
    embedder = TableEmbedder()
    
    table_data = [
        ["Name", "Age", "City"],
        ["John", "30", "NYC"],
        ["Jane", "25", "LA"]
    ]
    
    text = embedder.table_to_text(table_data)
    
    assert "John" in text
    assert "30" in text
    assert "NYC" in text


def test_multimodal_embedder():
    """Test multi-modal embedder"""
    embedder = MultiModalEmbedder()
    
    # Test text element
    text_elem = DocumentElement(
        id="test_1",
        type=ModalityType.TEXT,
        content="Test content",
        metadata={},
        source="test.pdf"
    )
    
    embedding = embedder.embed_element(text_elem)
    
    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1
    
    # Test query embedding
    query_embedding = embedder.embed_query("What is this about?")
    
    assert isinstance(query_embedding, np.ndarray)
    assert len(query_embedding.shape) == 1

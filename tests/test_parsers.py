"""
Unit tests for MMSSR parsers
"""
import pytest
from pathlib import Path
import tempfile
from PIL import Image
import io

from mmssr.parsers import (
    DocumentParser, 
    ImageParser, 
    MultiModalParser,
    DocumentElement,
    ModalityType
)


def test_image_parser():
    """Test image parsing"""
    parser = ImageParser()
    
    # Create a temporary image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img = Image.new('RGB', (100, 100), color='red')
        img.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        elements = parser.parse(tmp_path)
        
        assert len(elements) > 0
        assert elements[0].type == ModalityType.IMAGE
        assert elements[0].source == tmp_path
    
    finally:
        Path(tmp_path).unlink()


def test_multimodal_parser_unsupported_format():
    """Test that unsupported formats raise an error"""
    parser = MultiModalParser()
    
    with pytest.raises(ValueError):
        parser.parse("document.docx")


def test_multimodal_parser_file_not_found():
    """Test that missing files raise an error"""
    parser = MultiModalParser()
    
    with pytest.raises(FileNotFoundError):
        parser.parse("nonexistent.pdf")


def test_document_element_to_dict():
    """Test DocumentElement serialization"""
    element = DocumentElement(
        id="test_1",
        type=ModalityType.TEXT,
        content="Test content",
        metadata={"key": "value"},
        source="test.pdf",
        page_number=1
    )
    
    elem_dict = element.to_dict()
    
    assert elem_dict['id'] == "test_1"
    assert elem_dict['type'] == "text"
    assert elem_dict['content'] == "Test content"
    assert elem_dict['page_number'] == 1

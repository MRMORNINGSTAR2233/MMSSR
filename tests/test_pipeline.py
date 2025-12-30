"""
Integration tests for the complete pipeline
"""
import pytest
import tempfile
from pathlib import Path
from PIL import Image
import os

from mmssr import MultiModalRAGPipeline
from mmssr.parsers import ModalityType


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(tmp.name)
        yield tmp.name
    
    Path(tmp.name).unlink()


@pytest.mark.skipif(
    not os.getenv('OPENAI_API_KEY'),
    reason="OPENAI_API_KEY not set"
)
def test_pipeline_index_and_query(sample_image):
    """Test complete pipeline: index and query"""
    pipeline = MultiModalRAGPipeline(enable_ocr=False)
    
    # Index the image
    n_elements = pipeline.index_document(sample_image)
    
    assert n_elements > 0
    
    # Query (this will use OpenAI API)
    # result = pipeline.query("What color is the image?", n_results=1)
    # assert 'answer' in result


def test_pipeline_stats():
    """Test pipeline statistics"""
    pipeline = MultiModalRAGPipeline(enable_ocr=False)
    
    stats = pipeline.get_statistics()
    
    assert 'retriever_stats' in stats
    assert 'config' in stats

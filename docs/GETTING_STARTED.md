# Getting Started with MMSSR

A step-by-step guide to get you up and running with the Multi-Modal RAG framework.

## Quick Start (5 minutes)

### 1. Install MMSSR

```bash
git clone https://github.com/MRMORNINGSTAR2233/MMSSR.git
cd MMSSR
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up API Key

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Prepare Sample Data

```bash
python scripts/setup_sample_data.py
```

Add some documents to `data/sample/` (PDFs, images, etc.)

### 4. Run Your First Query

```python
from mmssr import MultiModalRAGPipeline

# Create pipeline
pipeline = MultiModalRAGPipeline()

# Index documents
pipeline.index_directory("data/sample")

# Ask a question
result = pipeline.query("What is this document about?")
print(result['answer'])
```

## Understanding the Basics

### What is Multi-Modal RAG?

Traditional RAG (Retrieval-Augmented Generation) only works with text. MMSSR extends this to handle:

- **Text**: Regular document content
- **Images**: Diagrams, photos, charts
- **Tables**: Structured data
- **Mixed Content**: Documents containing all of the above

### How It Works

1. **Parse**: Extract text, images, and tables from documents
2. **Embed**: Convert each element to vector embeddings
3. **Store**: Save embeddings in ChromaDB vector database
4. **Retrieve**: Find relevant content for user queries
5. **Generate**: Use GPT-4V to create answers from multi-modal context

## Common Use Cases

### Use Case 1: Technical Documentation

```python
from mmssr import MultiModalRAGPipeline

pipeline = MultiModalRAGPipeline(enable_ocr=True)

# Index technical manual
pipeline.index_document("user_manual.pdf")

# Ask about diagrams
result = pipeline.query("How do I assemble part A according to the diagram?")
print(result['answer'])

# Ask about specifications
result = pipeline.query("What are the technical specifications?")
print(result['answer'])
```

### Use Case 2: Research Papers

```python
# Index research papers
pipeline.index_directory("papers/")

# Find information from figures
result = pipeline.query("What does the architecture diagram show?")

# Analyze tables
result = pipeline.query("Compare the results in Table 3")
```

### Use Case 3: Product Catalogs

```python
from mmssr.parsers import ModalityType

# Index product catalog
pipeline.index_document("catalog.pdf")

# Search by image
result = pipeline.query(
    "Find products similar to this design",
    modality=ModalityType.IMAGE
)

# Search specifications
result = pipeline.query(
    "What are the dimensions?",
    modality=ModalityType.TABLE
)
```

## Step-by-Step Tutorial

### Step 1: Document Preparation

Supported formats:
- PDF (`.pdf`)
- Images (`.png`, `.jpg`, `.jpeg`)

Best practices:
- High-quality scans for OCR
- Clear images and diagrams
- Well-formatted tables

### Step 2: Creating a Pipeline

```python
from mmssr import MultiModalRAGPipeline

# Basic pipeline
pipeline = MultiModalRAGPipeline()

# With custom settings
pipeline = MultiModalRAGPipeline(
    enable_ocr=True  # For scanned documents
)
```

### Step 3: Indexing Documents

```python
# Single document
n_elements = pipeline.index_document("document.pdf")
print(f"Indexed {n_elements} elements")

# Multiple documents
stats = pipeline.index_directory("my_documents/")
print(f"Total: {stats['total_elements']}")
print(f"Text: {stats['modality_counts']['text']}")
print(f"Images: {stats['modality_counts']['image']}")
print(f"Tables: {stats['modality_counts']['table']}")
```

### Step 4: Querying

```python
# Basic query
result = pipeline.query("What is the main topic?")
print(result['answer'])

# With more context
result = pipeline.query("Explain in detail", n_results=10)

# Modality-specific
from mmssr.parsers import ModalityType
result = pipeline.query(
    "Describe the images",
    modality=ModalityType.IMAGE
)
```

### Step 5: Analyzing Results

```python
# Get detailed context
result = pipeline.query_with_context("What are the findings?")

print(f"Answer: {result['answer']}\n")

print("Context used:")
for ctx in result['context']:
    print(f"- {ctx['type']} from {ctx['source']} "
          f"(page {ctx['page']}, score: {ctx['score']:.2f})")
```

## Advanced Features

### Custom Models

```python
from mmssr.embedders import MultiModalEmbedder, TextEmbedder

# Use specific embedding model
text_embedder = TextEmbedder(model_name="all-mpnet-base-v2")
embedder = MultiModalEmbedder(text_embedder=text_embedder)

pipeline = MultiModalRAGPipeline(embedder=embedder)
```

### Image Captioning

```python
# Caption all images in a document
captions = pipeline.caption_images_in_document("report.pdf")

for cap in captions:
    print(f"Page {cap['page']}:")
    print(f"  {cap['caption']}\n")
```

### Document Summarization

```python
summary = pipeline.summarize_document("long_report.pdf")
print(summary)
```

## Tips and Best Practices

### Performance Optimization

1. **Use GPU**: Install PyTorch with CUDA for faster embeddings
2. **Batch Processing**: Index multiple documents at once
3. **Adjust Context**: Use `n_results` to control context size
4. **Clear Cache**: Periodically clear index to free memory

```python
# Clear index
pipeline.clear_index()
```

### Quality Improvements

1. **Enable OCR**: For scanned documents
2. **High-Quality Sources**: Better source = better results
3. **Specific Queries**: More specific questions get better answers
4. **Modality Filters**: Use when you know the content type

### Cost Management

OpenAI API costs can add up. To reduce costs:

1. Use local embeddings (default):
```python
pipeline = MultiModalRAGPipeline()  # Uses sentence-transformers
```

2. Limit context size:
```python
result = pipeline.query(query, n_results=3)  # Fewer tokens
```

3. Cache results for repeated queries

## Troubleshooting

### Common Issues

**Issue**: "OPENAI_API_KEY not set"
```python
# Solution: Set in .env file or environment
import os
os.environ['OPENAI_API_KEY'] = 'your-key'
```

**Issue**: Out of memory
```python
# Solution: Use lighter models or reduce batch size
from mmssr.config import config
config.processing.chunk_size = 500  # Smaller chunks
```

**Issue**: Slow processing
```python
# Solution: Disable OCR if not needed
pipeline = MultiModalRAGPipeline(enable_ocr=False)
```

**Issue**: Poor results
- Try increasing n_results
- Use modality-specific search
- Improve source document quality

## Next Steps

1. **Explore Examples**: Check `examples/` directory
2. **Read API Docs**: See `docs/API.md`
3. **Customize**: Modify configuration in `src/mmssr/config.py`
4. **Contribute**: Submit issues or PRs on GitHub

## Interactive Tutorial

Run the interactive example:

```bash
python examples/custom_pipeline.py
```

This will start an interactive Q&A session where you can:
- Ask questions about your documents
- See source attribution
- Experiment with different queries

## Learning Resources

- [LangChain Multi-Modal RAG](https://blog.langchain.dev/semi-structured-multi-modal-rag/)
- [NVIDIA RAG Guide](https://developer.nvidia.com/blog/build-enterprise-retrieval-augmented-generation-apps-with-nvidia-retrieval-qa-embedding-model/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Unstructured.io Docs](https://unstructured.io/docs)

## Support

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**Ready to build?** Start with the basic example:

```bash
python examples/basic_usage.py
```

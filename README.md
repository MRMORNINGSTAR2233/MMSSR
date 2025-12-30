# MMSSR - Multi-Modal and Semi-Structured RAG

A comprehensive framework for Retrieval-Augmented Generation (RAG) that handles multi-modal content including text, images, tables, and charts from complex documents like PDFs, presentations, and technical manuals.

## 🎯 Overview

Standard RAG systems only work with plain text, missing valuable context from images, diagrams, tables, and semi-structured data. MMSSR solves this by:

- **Multi-Modal Parsing**: Extract text, images, tables, and charts from PDFs and other documents
- **Specialized Embeddings**: Use appropriate models for each modality (CLIP for images, sentence-transformers for text)
- **Multi-Vector Retrieval**: Search across all modalities simultaneously with ChromaDB
- **Vision-Capable Generation**: Leverage GPT-4V or similar models to understand and reason about images
- **Unified Pipeline**: Simple API that handles the entire workflow from document ingestion to answer generation

## 🚀 Key Features

- ✅ **Multi-modal document parsing** with Unstructured.io
- ✅ **Image understanding** with CLIP embeddings and GPT-4V
- ✅ **Table extraction and analysis** from PDFs
- ✅ **Multiple LLM providers**: OpenAI (GPT-4), Google Gemini, Groq AI
- ✅ **Flexible embeddings**: OpenAI, Gemini, or local models
- ✅ **Separate vector stores** for each modality with unified retrieval
- ✅ **Automatic image captioning** for enhanced searchability
- ✅ **Modality-specific search** (search only in images, tables, or text)
- ✅ **Source attribution** with page numbers and document references
- ✅ **Flexible configuration** for custom models and settings

## 📋 Prerequisites

- Python 3.9+
- API keys (at least one):
  - **OpenAI API key** (for GPT-4 and GPT-4V)
  - **Google API key** (for Gemini 2.5)
  - **Groq API key** (for Llama 3.3 and fast inference)
- Optional: CUDA-capable GPU for faster local embeddings

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/MRMORNINGSTAR2233/MMSSR.git
cd MMSSR
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# At least one provider required
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Choose provider (openai, gemini, or groq)
LLM_PROVIDER=openai
```

### 5. Install the package (optional)

```bash
pip install -e .
```

## 📚 Quick Start

### Basic Usage

```python
from mmssr import MultiModalRAGPipeline

# Create pipeline
pipeline = MultiModalRAGPipeline(enable_ocr=True)

# Index a document
pipeline.index_document("path/to/document.pdf")

# Query the system
result = pipeline.query("What are the main findings?")
print(result['answer'])
```

### Index Multiple Documents

```python
# Index entire directory
stats = pipeline.index_directory("path/to/documents/")
print(f"Indexed {stats['total_elements']} elements")
print(f"Breakdown: {stats['modality_counts']}")
```

### Query with Context

```python
# Get answer with detailed context
result = pipeline.query_with_context(
    query="Explain the architecture diagram",
    n_results=5
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

## 🎨 Advanced Usage

### Using Different LLM Providers

```python
from mmssr.generator import MultiModalGenerator

# OpenAI GPT-4
generator_openai = MultiModalGenerator(
    provider="openai",
    model="gpt-4-turbo-preview",
    vision_model="gpt-4-vision-preview"
)

# Google Gemini 2.5
generator_gemini = MultiModalGenerator(
    provider="gemini",
    model="gemini-2.5-pro",
    vision_model="gemini-2.5-flash"
)

# Groq AI (Fast inference with Llama)
generator_groq = MultiModalGenerator(
    provider="groq",
    model="llama-3.3-70b-versatile",
    vision_model="llama-3.2-90b-vision-preview"
)

pipeline = MultiModalRAGPipeline(generator=generator_gemini)
```

### Using Gemini Embeddings

```python
from mmssr.embedders import GeminiTextEmbedder, MultiModalEmbedder

# Use Gemini for embeddings
embedder = MultiModalEmbedder(
    text_embedder=GeminiTextEmbedder(model_name="text-embedding-004")
)

pipeline = MultiModalRAGPipeline(embedder=embedder)
```

### Modality-Specific Search

```python
from mmssr.parsers import ModalityType

# Search only in images
result = pipeline.query(
    "Show me the architecture diagram",
    modality=ModalityType.IMAGE,
    n_results=3
)

# Search only in tables
result = pipeline.query(
    "What are the performance metrics?",
    modality=ModalityType.TABLE,
    n_results=5
)
```

### Image Captioning

```python
# Generate captions for all images in a document
captions = pipeline.caption_images_in_document("document.pdf")

for caption_info in captions:
    print(f"Page {caption_info['page']}: {caption_info['caption']}")
```

### Document Summarization

```python
# Generate summary
summary = pipeline.summarize_document("document.pdf")
print(summary)
```

### Custom Configuration

```python
from mmssr.embedders import MultiModalEmbedder, OpenAITextEmbedder
from mmssr.generator import MultiModalGenerator

# Use OpenAI embeddings for better quality
embedder = MultiModalEmbedder(use_openai_text=True)

# Configure generator
generator = MultiModalGenerator(
    model="gpt-4-turbo-preview",
    temperature=0.3,
    max_tokens=4096
)

# Create custom pipeline
pipeline = MultiModalRAGPipeline(
    embedder=embedder,
    generator=generator,
    enable_ocr=True
)
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MultiModalRAGPipeline                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│   Parser     │      │   Embedder   │     │  Generator   │
│              │      │              │     │              │
│ - PDF        │      │ - Text (ST)  │     │ - GPT-4V     │
│ - Images     │──────│ - Images     │─────│ - Vision     │
│ - Tables     │      │   (CLIP)     │     │ - Context    │
└──────────────┘      │ - Tables     │     └──────────────┘
                      └──────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │  Retriever   │
                      │              │
                      │ - ChromaDB   │
                      │ - Multi-vec  │
                      └──────────────┘
```

### Components

1. **Parser** ([parsers.py](src/mmssr/parsers.py))
   - Extracts text, images, and tables from documents
   - Supports PDF, images, and more
   - Uses Unstructured.io and pdfplumber

2. **Embedder** ([embedders.py](src/mmssr/embedders.py))
   - Text: sentence-transformers or OpenAI
   - Images: CLIP (openai/clip-vit-base-patch32)
   - Tables: Text conversion + text embeddings

3. **Retriever** ([retriever.py](src/mmssr/retriever.py))
   - Multi-vector storage in ChromaDB
   - Separate collections per modality
   - Unified search across all modalities

4. **Generator** ([generator.py](src/mmssr/generator.py))
   - GPT-4V for multi-modal generation
   - Handles text, tables, and images
   - Source attribution

## 📖 Examples

See the `examples/` directory for detailed examples:

- **[basic_usage.py](examples/basic_usage.py)**: Simple indexing and querying
- **[advanced_usage.py](examples/advanced_usage.py)**: Modality-specific search, image captioning
- **[custom_pipeline.py](examples/custom_pipeline.py)**: Custom configurations and interactive Q&A

Run an example:

```bash
python examples/basic_usage.py
```

## 🔬 Research Background

This framework is based on recent advances in multi-modal RAG:

1. **Multi-Vector Retrieval**: Decompose documents into modality-specific elements and embed each separately ([LangChain Multi-Vector Retriever](https://blog.langchain.dev/semi-structured-multi-modal-rag/))

2. **Vision-Language Models**: Use CLIP for unified image-text embeddings ([NVIDIA Multi-Modal RAG](https://developer.nvidia.com/blog/build-enterprise-retrieval-augmented-generation-apps-with-nvidia-retrieval-qa-embedding-model/))

3. **Semi-Structured Data**: Handle tables and layouts with specialized parsing ([Unstructured.io](https://unstructured.io))

4. **Vision-Capable LLMs**: GPT-4V, Gemini 2.5 Flash, Llama 3.2 Vision for understanding images in context

## 🤖 Supported Models

### LLM Providers

| Provider | Text Models | Vision Models | Speed | Cost |
|----------|-------------|---------------|-------|------|
| **OpenAI** | gpt-4-turbo-preview, gpt-4 | gpt-4-vision-preview | Medium | $$$ |
| **Google Gemini** | gemini-2.5-pro, gemini-3-flash | gemini-2.5-flash, gemini-3-flash | Fast | $$ |
| **Groq** | llama-3.3-70b-versatile | llama-3.2-90b-vision-preview | Very Fast | $ |

### Embedding Models

| Provider | Model | Dimension | Best For |
|----------|-------|-----------|----------|
| **Sentence Transformers** | all-MiniLM-L6-v2 | 384 | Fast, local |
| **OpenAI** | text-embedding-3-large | 3072 | High quality |
| **Google Gemini** | text-embedding-004 | 768 | Multimodal |
| **CLIP** | clip-vit-base-patch32 | 512 | Images |

### Choosing a Provider

- **OpenAI**: Best quality, highest cost, good for production
- **Gemini**: Great balance of speed/quality, multimodal native
- **Groq**: Fastest inference, cost-effective, good for high-volume

## 🛠️ Development

### Project Structure

```
MMSSR/
├── src/mmssr/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── parsers.py         # Document parsing
│   ├── embedders.py       # Multi-modal embeddings
│   ├── retriever.py       # Vector store and retrieval
│   ├── generator.py       # LLM generation
│   └── pipeline.py        # Main pipeline
├── examples/              # Usage examples
├── data/                  # Data directory (gitignored)
├── tests/                 # Unit tests
├── requirements.txt       # Dependencies
└── README.md
```

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black src/
flake8 src/
```

## 📊 Supported Document Types

| Type | Extensions | Features |
|------|-----------|----------|
| PDF | `.pdf` | Text, images, tables, OCR |
| Images | `.png`, `.jpg`, `.jpeg` | CLIP embeddings, captioning |
| More coming | `.docx`, `.pptx` | Planned |

## 🎯 Use Cases

- **Technical Documentation**: Search across manuals with diagrams and tables
- **Research Papers**: Find information in figures, equations, and tables
- **Product Catalogs**: Search by product images and specifications
- **Medical Records**: Analyze documents with charts, scans, and tables
- **Financial Reports**: Extract insights from tables, charts, and text

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for multi-vector retriever concepts
- [Unstructured.io](https://unstructured.io) for document parsing
- [OpenAI](https://openai.com/) for CLIP and GPT-4V
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for text embeddings

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🗺️ Roadmap

- [ ] Support for more document formats (.docx, .pptx)
- [ ] Local vision models (LLaVA, BLIP-2)
- [ ] Chart and diagram understanding
- [ ] Multi-language support
- [ ] Streaming responses
- [ ] Web UI interface
- [ ] Evaluation benchmarks (MMQA, VQA datasets)
- [ ] Fine-tuning support for domain-specific embeddings

---

**Built with ❤️ for better multi-modal understanding**
Multi-Modal Semi-Structured RAG

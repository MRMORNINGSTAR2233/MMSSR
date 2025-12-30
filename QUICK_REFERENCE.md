# MMSSR Quick Reference

One-page reference for common operations.

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY to .env
```

## Basic Usage

```python
from mmssr import MultiModalRAGPipeline

# Create pipeline
pipeline = MultiModalRAGPipeline()

# Index documents
pipeline.index_document("doc.pdf")
pipeline.index_directory("docs/")

# Query
result = pipeline.query("What is this about?")
print(result['answer'])
```

## Common Patterns

### Modality-Specific Search

```python
from mmssr.parsers import ModalityType

# Search only images
pipeline.query("diagram", modality=ModalityType.IMAGE)

# Search only tables
pipeline.query("metrics", modality=ModalityType.TABLE)

# Search only text
pipeline.query("overview", modality=ModalityType.TEXT)
```

### Get Detailed Results

```python
result = pipeline.query_with_context("question", n_results=5)

for ctx in result['context']:
    print(f"{ctx['type']}: {ctx['source']} (score: {ctx['score']})")
```

### Custom Configuration

```python
from mmssr.embedders import MultiModalEmbedder
from mmssr.generator import MultiModalGenerator

embedder = MultiModalEmbedder(use_openai_text=True)
generator = MultiModalGenerator(temperature=0.3)

pipeline = MultiModalRAGPipeline(
    embedder=embedder,
    generator=generator
)
```

### Image Captioning

```python
captions = pipeline.caption_images_in_document("doc.pdf")
for cap in captions:
    print(f"Page {cap['page']}: {cap['caption']}")
```

### Document Summarization

```python
summary = pipeline.summarize_document("doc.pdf")
print(summary)
```

## Configuration (.env)

```env
# Choose your LLM provider
LLM_PROVIDER=openai  # Options: openai, gemini, groq

# API Keys (at least one required)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
```

## Provider Selection

```python
from mmssr.generator import MultiModalGenerator

# OpenAI GPT-4
gen = MultiModalGenerator(provider="openai")

# Google Gemini (fast + cost-effective)
gen = MultiModalGenerator(provider="gemini")

# Groq (fastest inference)
gen = MultiModalGenerator(provider="groq")

pipeline = MultiModalRAGPipeline(generator=gen)
```

## File Structure

```
data/sample/         # Add documents here
data/chroma_db/      # Vector store (auto-created)
examples/            # Example scripts
docs/                # Documentation
src/mmssr/           # Source code
```

## Key Classes

- `MultiModalRAGPipeline`: Main interface
- `MultiModalParser`: Document parsing
- `MultiModalEmbedder`: Embeddings
- `MultiModalRetriever`: Vector search
- `MultiModalGenerator`: Answer generation

## Modality Types

```python
ModalityType.TEXT    # Text content
ModalityType.IMAGE   # Images/diagrams
ModalityType.TABLE   # Tables/structured data
ModalityType.CHART   # Charts (future)
```

## Common Commands

```bash
# Run examples
python examples/basic_usage.py
python examples/advanced_usage.py
python examples/custom_pipeline.py

# Setup data directories
python scripts/setup_sample_data.py

# Run tests
pytest tests/

# Format code
black src/
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API key error | Set `OPENAI_API_KEY` in `.env` |
| Out of memory | Reduce `n_results` or use smaller models |
| Slow processing | Disable OCR: `enable_ocr=False` |
| Import errors | `pip install -r requirements.txt` |

## Performance Tips

1. Enable GPU for faster embeddings
2. Use `index_directory()` for batch processing
3. Cache results for repeated queries
4. Adjust `n_results` based on needs
5. Use modality filters when possible

## API Quick Links

- Full API: `docs/API.md`
- Tutorial: `docs/GETTING_STARTED.md`
- Install: `docs/INSTALLATION.md`

---

**Need Help?** See README.md or open a GitHub issue.

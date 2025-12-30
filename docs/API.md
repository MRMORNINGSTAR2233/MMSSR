# MMSSR API Reference

Complete API documentation for the Multi-Modal and Semi-Structured RAG framework.

## Table of Contents

- [Pipeline](#pipeline)
- [Parsers](#parsers)
- [Embedders](#embedders)
- [Retriever](#retriever)
- [Generator](#generator)
- [Configuration](#configuration)

---

## Pipeline

### `MultiModalRAGPipeline`

Main pipeline class that orchestrates all components.

#### Constructor

```python
MultiModalRAGPipeline(
    parser: Optional[MultiModalParser] = None,
    embedder: Optional[MultiModalEmbedder] = None,
    retriever: Optional[MultiModalRetriever] = None,
    generator: Optional[MultiModalGenerator] = None,
    enable_ocr: bool = True
)
```

**Parameters:**
- `parser`: Custom document parser
- `embedder`: Custom multi-modal embedder
- `retriever`: Custom retriever
- `generator`: Custom generator
- `enable_ocr`: Enable OCR for scanned documents

#### Methods

##### `index_document(file_path: str) -> int`

Index a single document.

**Returns:** Number of elements indexed

```python
pipeline = MultiModalRAGPipeline()
n = pipeline.index_document("document.pdf")
print(f"Indexed {n} elements")
```

##### `index_directory(directory: str) -> Dict[str, int]`

Index all documents in a directory.

**Returns:** Dictionary with statistics

```python
stats = pipeline.index_directory("data/docs")
print(stats['total_elements'])
print(stats['modality_counts'])
```

##### `query(query: str, n_results: int = 5, modality: Optional[ModalityType] = None) -> Dict`

Query the RAG system.

**Parameters:**
- `query`: User question
- `n_results`: Number of context elements to retrieve
- `modality`: Optional filter by modality type

**Returns:** Dictionary with answer and metadata

```python
result = pipeline.query("What are the specifications?", n_results=10)
print(result['answer'])
print(result['sources'])
```

##### `query_with_context(query: str, n_results: int = 5) -> Dict`

Query and return detailed context information.

```python
result = pipeline.query_with_context("Explain the diagram")
for ctx in result['context']:
    print(f"{ctx['type']}: {ctx['source']} (score: {ctx['score']})")
```

##### `summarize_document(file_path: str) -> str`

Generate a summary of a document.

```python
summary = pipeline.summarize_document("report.pdf")
```

##### `caption_images_in_document(file_path: str) -> List[Dict]`

Generate captions for all images.

```python
captions = pipeline.caption_images_in_document("manual.pdf")
for cap in captions:
    print(f"Page {cap['page']}: {cap['caption']}")
```

---

## Parsers

### `MultiModalParser`

Parse multi-modal documents.

```python
from mmssr.parsers import MultiModalParser

parser = MultiModalParser(enable_ocr=True)
elements = parser.parse("document.pdf")
```

#### Methods

##### `parse(file_path: str) -> List[DocumentElement]`

Parse a single document.

##### `parse_directory(directory: str) -> List[DocumentElement]`

Parse all supported documents in a directory.

### `DocumentElement`

Represents an extracted element.

**Attributes:**
- `id`: Unique identifier
- `type`: ModalityType (TEXT, IMAGE, TABLE, CHART)
- `content`: Element content
- `metadata`: Additional metadata
- `source`: Source file path
- `page_number`: Page number (if applicable)

---

## Embedders

### `MultiModalEmbedder`

Unified embedder for all modalities.

```python
from mmssr.embedders import MultiModalEmbedder

embedder = MultiModalEmbedder(use_openai_text=False)
```

#### Constructor Parameters

- `text_embedder`: Custom text embedder
- `image_embedder`: Custom image embedder
- `table_embedder`: Custom table embedder
- `use_openai_text`: Use OpenAI embeddings for text

#### Methods

##### `embed_element(element: DocumentElement) -> np.ndarray`

Embed a single document element.

##### `embed_elements(elements: List[DocumentElement]) -> np.ndarray`

Embed multiple elements.

##### `embed_query(query: str, target_modality: Optional[ModalityType] = None) -> np.ndarray`

Embed a query for retrieval.

### `TextEmbedder`

Text-only embedder using sentence-transformers.

```python
from mmssr.embedders import TextEmbedder

embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
embedding = embedder.embed("Sample text")
```

### `ImageEmbedder`

Image embedder using CLIP.

```python
from mmssr.embedders import ImageEmbedder
from PIL import Image

embedder = ImageEmbedder(model_name="openai/clip-vit-base-patch32")
img = Image.open("image.png")
embedding = embedder.embed(img)
```

### `OpenAITextEmbedder`

Text embedder using OpenAI API.

```python
from mmssr.embedders import OpenAITextEmbedder

embedder = OpenAITextEmbedder(model_name="text-embedding-3-large")
embedding = embedder.embed("Sample text")
```

---

## Retriever

### `MultiModalRetriever`

Multi-modal retrieval system.

```python
from mmssr.retriever import MultiModalRetriever

retriever = MultiModalRetriever()
```

#### Methods

##### `index_documents(elements: List[DocumentElement]) -> None`

Index document elements.

```python
retriever.index_documents(elements)
```

##### `retrieve(query: str, n_results: int = 10, modality: Optional[ModalityType] = None) -> List[DocumentElement]`

Retrieve relevant elements.

```python
results = retriever.retrieve("machine learning", n_results=5)
```

##### `retrieve_with_scores(query: str, n_results: int = 10) -> List[Tuple[DocumentElement, float]]`

Retrieve elements with relevance scores.

```python
results = retriever.retrieve_with_scores("architecture diagram")
for elem, score in results:
    print(f"{elem.id}: {score}")
```

### `MultiModalVectorStore`

Vector database for multi-modal embeddings.

```python
from mmssr.retriever import MultiModalVectorStore

store = MultiModalVectorStore(
    persist_directory="./data/chroma_db",
    collection_name="my_collection"
)
```

---

## Generator

### `MultiModalGenerator`

Generate answers using vision-capable LLMs.

```python
from mmssr.generator import MultiModalGenerator

generator = MultiModalGenerator(
    model="gpt-4-turbo-preview",
    vision_model="gpt-4-vision-preview",
    temperature=0.7,
    max_tokens=2048
)
```

#### Methods

##### `generate(query: str, context_elements: List[DocumentElement]) -> Dict`

Generate answer from query and context.

```python
result = generator.generate(
    query="What is shown in the diagram?",
    context_elements=retrieved_elements
)
print(result['answer'])
```

##### `generate_summary(elements: List[DocumentElement]) -> str`

Generate a summary of elements.

##### `caption_image(image_element: DocumentElement) -> str`

Generate a caption for an image.

---

## Configuration

### `Config`

Global configuration class.

```python
from mmssr.config import config

# Access configuration
print(config.llm.text_model)
print(config.embedding.text_model)
print(config.vector_store.persist_directory)

# Validate API keys
config.validate_api_keys()
```

### Configuration Options

#### Embedding Config

```python
config.embedding.text_model = "all-MiniLM-L6-v2"
config.embedding.image_model = "openai/clip-vit-base-patch32"
config.embedding.dimension = 384
```

#### LLM Config

```python
config.llm.text_model = "gpt-4-turbo-preview"
config.llm.vision_model = "gpt-4-vision-preview"
config.llm.temperature = 0.7
config.llm.max_tokens = 2048
```

#### Vector Store Config

```python
config.vector_store.persist_directory = "./data/chroma_db"
config.vector_store.collection_name = "multimodal_rag"
```

#### Processing Config

```python
config.processing.chunk_size = 1000
config.processing.chunk_overlap = 200
config.processing.max_image_size = 1024
config.processing.enable_ocr = True
```

---

## Utility Functions

### `create_pipeline`

Quick setup function.

```python
from mmssr.pipeline import create_pipeline

pipeline = create_pipeline(
    enable_ocr=True,
    use_openai_embeddings=False
)
```

---

## Enums

### `ModalityType`

```python
from mmssr.parsers import ModalityType

ModalityType.TEXT
ModalityType.IMAGE
ModalityType.TABLE
ModalityType.CHART
```

---

## Error Handling

All methods can raise exceptions. Wrap in try-except blocks:

```python
try:
    result = pipeline.query("What is this?")
except FileNotFoundError:
    print("Document not found")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except Exception as e:
    print(f"Error: {e}")
```

---

## Best Practices

1. **API Keys**: Always set `OPENAI_API_KEY` in environment
2. **OCR**: Enable OCR only when needed (slower processing)
3. **Batch Processing**: Use `index_directory` for multiple files
4. **Memory**: Clear cache periodically with `pipeline.clear_index()`
5. **Error Handling**: Always handle exceptions in production code
6. **Modality Filters**: Use modality filters for specific searches
7. **Context Size**: Adjust `n_results` based on query complexity

---

For more examples, see the `examples/` directory.

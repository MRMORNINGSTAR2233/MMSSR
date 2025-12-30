# LLM Provider Guide

Comprehensive guide to using different LLM providers with MMSSR.

## Overview

MMSSR supports three major LLM providers:

1. **OpenAI** - GPT-4 and GPT-4 Vision
2. **Google Gemini** - Gemini 2.5 Pro/Flash
3. **Groq** - Llama 3.3 and Llama 3.2 Vision

## Quick Comparison

| Feature | OpenAI | Gemini | Groq |
|---------|--------|--------|------|
| **Quality** | Excellent | Excellent | Very Good |
| **Speed** | Medium | Fast | Very Fast |
| **Cost** | High | Medium | Low |
| **Vision** | GPT-4V | Native | Llama Vision |
| **Context** | 128K | 2M tokens | 128K |
| **Best For** | Production | Balance | High-volume |

## Setup

### 1. Get API Keys

**OpenAI**:
- Visit [platform.openai.com](https://platform.openai.com/api-keys)
- Create API key
- Add to `.env`: `OPENAI_API_KEY=sk-...`

**Google Gemini**:
- Visit [aistudio.google.com](https://aistudio.google.com/apikey)
- Create API key
- Add to `.env`: `GOOGLE_API_KEY=...`

**Groq**:
- Visit [console.groq.com](https://console.groq.com/keys)
- Create API key
- Add to `.env`: `GROQ_API_KEY=...`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `openai>=1.10.0`
- `google-generativeai>=0.8.0`
- `groq>=0.4.0`

## Usage Examples

### OpenAI (GPT-4)

```python
from mmssr import MultiModalRAGPipeline
from mmssr.generator import MultiModalGenerator

# Create OpenAI generator
generator = MultiModalGenerator(
    provider="openai",
    model="gpt-4-turbo-preview",
    vision_model="gpt-4-vision-preview",
    temperature=0.7
)

pipeline = MultiModalRAGPipeline(generator=generator)

# Use as normal
pipeline.index_document("document.pdf")
result = pipeline.query("What is this about?")
```

**Available Models**:
- Text: `gpt-4-turbo-preview`, `gpt-4`, `gpt-3.5-turbo`
- Vision: `gpt-4-vision-preview`

**Pros**:
- Highest quality outputs
- Best vision understanding
- Reliable and stable
- Great for complex reasoning

**Cons**:
- Most expensive
- Slower than alternatives
- Rate limits on free tier

### Google Gemini

```python
from mmssr.generator import MultiModalGenerator
from mmssr.embedders import MultiModalEmbedder, GeminiTextEmbedder

# Create Gemini generator
generator = MultiModalGenerator(
    provider="gemini",
    model="gemini-2.5-pro",
    vision_model="gemini-2.5-flash",
    temperature=0.7
)

# Optional: Use Gemini embeddings
embedder = MultiModalEmbedder(
    text_embedder=GeminiTextEmbedder(model_name="text-embedding-004")
)

pipeline = MultiModalRAGPipeline(
    generator=generator,
    embedder=embedder
)
```

**Available Models**:

Text Models:
- `gemini-2.5-pro` - Best reasoning (December 2025)
- `gemini-3-pro` - Most intelligent (Latest)
- `gemini-3-flash` - Fast, efficient

Vision Models:
- `gemini-2.5-flash` - Balanced multimodal
- `gemini-3-flash` - Latest multimodal

Embeddings:
- `text-embedding-004` - 768 dimensions

**Pros**:
- Native multimodal (text + images together)
- Very long context (2M tokens)
- Cost-effective
- Fast inference
- Strong reasoning

**Cons**:
- Newer, less battle-tested
- Rate limits vary by region

**Special Features**:
- Native tool use
- Google Search integration
- Code execution
- Thinking mode for complex reasoning

### Groq AI

```python
from mmssr.generator import MultiModalGenerator

# Create Groq generator (fastest inference)
generator = MultiModalGenerator(
    provider="groq",
    model="llama-3.3-70b-versatile",
    vision_model="llama-3.2-90b-vision-preview",
    temperature=0.7
)

pipeline = MultiModalRAGPipeline(generator=generator)
```

**Available Models**:

Text Models:
- `llama-3.3-70b-versatile` - Latest Llama (December 2025)
- `llama-3.1-70b-versatile` - Powerful general-purpose
- `mixtral-8x7b-32768` - Good for long context

Vision Models:
- `llama-3.2-90b-vision-preview` - Multimodal Llama
- `llama-3.2-11b-vision-preview` - Smaller, faster

**Pros**:
- **Extremely fast** inference (10-100x faster)
- Very cost-effective
- OpenAI-compatible API
- Great for high-volume applications
- Built on LPU (Language Processing Unit)

**Cons**:
- Vision support limited to Llama 3.2
- Smaller context windows
- Quality slightly below GPT-4

**Best Use Cases**:
- Real-time applications
- High request volume
- Cost-sensitive projects
- Chatbots and interactive apps

## Setting Provider via Environment

Set in `.env`:

```env
LLM_PROVIDER=gemini  # or openai, groq
```

Then use default configuration:

```python
from mmssr import MultiModalRAGPipeline

# Automatically uses provider from .env
pipeline = MultiModalRAGPipeline()
```

## Model Selection Guide

### For Production Apps
- **Primary**: OpenAI GPT-4 (reliability)
- **Backup**: Gemini 2.5 Pro (cost/performance)
- **High-volume**: Groq Llama 3.3

### For Development/Testing
- **Best balance**: Gemini 2.5 Flash
- **Fastest**: Groq Llama 3.3
- **Most accurate**: OpenAI GPT-4

### For Specific Tasks

**Complex Reasoning**:
1. Gemini 2.5 Pro (with thinking mode)
2. OpenAI GPT-4
3. Llama 3.3 70B

**Vision Tasks**:
1. OpenAI GPT-4V
2. Gemini 2.5 Flash
3. Llama 3.2 90B Vision

**Long Documents**:
1. Gemini (2M context)
2. OpenAI GPT-4 (128K)
3. Mixtral (32K via Groq)

**Real-time/Interactive**:
1. Groq (any model)
2. Gemini 3 Flash
3. OpenAI GPT-3.5 Turbo

## Cost Comparison

Approximate costs (as of December 2025):

### Text Generation (per 1M tokens)

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| OpenAI | GPT-4 Turbo | $10 | $30 |
| OpenAI | GPT-4V | $10 | $30 |
| Gemini | 2.5 Pro | $1.25 | $5 |
| Gemini | 2.5 Flash | $0.075 | $0.30 |
| Groq | Llama 3.3 70B | $0.59 | $0.79 |

### Embeddings (per 1M tokens)

| Provider | Model | Cost |
|----------|-------|------|
| OpenAI | text-embedding-3-large | $0.13 |
| Gemini | text-embedding-004 | $0.025 |
| Local | Sentence Transformers | Free |

## Performance Benchmarks

Response time for typical RAG query (5 context elements):

| Provider | Model | Avg Time | Tokens/sec |
|----------|-------|----------|------------|
| Groq | Llama 3.3 | 0.5s | 800-1000 |
| Gemini | 2.5 Flash | 1.5s | 200-300 |
| OpenAI | GPT-4 | 3-5s | 50-100 |

*Note: Actual performance varies by query complexity and network latency*

## Switching Providers

You can switch providers at runtime:

```python
from mmssr import MultiModalRAGPipeline
from mmssr.generator import MultiModalGenerator

# Start with Gemini
pipeline = MultiModalRAGPipeline()
pipeline.index_directory("docs/")

# Query with Gemini
result1 = pipeline.query("Question 1")

# Switch to Groq for speed
pipeline.generator = MultiModalGenerator(provider="groq")
result2 = pipeline.query("Question 2")

# Switch to OpenAI for quality
pipeline.generator = MultiModalGenerator(provider="openai")
result3 = pipeline.query("Question 3")
```

## Error Handling

```python
from mmssr.generator import MultiModalGenerator

try:
    generator = MultiModalGenerator(provider="gemini")
except ImportError:
    print("Install: pip install google-generativeai")
except ValueError as e:
    print(f"API key not set: {e}")
```

## Best Practices

1. **Start with Gemini** for best balance
2. **Use Groq** for high-volume/real-time
3. **Use OpenAI** when quality is critical
4. **Set fallbacks** for production:

```python
def create_generator_with_fallback():
    """Try providers in order of preference"""
    try:
        return MultiModalGenerator(provider="gemini")
    except Exception:
        try:
            return MultiModalGenerator(provider="openai")
        except Exception:
            return MultiModalGenerator(provider="groq")
```

5. **Monitor costs** with provider dashboards
6. **Cache responses** when possible
7. **Use appropriate models** for task complexity

## Troubleshooting

### "API key not set"
- Check `.env` file has correct key name
- Run `source .env` or restart terminal
- Verify key is valid in provider dashboard

### "Model not found"
- Check model name spelling
- Verify model is available in your region
- Some models require waitlist access

### "Rate limit exceeded"
- Implement exponential backoff
- Switch to different provider temporarily
- Upgrade to paid tier

### Slow responses
- Use Groq for faster inference
- Reduce context size (fewer n_results)
- Enable prompt caching (where available)

## References

- [OpenAI Documentation](https://platform.openai.com/docs)
- [Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [Groq Documentation](https://console.groq.com/docs)

---

**See also**: [API Reference](API.md) | [Getting Started](GETTING_STARTED.md)

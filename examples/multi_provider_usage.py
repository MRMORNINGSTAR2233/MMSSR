"""
Example: Using different LLM providers (OpenAI, Gemini, Groq)
"""
import os
from dotenv import load_dotenv

from mmssr import MultiModalRAGPipeline
from mmssr.generator import MultiModalGenerator
from mmssr.embedders import MultiModalEmbedder, GeminiTextEmbedder

load_dotenv()


def example_openai():
    """Use OpenAI (GPT-4)"""
    print("\n" + "="*60)
    print("OPENAI - GPT-4 TURBO")
    print("="*60)
    
    generator = MultiModalGenerator(
        provider="openai",
        model="gpt-4-turbo-preview",
        vision_model="gpt-4-vision-preview"
    )
    
    pipeline = MultiModalRAGPipeline(generator=generator)
    
    # Index and query
    if os.path.exists("data/sample"):
        pipeline.index_directory("data/sample")
        result = pipeline.query("What is this document about?")
        print(f"Answer: {result['answer'][:200]}...")


def example_gemini():
    """Use Google Gemini"""
    print("\n" + "="*60)
    print("GOOGLE GEMINI 2.5 PRO")
    print("="*60)
    
    generator = MultiModalGenerator(
        provider="gemini",
        model="gemini-2.5-pro",
        vision_model="gemini-2.5-flash"
    )
    
    # Optional: Use Gemini embeddings too
    try:
        embedder = MultiModalEmbedder(
            text_embedder=GeminiTextEmbedder()
        )
    except:
        print("Gemini embeddings not available, using default")
        embedder = None
    
    pipeline = MultiModalRAGPipeline(
        generator=generator,
        embedder=embedder
    )
    
    # Index and query
    if os.path.exists("data/sample"):
        pipeline.index_directory("data/sample")
        result = pipeline.query("Summarize the key points")
        print(f"Answer: {result['answer'][:200]}...")


def example_groq():
    """Use Groq AI (Fast inference)"""
    print("\n" + "="*60)
    print("GROQ AI - LLAMA 3.3 70B")
    print("="*60)
    
    generator = MultiModalGenerator(
        provider="groq",
        model="llama-3.3-70b-versatile",
        vision_model="llama-3.2-90b-vision-preview"
    )
    
    pipeline = MultiModalRAGPipeline(generator=generator)
    
    # Index and query
    if os.path.exists("data/sample"):
        pipeline.index_directory("data/sample")
        result = pipeline.query("What are the main topics?")
        print(f"Answer: {result['answer'][:200]}...")


def compare_providers():
    """Compare responses from different providers"""
    print("\n" + "="*60)
    print("PROVIDER COMPARISON")
    print("="*60)
    
    query = "Explain the main concept in simple terms"
    
    providers = [
        ("openai", "gpt-4-turbo-preview"),
        ("gemini", "gemini-2.5-pro"),
        ("groq", "llama-3.3-70b-versatile"),
    ]
    
    # Index once
    pipeline = MultiModalRAGPipeline()
    if os.path.exists("data/sample"):
        pipeline.index_directory("data/sample")
    else:
        print("No sample data found. Please add documents to data/sample/")
        return
    
    results = {}
    
    for provider, model in providers:
        try:
            print(f"\n--- {provider.upper()} ({model}) ---")
            
            generator = MultiModalGenerator(
                provider=provider,
                model=model
            )
            
            # Use same pipeline but different generator
            pipeline.generator = generator
            
            result = pipeline.query(query, n_results=3)
            results[provider] = result['answer']
            
            print(f"Answer: {result['answer'][:150]}...")
            
        except Exception as e:
            print(f"Error with {provider}: {e}")
    
    # Show full comparison
    print("\n" + "="*60)
    print("FULL COMPARISON")
    print("="*60)
    
    for provider, answer in results.items():
        print(f"\n{provider.upper()}:")
        print(f"{answer}\n")
        print("-" * 60)


def example_provider_from_env():
    """Use provider specified in environment variable"""
    print("\n" + "="*60)
    print("PROVIDER FROM ENVIRONMENT")
    print("="*60)
    
    # Set LLM_PROVIDER in .env file
    # LLM_PROVIDER=gemini
    
    from mmssr.config import config
    
    print(f"Using provider: {config.llm.provider}")
    print(f"Text model: {config.llm.text_model}")
    print(f"Vision model: {config.llm.vision_model}")
    
    # Pipeline will use provider from config
    pipeline = MultiModalRAGPipeline()
    
    if os.path.exists("data/sample"):
        pipeline.index_directory("data/sample")
        result = pipeline.query("What is this about?")
        print(f"\nAnswer: {result['answer'][:200]}...")


def main():
    """Run provider examples"""
    
    print("Multi-Provider RAG Examples")
    print("Make sure you have API keys set in .env:")
    print("  - OPENAI_API_KEY")
    print("  - GOOGLE_API_KEY")
    print("  - GROQ_API_KEY")
    
    # Check which providers are available
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GOOGLE_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    
    print(f"\nAvailable providers:")
    print(f"  OpenAI: {'✓' if has_openai else '✗'}")
    print(f"  Gemini: {'✓' if has_gemini else '✗'}")
    print(f"  Groq: {'✓' if has_groq else '✗'}")
    
    # Run examples
    try:
        if has_openai:
            example_openai()
        
        if has_gemini:
            example_gemini()
        
        if has_groq:
            example_groq()
        
        # Compare all available providers
        if has_openai and has_gemini and has_groq:
            compare_providers()
        
        # Environment-based example
        example_provider_from_env()
    
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Set API keys in .env file")
        print("2. Installed required packages: pip install -r requirements.txt")
        print("3. Added documents to data/sample/")


if __name__ == "__main__":
    main()

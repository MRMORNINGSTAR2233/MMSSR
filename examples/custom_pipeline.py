"""
Example: Building a custom RAG application with specific configurations
"""
import os
from dotenv import load_dotenv

from mmssr.pipeline import create_pipeline
from mmssr.embedders import MultiModalEmbedder, OpenAITextEmbedder, TextEmbedder, ImageEmbedder
from mmssr.retriever import MultiModalRetriever
from mmssr.generator import MultiModalGenerator
from mmssr import MultiModalRAGPipeline

load_dotenv()


def example_custom_embedder():
    """Create a pipeline with custom embedder configuration"""
    
    print("\n" + "="*60)
    print("CUSTOM EMBEDDER CONFIGURATION")
    print("="*60)
    
    # Option 1: Use OpenAI embeddings for better quality
    print("\n1. Using OpenAI embeddings...")
    pipeline_openai = create_pipeline(use_openai_embeddings=True)
    
    # Option 2: Use specific local models
    print("\n2. Using specific local models...")
    text_embedder = TextEmbedder(model_name="all-mpnet-base-v2")
    image_embedder = ImageEmbedder(model_name="openai/clip-vit-large-patch14")
    
    custom_embedder = MultiModalEmbedder(
        text_embedder=text_embedder,
        image_embedder=image_embedder,
    )
    
    pipeline_custom = MultiModalRAGPipeline(embedder=custom_embedder)
    
    return pipeline_openai, pipeline_custom


def example_custom_generator():
    """Create a pipeline with custom generator settings"""
    
    print("\n" + "="*60)
    print("CUSTOM GENERATOR CONFIGURATION")
    print("="*60)
    
    # Create generator with specific settings
    generator = MultiModalGenerator(
        model="gpt-4-turbo-preview",
        vision_model="gpt-4-vision-preview",
        temperature=0.3,  # More deterministic
        max_tokens=4096,  # Longer responses
    )
    
    pipeline = MultiModalRAGPipeline(generator=generator)
    
    return pipeline


def example_domain_specific_rag():
    """Example for domain-specific RAG (e.g., medical documents, technical manuals)"""
    
    print("\n" + "="*60)
    print("DOMAIN-SPECIFIC RAG")
    print("="*60)
    
    # Create pipeline optimized for technical documents
    pipeline = create_pipeline(
        enable_ocr=True,  # Important for scanned technical docs
        use_openai_embeddings=True,  # Better for domain-specific terminology
    )
    
    # Index domain-specific documents
    docs_dir = "data/technical_manuals"
    if os.path.exists(docs_dir):
        print(f"\nIndexing technical manuals from {docs_dir}...")
        stats = pipeline.index_directory(docs_dir)
        print(f"Indexed: {stats}")
    
    # Domain-specific queries
    queries = [
        "What are the safety specifications mentioned?",
        "Explain the assembly diagram",
        "What are the tolerance values in the specifications table?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = pipeline.query(query, n_results=7)  # More context for technical content
        print(f"Answer: {result['answer'][:300]}...")


def example_batch_processing():
    """Example of batch processing multiple documents"""
    
    print("\n" + "="*60)
    print("BATCH PROCESSING")
    print("="*60)
    
    pipeline = create_pipeline()
    
    # Process multiple directories
    directories = [
        "data/reports",
        "data/presentations",
        "data/research_papers",
    ]
    
    total_indexed = 0
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\nProcessing {directory}...")
            stats = pipeline.index_directory(directory)
            total_indexed += stats['total_elements']
            print(f"  Indexed {stats['total_elements']} elements")
    
    print(f"\nTotal indexed across all directories: {total_indexed}")
    
    # Get overall statistics
    print("\nFinal statistics:")
    stats = pipeline.get_statistics()
    print(stats['retriever_stats'])


def example_interactive_session():
    """Interactive Q&A session"""
    
    print("\n" + "="*60)
    print("INTERACTIVE Q&A SESSION")
    print("="*60)
    
    pipeline = create_pipeline()
    
    # Index documents
    data_dir = "data/sample"
    if os.path.exists(data_dir):
        print(f"\nIndexing documents from {data_dir}...")
        pipeline.index_directory(data_dir)
        print("Documents indexed. Ready for questions!")
    else:
        print(f"Creating {data_dir}...")
        os.makedirs(data_dir, exist_ok=True)
        print(f"Please add documents to {data_dir}")
        return
    
    print("\nType your questions (or 'quit' to exit):")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Process query
            result = pipeline.query_with_context(query, n_results=5)
            
            print(f"\nAssistant: {result['answer']}")
            
            # Show sources
            if result.get('sources'):
                print("\nSources:")
                for src in result['sources']:
                    print(f"  • {src['source']} (page {src['page']})")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Run custom configuration examples"""
    
    # Example 1: Custom embedders
    # pipeline_openai, pipeline_custom = example_custom_embedder()
    
    # Example 2: Custom generator
    # pipeline_gen = example_custom_generator()
    
    # Example 3: Domain-specific RAG
    # example_domain_specific_rag()
    
    # Example 4: Batch processing
    # example_batch_processing()
    
    # Example 5: Interactive session
    example_interactive_session()


if __name__ == "__main__":
    main()

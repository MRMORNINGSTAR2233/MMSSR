"""
Advanced example: Multi-modal search and analysis
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from mmssr import MultiModalRAGPipeline
from mmssr.parsers import ModalityType

load_dotenv()


def search_by_modality(pipeline: MultiModalRAGPipeline):
    """Demonstrate modality-specific search"""
    
    print("\n" + "="*60)
    print("MODALITY-SPECIFIC SEARCH")
    print("="*60)
    
    query = "machine learning architecture"
    
    # Search only in text
    print(f"\nQuery: {query}")
    print("\n1. Searching in TEXT only:")
    result = pipeline.query(query, n_results=3, modality=ModalityType.TEXT)
    print(f"Answer: {result['answer'][:200]}...")
    
    # Search only in tables
    print("\n2. Searching in TABLES only:")
    result = pipeline.query(query, n_results=3, modality=ModalityType.TABLE)
    print(f"Answer: {result['answer'][:200]}...")
    
    # Search only in images
    print("\n3. Searching in IMAGES only:")
    result = pipeline.query(query, n_results=3, modality=ModalityType.IMAGE)
    print(f"Answer: {result['answer'][:200]}...")


def analyze_document_structure(pipeline: MultiModalRAGPipeline):
    """Analyze what types of content are in the indexed documents"""
    
    print("\n" + "="*60)
    print("DOCUMENT STRUCTURE ANALYSIS")
    print("="*60)
    
    stats = pipeline.get_statistics()
    retriever_stats = stats['retriever_stats']
    
    print("\nIndexed content by modality:")
    for modality, count in retriever_stats.items():
        if modality != 'cached_elements':
            print(f"  {modality.upper()}: {count} elements")
    
    print(f"\nTotal cached elements: {retriever_stats.get('cached_elements', 0)}")


def extract_and_caption_images(pipeline: MultiModalRAGPipeline, doc_path: str):
    """Extract and caption all images from a document"""
    
    print("\n" + "="*60)
    print("IMAGE EXTRACTION AND CAPTIONING")
    print("="*60)
    
    if not os.path.exists(doc_path):
        print(f"Document not found: {doc_path}")
        return
    
    print(f"\nProcessing: {doc_path}")
    captions = pipeline.caption_images_in_document(doc_path)
    
    if captions:
        print(f"\nFound {len(captions)} images:")
        for i, caption_info in enumerate(captions, 1):
            print(f"\n--- Image {i} ---")
            print(f"ID: {caption_info['id']}")
            print(f"Page: {caption_info['page']}")
            print(f"Caption: {caption_info['caption']}")
    else:
        print("No images found in the document")


def generate_document_summary(pipeline: MultiModalRAGPipeline, doc_path: str):
    """Generate a summary of a document"""
    
    print("\n" + "="*60)
    print("DOCUMENT SUMMARIZATION")
    print("="*60)
    
    if not os.path.exists(doc_path):
        print(f"Document not found: {doc_path}")
        return
    
    print(f"\nSummarizing: {doc_path}")
    summary = pipeline.summarize_document(doc_path)
    
    print(f"\nSummary:\n{summary}")


def query_with_detailed_context(pipeline: MultiModalRAGPipeline):
    """Query and show detailed context information"""
    
    print("\n" + "="*60)
    print("DETAILED CONTEXT RETRIEVAL")
    print("="*60)
    
    query = "What are the key findings?"
    
    print(f"\nQuery: {query}\n")
    result = pipeline.query_with_context(query, n_results=5)
    
    print(f"Answer: {result['answer']}\n")
    
    print("Retrieved Context:")
    for i, ctx in enumerate(result['context'], 1):
        print(f"\n--- Context {i} ---")
        print(f"Type: {ctx['type']}")
        print(f"Source: {ctx['source']} (page {ctx['page']})")
        print(f"Relevance Score: {ctx['score']:.4f}")
        
        # Show content preview
        content_preview = str(ctx['content'])[:200]
        if ctx['type'] != 'image':
            print(f"Content: {content_preview}...")


def main():
    """Run advanced examples"""
    
    # Initialize pipeline
    print("Initializing Multi-Modal RAG Pipeline...")
    pipeline = MultiModalRAGPipeline(enable_ocr=True)
    
    # Index documents
    data_dir = "data/sample"
    if os.path.exists(data_dir):
        print(f"\nIndexing directory: {data_dir}")
        stats = pipeline.index_directory(data_dir)
        print(f"Indexed {stats['total_elements']} total elements")
        print(f"Breakdown: {stats['modality_counts']}")
    else:
        print(f"\nData directory not found: {data_dir}")
        print("Creating sample directory...")
        os.makedirs(data_dir, exist_ok=True)
        print(f"Please add documents to {data_dir} and run again")
        return
    
    # Run examples
    try:
        analyze_document_structure(pipeline)
        search_by_modality(pipeline)
        query_with_detailed_context(pipeline)
        
        # Document-specific examples (if you have a specific document)
        doc_path = "data/sample/example.pdf"
        if os.path.exists(doc_path):
            generate_document_summary(pipeline, doc_path)
            extract_and_caption_images(pipeline, doc_path)
    
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Added documents to the data/sample/ directory")


if __name__ == "__main__":
    main()

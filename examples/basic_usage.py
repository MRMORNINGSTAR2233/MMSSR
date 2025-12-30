"""
Basic example of using the Multi-Modal RAG pipeline
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from mmssr import MultiModalRAGPipeline

# Load environment variables
load_dotenv()


def main():
    """
    Basic usage example:
    1. Create pipeline
    2. Index documents
    3. Query the system
    """
    
    # Create the pipeline
    print("Initializing Multi-Modal RAG Pipeline...")
    pipeline = MultiModalRAGPipeline(enable_ocr=True)
    
    # Index a document (replace with your document path)
    doc_path = "data/sample/example.pdf"
    
    if os.path.exists(doc_path):
        print(f"\nIndexing document: {doc_path}")
        n_elements = pipeline.index_document(doc_path)
        print(f"Indexed {n_elements} elements")
    else:
        print(f"\nDocument not found: {doc_path}")
        print("Please add documents to the data/sample/ directory")
        
        # Alternative: Index a directory
        data_dir = "data/sample"
        if os.path.exists(data_dir):
            print(f"\nIndexing directory: {data_dir}")
            stats = pipeline.index_directory(data_dir)
            print(f"Indexing statistics: {stats}")
    
    # Query examples
    queries = [
        "What is the main topic of this document?",
        "Are there any tables? If so, what do they contain?",
        "Describe any images or diagrams in the document",
    ]
    
    print("\n" + "="*60)
    print("QUERYING THE SYSTEM")
    print("="*60)
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        try:
            result = pipeline.query(query, n_results=5)
            
            print(f"Answer: {result['answer']}\n")
            
            if 'sources' in result:
                print("Sources:")
                for source in result['sources']:
                    print(f"  - {source['source']} (page {source['page']}, type: {source['type']})")
        
        except Exception as e:
            print(f"Error: {e}")
    
    # Get statistics
    print("\n" + "="*60)
    print("PIPELINE STATISTICS")
    print("="*60)
    stats = pipeline.get_statistics()
    print(f"Retriever stats: {stats['retriever_stats']}")
    print(f"Configuration: {stats['config']}")


if __name__ == "__main__":
    main()

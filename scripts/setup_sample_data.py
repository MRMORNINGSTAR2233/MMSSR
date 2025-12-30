#!/usr/bin/env python3
"""
Setup script to create sample data directories and download example documents
"""
import os
from pathlib import Path


def create_directory_structure():
    """Create the necessary directory structure"""
    
    directories = [
        "data/sample",
        "data/chroma_db",
        "data/raw",
        "data/processed",
        "models/downloaded",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")
    
    # Create a sample README in data/sample
    sample_readme = """# Sample Data Directory

Place your test documents here:
- PDFs with mixed content (text, images, tables)
- Technical manuals
- Research papers
- Product catalogs

Supported formats:
- .pdf
- .png, .jpg, .jpeg

Example documents:
1. Add a technical manual with diagrams
2. Add a research paper with tables and figures
3. Add product specifications with images

Then run:
```bash
python examples/basic_usage.py
```
"""
    
    with open("data/sample/README.md", "w") as f:
        f.write(sample_readme)
    
    print("\n✓ Directory structure created successfully!")
    print("\nNext steps:")
    print("1. Add your documents to data/sample/")
    print("2. Copy .env.example to .env and add your API keys")
    print("3. Run: python examples/basic_usage.py")


if __name__ == "__main__":
    create_directory_structure()

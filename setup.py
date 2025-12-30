from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mmssr",
    version="0.1.0",
    author="MMSSR Team",
    description="Multi-Modal and Semi-Structured RAG Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MRMORNINGSTAR2233/MMSSR",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.1.0",
        "openai>=1.10.0",
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "sentence-transformers>=2.3.0",
        "chromadb>=0.4.22",
        "unstructured>=0.12.0",
        "pillow>=10.0.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
)

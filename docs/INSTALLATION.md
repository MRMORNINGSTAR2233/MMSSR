# Installation Guide

Detailed installation instructions for MMSSR.

## System Requirements

- **Python**: 3.9 or higher
- **RAM**: Minimum 8GB (16GB recommended for large documents)
- **Storage**: At least 5GB for models and dependencies
- **GPU**: Optional, but recommended for faster processing
- **API Keys**: OpenAI API key for GPT-4V

## Platform-Specific Instructions

### macOS

```bash
# Install Python 3.9+ (if not already installed)
brew install python@3.11

# Clone repository
git clone https://github.com/MRMORNINGSTAR2233/MMSSR.git
cd MMSSR

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Verify installation
python -c "import mmssr; print('Installation successful!')"
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip
sudo apt-get install -y tesseract-ocr poppler-utils libmagic1

# Clone repository
git clone https://github.com/MRMORNINGSTAR2233/MMSSR.git
cd MMSSR

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment
cp .env.example .env
nano .env  # Add your OPENAI_API_KEY

# Verify installation
python -c "import mmssr; print('Installation successful!')"
```

### Windows

```powershell
# Install Python 3.11 from python.org

# Clone repository
git clone https://github.com/MRMORNINGSTAR2233/MMSSR.git
cd MMSSR

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment
copy .env.example .env
notepad .env  # Add your OPENAI_API_KEY

# Verify installation
python -c "import mmssr; print('Installation successful!')"
```

## Optional: GPU Support

For faster processing with CUDA-enabled GPUs:

```bash
# Install PyTorch with CUDA support
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Installing Additional OCR Support

### Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

## Development Installation

For development, install additional tools:

```bash
pip install -e ".[dev]"

# Or manually
pip install pytest black flake8 jupyter ipython
```

## Troubleshooting

### Issue: CLIP installation fails

```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### Issue: ChromaDB errors

```bash
pip uninstall chromadb
pip install chromadb --upgrade
```

### Issue: Unstructured parsing fails

```bash
pip install "unstructured[all-docs]" --upgrade
```

### Issue: Import errors

Make sure you're using Python 3.9+:
```bash
python --version
```

### Issue: Out of memory

Reduce batch sizes or use lighter models in config.py

## Verifying Installation

Run the test suite:

```bash
pytest tests/
```

Run a basic example:

```bash
python scripts/setup_sample_data.py
python examples/basic_usage.py
```

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstalling

```bash
# Deactivate virtual environment
deactivate

# Remove the project directory
cd ..
rm -rf MMSSR
```

## Docker Installation (Alternative)

Coming soon! We're working on a Docker image for easier deployment.

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing GitHub issues
3. Create a new issue with:
   - Your OS and Python version
   - Error message
   - Steps to reproduce

---

**Next Steps**: After installation, see [Quick Start Guide](../README.md#quick-start)

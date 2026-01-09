"""
Document parsing utilities for multi-modal content extraction
"""
import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

from PIL import Image
import pdfplumber
from unstructured.partition.auto import partition
from unstructured.documents.elements import (
    Title, NarrativeText, ListItem, Table, Image as UnstructuredImage
)

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Types of modalities in documents"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"


@dataclass
class DocumentElement:
    """Represents a single element from a document"""
    id: str
    type: ModalityType
    content: Any
    metadata: Dict[str, Any]
    source: str
    page_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert element to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "source": self.source,
            "page_number": self.page_number,
        }


class DocumentParser:
    """Base class for document parsing"""
    
    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        
    def parse(self, file_path: Union[str, Path]) -> List[DocumentElement]:
        """Parse a document into elements"""
        raise NotImplementedError


class PDFParser(DocumentParser):
    """Parse PDF documents into multi-modal elements"""
    
    def parse(self, file_path: Union[str, Path]) -> List[DocumentElement]:
        """
        Parse PDF using multiple strategies:
        1. Use unstructured.io for layout detection
        2. Extract tables with pdfplumber
        3. Extract images
        """
        file_path = Path(file_path)
        elements = []
        
        logger.info(f"Parsing PDF: {file_path}")
        
        # Use unstructured.io for initial partitioning
        try:
            partitioned_elements = partition(
                filename=str(file_path),
                strategy="hi_res" if self.enable_ocr else "fast",
                include_page_breaks=True,
            )
            
            element_counter = 0
            for elem in partitioned_elements:
                element_id = f"{file_path.stem}_{element_counter}"
                element_counter += 1
                
                metadata = elem.metadata.to_dict() if hasattr(elem.metadata, 'to_dict') else {}
                page_num = metadata.get('page_number')
                
                if isinstance(elem, (Title, NarrativeText, ListItem)):
                    # Text elements
                    elements.append(DocumentElement(
                        id=element_id,
                        type=ModalityType.TEXT,
                        content=elem.text,
                        metadata=metadata,
                        source=str(file_path),
                        page_number=page_num,
                    ))
                    
                elif isinstance(elem, Table):
                    # Table elements
                    elements.append(DocumentElement(
                        id=element_id,
                        type=ModalityType.TABLE,
                        content=elem.text,  # Table as text
                        metadata={**metadata, "table_html": elem.metadata.text_as_html if hasattr(elem.metadata, 'text_as_html') else None},
                        source=str(file_path),
                        page_number=page_num,
                    ))
                    
        except Exception as e:
            logger.warning(f"Unstructured parsing failed: {e}. Falling back to basic extraction.")
            elements.extend(self._fallback_parse(file_path))
        
        # Extract images separately
        elements.extend(self._extract_images_from_pdf(file_path))
        
        # Extract tables with pdfplumber for better table handling
        elements.extend(self._extract_tables_from_pdf(file_path))
        
        logger.info(f"Extracted {len(elements)} elements from {file_path}")
        return elements
    
    def _fallback_parse(self, file_path: Path) -> List[DocumentElement]:
        """Fallback parsing using pdfplumber"""
        elements = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    elements.append(DocumentElement(
                        id=f"{file_path.stem}_page_{page_num}",
                        type=ModalityType.TEXT,
                        content=text,
                        metadata={"extraction_method": "fallback"},
                        source=str(file_path),
                        page_number=page_num,
                    ))
        
        return elements
    
    def _extract_images_from_pdf(self, file_path: Path) -> List[DocumentElement]:
        """Extract images from PDF"""
        images = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract images from page
                    if hasattr(page, 'images'):
                        for img_idx, img in enumerate(page.images):
                            try:
                                # Get image coordinates and metadata
                                image_id = f"{file_path.stem}_img_p{page_num}_{img_idx}"
                                
                                images.append(DocumentElement(
                                    id=image_id,
                                    type=ModalityType.IMAGE,
                                    content=img,  # Image object
                                    metadata={
                                        "bbox": (img.get('x0'), img.get('top'), img.get('x1'), img.get('bottom')),
                                        "extraction_method": "pdfplumber"
                                    },
                                    source=str(file_path),
                                    page_number=page_num,
                                ))
                            except Exception as e:
                                logger.warning(f"Failed to extract image {img_idx} from page {page_num}: {e}")
        
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
        
        return images
    
    def _extract_tables_from_pdf(self, file_path: Path) -> List[DocumentElement]:
        """Extract tables from PDF using pdfplumber"""
        tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    extracted_tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(extracted_tables):
                        if table:
                            table_id = f"{file_path.stem}_table_p{page_num}_{table_idx}"
                            
                            # Convert table to text representation
                            table_text = self._table_to_text(table)
                            
                            tables.append(DocumentElement(
                                id=table_id,
                                type=ModalityType.TABLE,
                                content=table_text,
                                metadata={
                                    "raw_table": table,
                                    "extraction_method": "pdfplumber"
                                },
                                source=str(file_path),
                                page_number=page_num,
                            ))
        
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return tables
    
    @staticmethod
    def _table_to_text(table: List[List[str]]) -> str:
        """Convert table data to text representation"""
        if not table:
            return ""
        
        # Create a text representation
        lines = []
        for row in table:
            if row:
                # Clean None values
                cleaned_row = [str(cell) if cell is not None else "" for cell in row]
                lines.append(" | ".join(cleaned_row))
        
        return "\n".join(lines)


class ImageParser(DocumentParser):
    """Parse standalone image files"""
    
    def parse(self, file_path: Union[str, Path]) -> List[DocumentElement]:
        """Parse an image file"""
        file_path = Path(file_path)
        elements = []
        
        try:
            # Load image
            img = Image.open(file_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save image as base64 for storage
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            elements.append(DocumentElement(
                id=f"{file_path.stem}_img",
                type=ModalityType.IMAGE,
                content=img_base64,
                metadata={
                    "format": img.format,
                    "size": img.size,
                    "mode": img.mode,
                },
                source=str(file_path),
            ))
            
        except Exception as e:
            logger.error(f"Failed to parse image {file_path}: {e}")
        
        return elements


class MultiModalParser:
    """Main parser that handles multiple document types"""
    
    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        self.parsers = {
            '.pdf': PDFParser(enable_ocr=enable_ocr),
            '.png': ImageParser(enable_ocr=enable_ocr),
            '.jpg': ImageParser(enable_ocr=enable_ocr),
            '.jpeg': ImageParser(enable_ocr=enable_ocr),
        }
    
    def parse(self, file_path: Union[str, Path]) -> List[DocumentElement]:
        """Parse a document based on its file type"""
        file_path = Path(file_path)
        
        # Check file type first so we raise a clear error for unsupported formats
        suffix = file_path.suffix.lower()
        parser = self.parsers.get(suffix)
        
        if not parser:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return parser.parse(file_path)
    
    def parse_directory(self, directory: Union[str, Path]) -> List[DocumentElement]:
        """Parse all supported documents in a directory"""
        directory = Path(directory)
        all_elements = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.parsers:
                try:
                    elements = self.parse(file_path)
                    all_elements.extend(elements)
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}")
        
        return all_elements

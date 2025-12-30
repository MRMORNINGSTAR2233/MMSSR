"""
Multi-modal generation using LLMs with vision capabilities
"""
import base64
from typing import List, Dict, Any, Optional
import logging

import openai
from openai import OpenAI

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from .parsers import DocumentElement, ModalityType
from .config import config

logger = logging.getLogger(__name__)


class MultiModalGenerator:
    """
    Generator that can process multi-modal context and generate answers
    
    Supports multiple providers: OpenAI, Gemini, Groq
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        vision_model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.provider = provider or config.llm.provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set models based on provider
        if model:
            self.model = model
        else:
            self.model = config.llm.text_model
            
        if vision_model:
            self.vision_model = vision_model
        else:
            self.vision_model = config.llm.vision_model
        
        # Initialize clients based on provider
        if self.provider == "openai":
            self.client = OpenAI(api_key=config.openai_api_key)
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
            genai.configure(api_key=config.google_api_key)
            self.client = genai
        elif self.provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("groq not installed. Run: pip install groq")
            self.client = Groq(api_key=config.groq_api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        logger.info(f"Initialized generator with provider: {self.provider}, model: {self.model}, vision: {self.vision_model}")
    
    def generate(
        self,
        query: str,
        context_elements: List[DocumentElement],
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate answer given query and retrieved context
        
        Args:
            query: User query
            context_elements: Retrieved document elements
            include_sources: Whether to include source information in response
            
        Returns:
            Dictionary with answer and metadata
        """
        # Check if we have images in context
        has_images = any(elem.type == ModalityType.IMAGE for elem in context_elements)
        
        if has_images:
            # Use vision model
            response = self._generate_with_vision(query, context_elements)
        else:
            # Use text-only model
            response = self._generate_text_only(query, context_elements)
        
        # Prepare result
        result = {
            "answer": response,
            "query": query,
            "n_context_elements": len(context_elements),
        }
        
        if include_sources:
            result["sources"] = self._extract_sources(context_elements)
        
        return result
    
    def _generate_text_only(
        self,
        query: str,
        context_elements: List[DocumentElement],
    ) -> str:
        """Generate answer using text-only model"""
        # Build context from elements
        context_parts = []
        
        for i, elem in enumerate(context_elements, 1):
            if elem.type == ModalityType.TEXT:
                context_parts.append(f"[Text {i}]: {elem.content}")
            elif elem.type == ModalityType.TABLE:
                context_parts.append(f"[Table {i}]:\n{elem.content}")
            elif elem.type == ModalityType.IMAGE:
                # For text-only model, add image description if available
                context_parts.append(f"[Image {i}]: Image from {elem.source}, page {elem.page_number or 'N/A'}")
        
        context_text = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = self._create_text_prompt(query, context_text)
        
        # Generate response based on provider
        try:
            if self.provider == "openai":
                return self._generate_openai_text(prompt)
            elif self.provider == "gemini":
                return self._generate_gemini_text(prompt)
            elif self.provider == "groq":
                return self._generate_groq_text(prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_openai_text(self, prompt: str) -> str:
        """Generate using OpenAI"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. "
                               "If the context contains tables, analyze them carefully. "
                               "Always cite which context element (Text 1, Table 2, etc.) you're using in your answer."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return response.choices[0].message.content
    
    def _generate_gemini_text(self, prompt: str) -> str:
        """Generate using Gemini"""
        model = self.client.GenerativeModel(
            model_name=self.model,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }
        )
        
        full_prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the context contains tables, analyze them carefully. "
            "Always cite which context element (Text 1, Table 2, etc.) you're using in your answer.\n\n"
            + prompt
        )
        
        response = model.generate_content(full_prompt)
        return response.text
    
    def _generate_groq_text(self, prompt: str) -> str:
        """Generate using Groq"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. "
                               "If the context contains tables, analyze them carefully. "
                               "Always cite which context element (Text 1, Table 2, etc.) you're using in your answer."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return response.choices[0].message.content
    
    def _generate_with_vision(
        self,
        query: str,
        context_elements: List[DocumentElement],
    ) -> str:
        """Generate answer using vision model"""
        
        if self.provider == "openai":
            return self._generate_openai_vision(query, context_elements)
        elif self.provider == "gemini":
            return self._generate_gemini_vision(query, context_elements)
        elif self.provider == "groq":
            return self._generate_groq_vision(query, context_elements)
        else:
            logger.warning(f"Vision not fully supported for {self.provider}, falling back to text")
            return self._generate_text_only(query, context_elements)
    
    def _generate_openai_vision(
        self,
        query: str,
        context_elements: List[DocumentElement],
    ) -> str:
        """Generate answer using OpenAI GPT-4V"""
        # Build multi-modal messages
        message_content = []
        
        # Add query
        message_content.append({
            "type": "text",
            "text": f"Question: {query}\n\nContext:"
        })
        
        # Add context elements
        for i, elem in enumerate(context_elements, 1):
            if elem.type == ModalityType.TEXT:
                message_content.append({
                    "type": "text",
                    "text": f"\n[Text {i}]: {elem.content}"
                })
            
            elif elem.type == ModalityType.TABLE:
                message_content.append({
                    "type": "text",
                    "text": f"\n[Table {i}]:\n{elem.content}"
                })
            
            elif elem.type == ModalityType.IMAGE:
                # Add image
                try:
                    # Ensure image is base64 encoded
                    if isinstance(elem.content, str):
                        image_data = elem.content
                    else:
                        # Convert to base64 if needed
                        image_data = base64.b64encode(elem.content).decode()
                    
                    message_content.append({
                        "type": "text",
                        "text": f"\n[Image {i}]:"
                    })
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                            "detail": "high"
                        }
                    })
                except Exception as e:
                    logger.warning(f"Failed to add image {i}: {e}")
                    message_content.append({
                        "type": "text",
                        "text": f"\n[Image {i}]: (Image unavailable)"
                    })
        
        message_content.append({
            "type": "text",
            "text": "\n\nBased on the above context, please answer the question. "
                    "If you reference specific information, cite which element (Text 1, Table 2, Image 3, etc.) it came from."
        })
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can analyze text, tables, and images to answer questions. "
                                   "Always cite your sources by referring to the element numbers."
                    },
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Vision generation failed: {e}")
            # Fallback to text-only
            logger.info("Falling back to text-only generation")
            return self._generate_text_only(query, context_elements)
    
    def _generate_gemini_vision(
        self,
        query: str,
        context_elements: List[DocumentElement],
    ) -> str:
        """Generate answer using Gemini with vision"""
        try:
            from PIL import Image as PILImage
            
            model = self.client.GenerativeModel(
                model_name=self.vision_model,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            
            # Build content parts
            content_parts = []
            
            # Add instruction
            instruction = (
                f"Question: {query}\n\n"
                "Context:\n"
                "You are a helpful assistant that can analyze text, tables, and images. "
                "Always cite your sources by referring to element numbers.\n\n"
            )
            content_parts.append(instruction)
            
            # Add elements
            for i, elem in enumerate(context_elements, 1):
                if elem.type == ModalityType.TEXT:
                    content_parts.append(f"[Text {i}]: {elem.content}\n\n")
                
                elif elem.type == ModalityType.TABLE:
                    content_parts.append(f"[Table {i}]:\n{elem.content}\n\n")
                
                elif elem.type == ModalityType.IMAGE:
                    try:
                        content_parts.append(f"[Image {i}]:\n")
                        
                        # Convert base64 to PIL Image for Gemini
                        if isinstance(elem.content, str):
                            import io
                            image_data = base64.b64decode(elem.content)
                            image = PILImage.open(io.BytesIO(image_data))
                            content_parts.append(image)
                        
                    except Exception as e:
                        logger.warning(f"Failed to add image {i}: {e}")
                        content_parts.append(f"[Image {i}]: (Image unavailable)\n\n")
            
            content_parts.append("\nBased on the above context, please answer the question.")
            
            response = model.generate_content(content_parts)
            return response.text
        
        except Exception as e:
            logger.error(f"Gemini vision generation failed: {e}")
            return self._generate_text_only(query, context_elements)
    
    def _generate_groq_vision(
        self,
        query: str,
        context_elements: List[DocumentElement],
    ) -> str:
        """Generate answer using Groq with vision"""
        # Groq supports vision via llama-3.2-90b-vision-preview
        message_content = []
        
        # Add query
        message_content.append({
            "type": "text",
            "text": f"Question: {query}\n\nContext:"
        })
        
        # Add context elements
        for i, elem in enumerate(context_elements, 1):
            if elem.type == ModalityType.TEXT:
                message_content.append({
                    "type": "text",
                    "text": f"\n[Text {i}]: {elem.content}"
                })
            
            elif elem.type == ModalityType.TABLE:
                message_content.append({
                    "type": "text",
                    "text": f"\n[Table {i}]:\n{elem.content}"
                })
            
            elif elem.type == ModalityType.IMAGE:
                try:
                    if isinstance(elem.content, str):
                        image_data = elem.content
                    else:
                        image_data = base64.b64encode(elem.content).decode()
                    
                    message_content.append({
                        "type": "text",
                        "text": f"\n[Image {i}]:"
                    })
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    })
                except Exception as e:
                    logger.warning(f"Failed to add image {i}: {e}")
                    message_content.append({
                        "type": "text",
                        "text": f"\n[Image {i}]: (Image unavailable)"
                    })
        
        message_content.append({
            "type": "text",
            "text": "\n\nBased on the above context, please answer the question."
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can analyze text, tables, and images."
                    },
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Groq vision generation failed: {e}")
            return self._generate_text_only(query, context_elements)
    
    def _create_text_prompt(self, query: str, context: str) -> str:
        """Create prompt for text-only generation"""
        return f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. If you use information from the context, cite which element it came from."""
    
    def _extract_sources(self, elements: List[DocumentElement]) -> List[Dict[str, Any]]:
        """Extract source information from elements"""
        sources = []
        seen_sources = set()
        
        for elem in elements:
            source_key = (elem.source, elem.page_number)
            if source_key not in seen_sources:
                sources.append({
                    "source": elem.source,
                    "page": elem.page_number,
                    "type": elem.type.value,
                })
                seen_sources.add(source_key)
        
        return sources
    
    def generate_summary(self, elements: List[DocumentElement]) -> str:
        """Generate a summary of document elements"""
        context_parts = []
        
        for elem in elements:
            if elem.type in [ModalityType.TEXT, ModalityType.TABLE]:
                context_parts.append(elem.content[:500])  # Truncate
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Please provide a concise summary of the following content:

{context}

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Error generating summary"
    
    def caption_image(self, image_element: DocumentElement) -> str:
        """Generate a caption for an image"""
        if image_element.type != ModalityType.IMAGE:
            raise ValueError("Element must be an image")
        
        try:
            image_data = image_element.content
            if not isinstance(image_data, str):
                image_data = base64.b64encode(image_data).decode()
            
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that describes images in detail."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please provide a detailed description of this image:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.5,
                max_tokens=300,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Image captioning failed: {e}")
            return "Error generating image caption"

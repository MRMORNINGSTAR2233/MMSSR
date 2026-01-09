"""
Multi-vector retriever for multi-modal RAG
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

import chromadb
from chromadb.config import Settings
import numpy as np

from .parsers import DocumentElement, ModalityType
from .embedders import MultiModalEmbedder
from .config import config

logger = logging.getLogger(__name__)


class MultiModalVectorStore:
    """
    Vector store for multi-modal embeddings using ChromaDB
    
    Supports multiple collections for different modalities
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or config.vector_store.persist_directory
        self.collection_name = collection_name or config.vector_store.collection_name
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collections for different modalities
        self.collections: Dict[str, chromadb.Collection] = {}
        
        logger.info(f"Initialized vector store at {self.persist_directory}")
    
    def _get_collection(self, modality: ModalityType, embedding_dim: int) -> chromadb.Collection:
        """Get or create collection for a modality"""
        collection_key = f"{self.collection_name}_{modality.value}"
        
        if collection_key not in self.collections:
            # Create or get collection
            try:
                collection = self.client.get_or_create_collection(
                    name=collection_key,
                    metadata={"modality": modality.value, "dimension": embedding_dim}
                )
                self.collections[collection_key] = collection
                logger.info(f"Created/loaded collection: {collection_key}")
            except Exception as e:
                logger.error(f"Failed to create collection {collection_key}: {e}")
                raise
        
        return self.collections[collection_key]
    
    def add_elements(
        self,
        elements: List[DocumentElement],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add document elements with their embeddings to the vector store
        
        Args:
            elements: List of document elements
            embeddings: Corresponding embeddings (n_elements, embedding_dim)
        """
        if len(elements) != len(embeddings):
            raise ValueError("Number of elements and embeddings must match")
        
        # Group elements by modality
        modality_groups: Dict[ModalityType, List[Tuple[DocumentElement, np.ndarray]]] = {}
        
        for element, embedding in zip(elements, embeddings):
            if element.type not in modality_groups:
                modality_groups[element.type] = []
            modality_groups[element.type].append((element, embedding))
        
        # Add to respective collections
        for modality, group in modality_groups.items():
            elements_batch = [item[0] for item in group]
            embeddings_batch = np.array([item[1] for item in group])
            
            self._add_to_collection(modality, elements_batch, embeddings_batch)
    
    def _add_to_collection(
        self,
        modality: ModalityType,
        elements: List[DocumentElement],
        embeddings: np.ndarray,
    ) -> None:
        """Add elements to a specific modality collection"""
        collection = self._get_collection(modality, embeddings.shape[1])
        
        # Prepare data for ChromaDB
        ids = [elem.id for elem in elements]
        embeddings_list = embeddings.tolist()
        
        # Metadata (exclude heavy content)
        metadatas = []
        documents = []
        
        for elem in elements:
            # Store lightweight metadata
            metadata = {
                "type": elem.type.value,
                "source": elem.source,
                "page_number": elem.page_number or -1,
                **{k: v for k, v in elem.metadata.items() 
                   if isinstance(v, (str, int, float, bool))}
            }
            metadatas.append(metadata)
            
            # Store content as document (text representation)
            if elem.type == ModalityType.IMAGE:
                # Store image reference, not full base64
                doc = f"Image from {elem.source}, page {elem.page_number or 'N/A'}"
            else:
                doc = str(elem.content)[:1000]  # Truncate long content
            
            documents.append(doc)
        
        # Add to collection
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents,
            )
            logger.info(f"Added {len(elements)} elements to {modality.value} collection")
        except Exception as e:
            logger.error(f"Failed to add elements to collection: {e}")
            raise
    
    def query(
        self,
        query_embedding: np.ndarray,
        modality: Optional[ModalityType] = None,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store
        
        Args:
            query_embedding: Query embedding vector
            modality: Optional modality filter
            n_results: Number of results to return
            
        Returns:
            List of retrieved elements with metadata
        """
        results = []
        
        # Determine which collections to query
        if modality:
            collections_to_query = [modality]
        else:
            # Query all modalities
            collections_to_query = list(ModalityType)
        
        for mod in collections_to_query:
            collection_key = f"{self.collection_name}_{mod.value}"
            
            if collection_key in self.collections or self._collection_exists(collection_key):
                try:
                    collection = self._get_collection(mod, len(query_embedding))
                    
                    # Query collection
                    query_results = collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=n_results,
                    )
                    
                    # Parse results
                    for i in range(len(query_results['ids'][0])):
                        results.append({
                            'id': query_results['ids'][0][i],
                            'document': query_results['documents'][0][i],
                            'metadata': query_results['metadatas'][0][i],
                            'distance': query_results['distances'][0][i],
                            'modality': mod.value,
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to query {mod.value} collection: {e}")
        
        # Sort by distance (lower is better)
        results.sort(key=lambda x: x['distance'])
        
        return results[:n_results]
    
    def _collection_exists(self, name: str) -> bool:
        """Check if a collection exists"""
        try:
            self.client.get_collection(name)
            return True
        except:
            return False
    
    def delete_collection(self, modality: Optional[ModalityType] = None) -> None:
        """Delete collection(s)"""
        if modality:
            collection_key = f"{self.collection_name}_{modality.value}"
            try:
                self.client.delete_collection(collection_key)
                if collection_key in self.collections:
                    del self.collections[collection_key]
                logger.info(f"Deleted collection: {collection_key}")
            except Exception as e:
                logger.warning(f"Failed to delete collection {collection_key}: {e}")
        else:
            # Delete all collections
            for mod in ModalityType:
                self.delete_collection(mod)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        stats = {}
        
        for mod in ModalityType:
            collection_key = f"{self.collection_name}_{mod.value}"
            if self._collection_exists(collection_key):
                try:
                    # Use existing collection without attempting to create it with a dummy dimension
                    collection = self.client.get_collection(collection_key)
                    stats[mod.value] = collection.count()
                except Exception:
                    # Fallback to cached collection if available
                    collection = self.collections.get(collection_key)
                    if collection:
                        stats[mod.value] = collection.count()
                    else:
                        stats[mod.value] = 0
        
        return stats


class MultiModalRetriever:
    """
    Multi-modal retriever that combines vector store and embedder
    """
    
    def __init__(
        self,
        embedder: Optional[MultiModalEmbedder] = None,
        vector_store: Optional[MultiModalVectorStore] = None,
    ):
        self.embedder = embedder or MultiModalEmbedder()
        self.vector_store = vector_store or MultiModalVectorStore()
        
        # Cache for full elements (since ChromaDB stores truncated content)
        self.element_cache: Dict[str, DocumentElement] = {}
    
    def index_documents(self, elements: List[DocumentElement]) -> None:
        """
        Index document elements in the vector store
        
        Args:
            elements: List of document elements to index
        """
        logger.info(f"Indexing {len(elements)} elements...")
        
        # Generate embeddings
        embeddings = self.embedder.embed_elements(elements)
        
        # Add to vector store
        self.vector_store.add_elements(elements, embeddings)
        
        # Cache elements for later retrieval
        for element in elements:
            self.element_cache[element.id] = element
        
        logger.info(f"Successfully indexed {len(elements)} elements")
    
    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        modality: Optional[ModalityType] = None,
        return_full_elements: bool = True,
    ) -> List[DocumentElement]:
        """
        Retrieve relevant elements for a query
        
        Args:
            query: Query string
            n_results: Number of results to return
            modality: Optional modality filter
            return_full_elements: Whether to return full cached elements
            
        Returns:
            List of retrieved document elements
        """
        # Embed query
        query_embedding = self.embedder.embed_query(query, target_modality=modality)
        
        # Query vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            modality=modality,
            n_results=n_results,
        )
        
        # Return results
        if return_full_elements:
            # Try to get full elements from cache
            retrieved_elements = []
            for result in results:
                elem_id = result['id']
                if elem_id in self.element_cache:
                    retrieved_elements.append(self.element_cache[elem_id])
                else:
                    # Reconstruct from stored data
                    elem = DocumentElement(
                        id=elem_id,
                        type=ModalityType(result['metadata']['type']),
                        content=result['document'],
                        metadata=result['metadata'],
                        source=result['metadata']['source'],
                        page_number=result['metadata'].get('page_number'),
                    )
                    retrieved_elements.append(elem)
            
            return retrieved_elements
        else:
            # Return raw results
            return results
    
    def retrieve_with_scores(
        self,
        query: str,
        n_results: int = 10,
        modality: Optional[ModalityType] = None,
    ) -> List[Tuple[DocumentElement, float]]:
        """
        Retrieve elements with relevance scores
        
        Returns:
            List of (element, score) tuples
        """
        query_embedding = self.embedder.embed_query(query, target_modality=modality)
        results = self.vector_store.query(
            query_embedding=query_embedding,
            modality=modality,
            n_results=n_results,
        )
        
        retrieved = []
        for result in results:
            elem_id = result['id']
            score = 1.0 / (1.0 + result['distance'])  # Convert distance to similarity
            
            if elem_id in self.element_cache:
                elem = self.element_cache[elem_id]
            else:
                elem = DocumentElement(
                    id=elem_id,
                    type=ModalityType(result['metadata']['type']),
                    content=result['document'],
                    metadata=result['metadata'],
                    source=result['metadata']['source'],
                    page_number=result['metadata'].get('page_number'),
                )
            
            retrieved.append((elem, score))
        
        return retrieved
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        stats = self.vector_store.get_stats()
        stats['cached_elements'] = len(self.element_cache)
        return stats

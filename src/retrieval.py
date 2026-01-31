"""
Retrieval module for Answer IQ.
Handles FAISS index management and semantic search.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import faiss

from .config import (
    TOP_K_RESULTS, 
    SIMILARITY_THRESHOLD, 
    FAISS_INDEX_FILE, 
    METADATA_FILE,
    EMBEDDING_DIMENSION
)
from .ingestion import TextChunk
from .embeddings import EmbeddingModel


@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    chunk: TextChunk
    score: float
    rank: int


class FAISSIndex:
    """
    FAISS-based vector index for efficient similarity search.
    Supports persistence and incremental updates.
    """
    
    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        """
        Initialize the FAISS index.
        
        Args:
            dimension: Embedding vector dimension
        """
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner product for cosine sim
        self.chunk_ids: List[str] = []
        self.metadata: Dict[str, dict] = {}
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Create a new FAISS index."""
        # Using IndexFlatIP for normalized vectors (equivalent to cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_ids = []
        self.metadata = {}
    
    def add(self, embeddings: np.ndarray, chunks: List[TextChunk]) -> None:
        """
        Add embeddings and their associated chunks to the index.
        
        Args:
            embeddings: Numpy array of shape (n, dimension)
            chunks: List of TextChunk objects
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store chunk references
        for chunk in chunks:
            self.chunk_ids.append(chunk.id)
            self.metadata[chunk.id] = {
                'content': chunk.content,
                'document_id': chunk.document_id,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata
            }
    
    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K_RESULTS) -> List[Tuple[str, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector of shape (dimension,)
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Reshape for FAISS
        query = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                results.append((chunk_id, float(score)))
        
        return results
    
    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """Get the content of a chunk by ID."""
        if chunk_id in self.metadata:
            return self.metadata[chunk_id]['content']
        return None
    
    def get_chunk_metadata(self, chunk_id: str) -> Optional[dict]:
        """Get full metadata for a chunk."""
        return self.metadata.get(chunk_id)
    
    def save(self, index_path: Path = FAISS_INDEX_FILE, 
             metadata_path: Path = METADATA_FILE) -> None:
        """
        Save the index and metadata to disk.
        
        Args:
            index_path: Path for FAISS index file
            metadata_path: Path for metadata JSON file
        """
        # Create directories if needed
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_export = {
            'dimension': self.dimension,
            'chunk_ids': self.chunk_ids,
            'metadata': self.metadata
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_export, f, ensure_ascii=False, indent=2)
    
    def load(self, index_path: Path = FAISS_INDEX_FILE, 
             metadata_path: Path = METADATA_FILE) -> bool:
        """
        Load the index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not index_path.exists() or not metadata_path.exists():
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.dimension = data['dimension']
            self.chunk_ids = data['chunk_ids']
            self.metadata = data['metadata']
            
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            self._initialize_index()
            return False
    
    def clear(self) -> None:
        """Clear the index and metadata."""
        self._initialize_index()
    
    @property
    def size(self) -> int:
        """Return number of indexed vectors."""
        return self.index.ntotal if self.index else 0


class Retriever:
    """
    High-level retrieval interface.
    Combines embedding generation and FAISS search.
    """
    
    def __init__(self, embedding_model: EmbeddingModel, index: FAISSIndex):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: Model for generating query embeddings
            index: FAISS index for search
        """
        self.embedding_model = embedding_model
        self.index = index
    
    def index_chunks(self, chunks: List[TextChunk], 
                     show_progress: bool = False) -> int:
        """
        Index a list of chunks.
        
        Args:
            chunks: List of TextChunk objects to index
            show_progress: Whether to show progress
            
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress=show_progress)
        
        # Add to index
        self.index.add(embeddings, chunks)
        
        return len(chunks)
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS, 
               threshold: float = SIMILARITY_THRESHOLD) -> List[SearchResult]:
        """
        Search for chunks relevant to the query.
        
        Args:
            query: Natural language query
            top_k: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(query)
        
        # Search index
        raw_results = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for rank, (chunk_id, score) in enumerate(raw_results):
            if score >= threshold:
                metadata = self.index.get_chunk_metadata(chunk_id)
                if metadata:
                    chunk = TextChunk(
                        id=chunk_id,
                        document_id=metadata['document_id'],
                        content=metadata['content'],
                        chunk_index=metadata['chunk_index'],
                        start_char=0,
                        end_char=len(metadata['content']),
                        metadata=metadata['metadata']
                    )
                    results.append(SearchResult(
                        chunk=chunk,
                        score=score,
                        rank=rank + 1
                    ))
        
        return results
    
    def get_context(self, query: str, top_k: int = TOP_K_RESULTS) -> Tuple[str, List[SearchResult]]:
        """
        Get formatted context for a query.
        
        Args:
            query: Natural language query
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (formatted_context_string, list_of_results)
        """
        results = self.search(query, top_k)
        
        if not results:
            return "", []
        
        context_parts = []
        for i, result in enumerate(results):
            source = result.chunk.metadata.get('filename', 'Unknown')
            context_parts.append(
                f"[Source: {source}]\n{result.chunk.content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        return context, results

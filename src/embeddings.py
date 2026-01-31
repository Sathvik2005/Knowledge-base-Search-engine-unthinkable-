"""
Embeddings module for Answer IQ.
Handles vector embedding generation using transformer models.
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME, DEVICE


class EmbeddingModel:
    """
    Wrapper for transformer-based embedding models.
    Uses Sentence Transformers for efficient text embedding.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None
    
    def load(self) -> None:
        """Load the embedding model into memory."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=DEVICE)
            # Get embedding dimension
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self._dimension = test_embedding.shape[1]
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            self.load()
        return self._dimension
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of shape (len(texts), embedding_dimension)
        """
        if self.model is None:
            self.load()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Numpy array of shape (embedding_dimension,)
        """
        return self.encode([text])[0]


class EmbeddingCache:
    """
    Cache for storing pre-computed embeddings.
    Enables efficient reuse of embeddings across sessions.
    """
    
    def __init__(self):
        self._cache: dict = {}
    
    def get(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get cached embedding by chunk ID."""
        return self._cache.get(chunk_id)
    
    def set(self, chunk_id: str, embedding: np.ndarray) -> None:
        """Cache an embedding."""
        self._cache[chunk_id] = embedding
    
    def has(self, chunk_id: str) -> bool:
        """Check if embedding is cached."""
        return chunk_id in self._cache
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
    
    @property
    def size(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)

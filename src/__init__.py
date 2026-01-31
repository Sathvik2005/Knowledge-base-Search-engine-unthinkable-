"""
Answer IQ - Source Package Initialization
"""

from .config import *
from .ingestion import DocumentLoader, TextSplitter, IngestionPipeline, Document, TextChunk
from .embeddings import EmbeddingModel, EmbeddingCache
from .retrieval import FAISSIndex, Retriever, SearchResult
from .generation import AnswerGenerator, QueryProcessor, GeneratedAnswer
from .audio import WhisperTranscriber, AudioProcessor, TranscriptionResult

__version__ = "2.0.0"
__all__ = [
    # Config
    'CHUNK_SIZE', 'CHUNK_OVERLAP', 'EMBEDDING_MODEL_NAME', 'GENERATION_MODEL_NAME',
    'TOP_K_RESULTS', 'WHISPER_MODEL_SIZE',
    # Ingestion
    'DocumentLoader', 'TextSplitter', 'IngestionPipeline', 'Document', 'TextChunk',
    # Embeddings
    'EmbeddingModel', 'EmbeddingCache',
    # Retrieval
    'FAISSIndex', 'Retriever', 'SearchResult',
    # Generation
    'AnswerGenerator', 'QueryProcessor', 'GeneratedAnswer',
    # Audio
    'WhisperTranscriber', 'AudioProcessor', 'TranscriptionResult',
]

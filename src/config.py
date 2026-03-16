"""
Configuration settings for Answer IQ (RAG-Gemini) system.
All configurable parameters are centralized here for easy maintenance.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
INDEX_DIR = CACHE_DIR / "faiss_index"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# Document processing settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SUPPORTED_FILE_TYPES = [".pdf", ".txt"]

# Embedding model settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Retrieval settings
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.3

# Generation model settings
GENERATION_MODEL_NAME = "google/flan-t5-base"
GENERATION_API_MODEL = "gpt-4o-mini"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
DO_SAMPLE = True

# Whisper settings - Using HuggingFace Transformers pipeline
WHISPER_MODEL_NAME = "openai/whisper-large-v3"  # High accuracy model
WHISPER_MODEL_SIZE = "large-v3"  # For display purposes
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]

# FAISS index settings
FAISS_INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"

# Session settings
HISTORY_FILE = CACHE_DIR / "query_history.json"
MAX_HISTORY_ENTRIES = 100

# Device configuration
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Disable TensorFlow to avoid Keras conflicts
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

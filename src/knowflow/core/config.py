"""
KnowFlow AI - Configuration Module
Centralized settings, constants, and paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
UPLOADS_DIR = DATA_DIR / "uploads"
EVAL_DIR = PROJECT_ROOT / "eval_data"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB Settings
CHROMA_COLLECTION_NAME = "knowflow_kb"

# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking Parameters
DEFAULT_CHUNK_SIZE = 600       # Target characters per chunk (~100-120 words)
DEFAULT_CHUNK_OVERLAP = 120    # Overlap characters to preserve context across boundaries
MIN_CHUNK_SIZE = 50            # Minimum meaningful chunk size

# Retrieval Parameters
DEFAULT_TOP_K = 4              # Number of top chunks returned after MMR
DEFAULT_FETCH_K = 20           # Number of initial candidates fetched for MMR
DEFAULT_LAMBDA_MMR = 0.6       # Balance between query relevance (1.0) and diversity (0.0)
MIN_SIMILARITY_THRESHOLD = 0.25 # Minimum cosine similarity to avoid irrelevant hallucinations

# Generation & LLM
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
FALLBACK_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant"
]
MAX_GENERATION_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.15

# Supported File Extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max per file

import os

from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "anthropic"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")  # butuh ANTHROPIC_API_KEY di .env

# Embedding (multilingual, cocok untuk modul berbahasa Indonesia)
# Model e5 butuh prefix "query: " / "passage: ". Kalau ganti model non-e5, kosongkan prefix-nya.
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "query: ")
PASSAGE_PREFIX = os.getenv("PASSAGE_PREFIX", "passage: ")

# Storage
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")

# Retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))       # karakter per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))

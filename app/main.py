"""API FastAPI: upload PDF, list/hapus dokumen, tanya jawab."""
import json
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import llm
from . import config
from .config import TOP_K, UPLOAD_DIR
from .ingest import build_chunks
from .store import VectorStore

STATIC = Path(__file__).parent / "static"
store: VectorStore | None = None

# Cache saran pertanyaan per dokumen: {nama: {"questions": [...], "at": epoch}}.
# Membuat saran butuh ~10 detik di LLM lokal, jadi cukup sekali per dokumen.
SUGGEST_FILE = Path(UPLOAD_DIR).parent / "suggestions.json"
_suggest_lock = threading.Lock()


def _read_cache() -> dict:
    try:
        return json.loads(SUGGEST_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _update_cache(name: str, entry: dict | None) -> None:
    """Simpan (entry) atau hapus (None) cache satu dokumen."""
    with _suggest_lock:
        cache = _read_cache()
        if entry is None:
            cache.pop(name, None)
        else:
            cache[name] = entry
        SUGGEST_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    store = VectorStore()  # load model embedding sekali saat startup
    yield


app = FastAPI(title="RAG Modul Kuliah", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=TOP_K, ge=1, le=20)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


# Endpoint pakai `def` biasa (bukan async): FastAPI otomatis menjalankannya di threadpool,
# jadi proses embedding / panggilan LLM yang blocking tidak membekukan server.
@app.post("/documents")
def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Hanya file PDF.")

    dest = Path(UPLOAD_DIR) / name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = build_chunks(dest, name)
    if not chunks:
        raise HTTPException(
            422, "Tidak ada teks yang bisa diekstrak. PDF hasil scan perlu OCR dulu."
        )
    n = store.add_document(name, chunks)
    _update_cache(name, None)  # isi berubah, saran lama tidak berlaku
    return {"name": name, "chunks": n, "pages": len({c["page"] for c in chunks})}


@app.get("/documents")
def list_documents():
    return store.list_documents()


@app.get("/suggestions")
def cached_suggestions():
    """Semua saran yang sudah tersimpan, tanpa memanggil LLM."""
    return _read_cache()


@app.get("/documents/{name}/suggestions")
def suggestions(name: str, refresh: bool = False):
    if not refresh and (hit := _read_cache().get(name)):
        return {**hit, "cached": True}
    passages = store.sample(name)
    if not passages:
        raise HTTPException(404, "Dokumen tidak ditemukan.")
    try:
        questions = llm.suggest(passages)
    except Exception as e:
        raise HTTPException(502, f"Gagal memanggil LLM: {e}")
    entry = {"questions": questions, "at": time.time()}
    if questions:  # jangan cache hasil kosong, biar bisa dicoba lagi
        _update_cache(name, entry)
    return {**entry, "cached": False}


@app.get("/stats")
def stats():
    llm_model = config.ANTHROPIC_MODEL if config.LLM_PROVIDER == "anthropic" else config.OLLAMA_MODEL
    return {
        "documents": len(store.list_documents()),
        "vectors": store.col.count(),
        "dim": store.dim,
        "embed_model": config.EMBED_MODEL.split("/")[-1],
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "top_k": config.TOP_K,
        "llm": f"{config.LLM_PROVIDER} · {llm_model}",
    }


@app.delete("/documents/{name}")
def delete_document(name: str):
    store.delete_document(name)
    (Path(UPLOAD_DIR) / Path(name).name).unlink(missing_ok=True)
    _update_cache(name, None)
    return {"deleted": name}


@app.post("/ask")
def ask(req: AskRequest):
    hits = store.search(req.question, req.top_k)
    if not hits:
        return {"answer": "Belum ada dokumen. Upload PDF dulu.", "sources": []}

    try:
        answer = llm.generate(req.question, hits)
    except Exception as e:  # koneksi Ollama mati, API key salah, dll.
        raise HTTPException(502, f"Gagal memanggil LLM: {e}")

    return {
        "answer": answer,
        "sources": [
            {
                "n": i,
                "source": h["source"],
                "page": h["page"],
                "score": h["score"],
                "snippet": h["text"][:300],
            }
            for i, h in enumerate(hits, start=1)
        ],
    }

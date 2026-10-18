# RAG Modul Kuliah

Chatbot tanya-jawab berbasis PDF modul kuliah. Jawaban hanya diambil dari dokumen yang diupload, lengkap dengan sitasi halaman.

## Alur

```
UPLOAD:  PDF → ekstrak teks per halaman → chunk (800 char, overlap 150) → embedding → ChromaDB
TANYA:   pertanyaan → embedding → cari top-5 chunk paling mirip → prompt (konteks + pertanyaan) → LLM → jawaban + sitasi
```

| File                    | Isi                                                   |
| ----------------------- | ----------------------------------------------------- |
| `app/ingest.py`         | Ekstraksi PDF (pypdf) dan chunking                    |
| `app/store.py`          | Embedding (`multilingual-e5-small`) + ChromaDB        |
| `app/llm.py`            | System prompt, susun konteks, panggil Ollama / Claude |
| `app/main.py`           | Endpoint FastAPI                                      |
| `app/static/index.html` | UI sederhana                                          |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

**Pilih LLM:**

- **Ollama (lokal, gratis):** install dari ollama.com, lalu jalankan `ollama pull qwen2.5:3b`. Kalau RAM pas-pasan, pakai `qwen2.5:1.5b`.
- **Claude API:** set `LLM_PROVIDER=anthropic` dan `ANTHROPIC_API_KEY` di `.env`.

## Jalankan

```bash
uvicorn app.main:app --reload
```

Buka http://localhost:8000, upload PDF, lalu tanya. Dokumentasi API otomatis ada di http://localhost:8000/docs.

Run pertama akan download model embedding (~120 MB). Setelah itu bisa jalan offline (kalau pakai Ollama).

## API

| Method | Path                | Body                              |
| ------ | ------------------- | --------------------------------- |
| POST   | `/documents`        | form-data `file` (PDF)            |
| GET    | `/documents`        | –                                 |
| DELETE | `/documents/{name}` | –                                 |
| POST   | `/ask`              | `{"question": "...", "top_k": 5}` |

## Eksperimen lanjutan

1. **Ubah `CHUNK_SIZE` / `TOP_K`.** Bandingkan kualitas jawaban dengan chunk 400 vs 1200. Chunk terlalu kecil membuat konteks terpotong; terlalu besar membuat retrieval kurang presisi.
2. **Bikin eval set.** Siapkan 20 pertanyaan + jawaban benar + halaman sumber, lalu ukur apakah halaman yang benar masuk top-k (_hit rate_). Ini cara objektif membandingkan perubahan.
3. **Hybrid search.** Gabungkan vector search dengan BM25 (`rank_bm25`). Membantu untuk istilah teknis/kode yang embedding kurang tangkap.
4. **Reranker.** Ambil top-20, lalu rerank pakai cross-encoder (`BAAI/bge-reranker-v2-m3`) jadi top-5.
5. **Streaming jawaban.** Pakai `"stream": true` di Ollama + `StreamingResponse` di FastAPI.
6. **Riwayat chat.** Kirim beberapa pertanyaan sebelumnya supaya follow-up seperti "jelaskan lebih detail" bisa dipahami.
7. **OCR.** PDF hasil scan saat ini ditolak. Tambahkan `ocrmypdf` sebagai fallback.

## Batasan

- PDF hasil scan (gambar) tidak bisa diekstrak tanpa OCR.
- Tabel dan rumus sering berantakan saat diekstrak pypdf.
- Model 1.5–3B kadang tetap mengarang. Cek sitasinya.

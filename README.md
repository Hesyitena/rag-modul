# RAG Modul Kuliah

Chatbot tanya-jawab berbasis PDF modul kuliah. Jawaban hanya diambil dari dokumen yang diupload, lengkap dengan sitasi halaman. Bisa jalan sepenuhnya offline di laptop (pakai Ollama).

Baru kenal RAG, chunk, atau embedding? Baca [`docs/konsep-rag.md`](docs/konsep-rag.md) dulu.

## Fitur

- Upload PDF (klik atau seret), lalu tanya isinya. Jawaban memuat nomor sitasi; klik nomornya untuk melihat kutipan dan halaman sumber.
- **Saran pertanyaan** otomatis setelah upload, dibuat LLM dari isi PDF. Bisa dibuka lagi kapan saja lewat tombol ⓘ di samping dokumen, dan dibuat ulang lewat "↻ buat ulang".
- **Panel Indeks** di sidebar: jumlah chunk/vektor, dimensi vektor, model embedding, ukuran chunk, Top-K, dan LLM yang dipakai.
- Pilihan LLM: Ollama (lokal, gratis) atau Claude API.

## Alur

```
UPLOAD:  PDF → ekstrak teks per halaman → chunk (800 char, overlap 150) → embedding → ChromaDB
TANYA:   pertanyaan → embedding → cari top-5 chunk paling mirip → prompt (konteks + pertanyaan) → LLM → jawaban + sitasi
```

| File                    | Isi                                                                     |
| ----------------------- | ----------------------------------------------------------------------- |
| `app/config.py`         | Membaca pengaturan dari `.env`                                          |
| `app/ingest.py`         | Ekstraksi PDF (pypdf) dan chunking                                      |
| `app/store.py`          | Embedding (`multilingual-e5-small`) + ChromaDB                          |
| `app/llm.py`            | System prompt, susun konteks, panggil Ollama / Claude, saran pertanyaan |
| `app/main.py`           | Endpoint FastAPI + cache saran pertanyaan                               |
| `app/static/index.html` | Antarmuka web (satu file, tanpa build)                                  |
| `docs/konsep-rag.md`    | Penjelasan konsep RAG, chunk, embedding, ChromaDB                       |

## Prasyarat

- **Python 3.10 atau lebih baru** (`python --version`).
- **Salah satu LLM:**
  - **Ollama (lokal, gratis):** install dari [ollama.com](https://ollama.com), lalu `ollama pull qwen2.5:3b`. Kalau RAM pas-pasan, pakai `qwen2.5:1.5b` dan ubah `OLLAMA_MODEL` di `.env`.
  - **Claude API:** butuh API key dari [console.anthropic.com](https://console.anthropic.com).
- Ruang disk ± 3 GB untuk dependensi (PyTorch cukup besar) dan model embedding.

## Setup

**Windows (PowerShell):**

```powershell
git clone https://github.com/Hesyitena/rag-modul.git
cd rag-modul
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Kalau `Activate.ps1` ditolak dengan pesan *running scripts is disabled*, jalankan sekali `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, lalu aktifkan lagi.

**Linux / macOS:**

```bash
git clone https://github.com/Hesyitena/rag-modul.git
cd rag-modul
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Pastikan prompt terminal menampilkan `(.venv)` sebelum `pip install`, supaya paket tidak masuk ke Python global.

**Kalau pakai Claude API:** di `.env`, set `LLM_PROVIDER=anthropic` dan isi `ANTHROPIC_API_KEY`. Jangan commit file `.env` (sudah ada di `.gitignore`).

## Jalankan

```bash
uvicorn app.main:app --reload
```

Buka http://localhost:8000, upload PDF, lalu tanya. Dokumentasi API interaktif ada di http://localhost:8000/docs.

Run pertama akan mengunduh model embedding (~120 MB), jadi startup agak lama. Setelah itu bisa jalan offline (kalau pakai Ollama).

## Konfigurasi

Semua pengaturan ada di `.env`. Penjelasan tiap variabel ada di [`docs/konsep-rag.md` bagian 10](docs/konsep-rag.md#10-konfigurasi-env).

> **Penting:** setelah mengubah `CHUNK_SIZE`, `CHUNK_OVERLAP`, atau `EMBED_MODEL`, **upload ulang semua PDF**. Chunk dan vektor yang sudah tersimpan dibuat dengan pengaturan lama dan tidak ikut berubah. Restart server juga diperlukan setiap kali `.env` diubah.

## Lokasi data

Semua data tersimpan lokal di folder `data/` (tidak ikut ke git):

| Path                    | Isi                                     |
| ----------------------- | --------------------------------------- |
| `data/uploads/`         | Salinan PDF yang diupload               |
| `data/chroma/`          | Database vektor ChromaDB                |
| `data/suggestions.json` | Cache saran pertanyaan per dokumen      |

Untuk mulai dari nol: matikan server, hapus folder `data/`, lalu jalankan lagi.

## API

| Method | Path                            | Keterangan                                                                      |
| ------ | ------------------------------- | ------------------------------------------------------------------------------- |
| POST   | `/documents`                    | Upload PDF (form-data `file`). Nama sama = menimpa dokumen lama                 |
| GET    | `/documents`                    | Daftar dokumen dan jumlah chunk                                                 |
| DELETE | `/documents/{name}`             | Hapus dokumen dari ChromaDB, disk, dan cache saran                              |
| POST   | `/ask`                          | Body `{"question": "...", "top_k": 5}`. Mengembalikan jawaban + sumber          |
| GET    | `/documents/{name}/suggestions` | 3 saran pertanyaan. Dari cache kalau ada; `?refresh=true` memaksa LLM membuat ulang |
| GET    | `/suggestions`                  | Semua saran yang tersimpan, tanpa memanggil LLM                                 |
| GET    | `/stats`                        | Jumlah chunk/vektor, dimensi, model, dan konfigurasi retrieval                  |

## Troubleshooting

| Gejala                                                          | Penyebab dan solusi                                                                                   |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `Gagal memanggil LLM: ... No connection could be made` / `Connection refused` | Ollama belum jalan. Buka aplikasi Ollama atau jalankan `ollama serve`                   |
| `Gagal memanggil LLM: Client error '404 Not Found'`             | Model belum di-pull atau salah nama. Cek `ollama list`, lalu `ollama pull <OLLAMA_MODEL>`             |
| `Tidak ada teks yang bisa diekstrak` (422)                      | PDF hasil scan (gambar). Perlu OCR dulu, misalnya dengan `ocrmypdf`                                   |
| Startup pertama lama / terlihat macet                           | Sedang mengunduh model embedding. Tunggu sampai selesai                                               |
| Peringatan `huggingface_hub ... symlinks` atau `HF_TOKEN`       | Aman diabaikan. Model tetap tersimpan, hanya memakai sedikit lebih banyak ruang disk di Windows       |
| Ubah `CHUNK_SIZE`/`TOP_K` tapi hasil tidak berubah              | Restart server; untuk `CHUNK_SIZE`/`CHUNK_OVERLAP`, upload ulang PDF juga                             |
| Jawaban mengarang / tidak sesuai modul                          | Model kecil kurang patuh. Cek sitasi, coba model lebih besar, atau naikkan `TOP_K`                    |
| `ModuleNotFoundError` saat `uvicorn`                            | venv belum aktif. Aktifkan dulu, lalu jalankan dari folder `rag-modul`                                |

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
- Model 1.5–3B kadang tetap mengarang, termasuk di saran pertanyaan. Cek sitasinya.
- Tiap pertanyaan berdiri sendiri; belum ada riwayat percakapan.

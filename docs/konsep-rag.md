# Konsep RAG: Chunk, Embedding, dan ChromaDB

Dokumen ini menjelaskan bagaimana aplikasi ini menjawab pertanyaan dari PDF modul kuliah, dengan merujuk langsung ke kode di `app/`.

## Daftar isi

1. [Masalah yang diselesaikan](#1-masalah-yang-diselesaikan)
2. [Gambaran besar](#2-gambaran-besar)
3. [Ekstraksi teks PDF](#3-ekstraksi-teks-pdf)
4. [Chunking](#4-chunking)
5. [Embedding](#5-embedding)
6. [ChromaDB](#6-chromadb)
7. [Pencarian](#7-pencarian)
8. [Generasi jawaban oleh LLM](#8-generasi-jawaban-oleh-llm)
9. [Contoh nyata](#9-contoh-nyata)
10. [Konfigurasi `.env`](#10-konfigurasi-env)
11. [Cara menjalankan](#11-cara-menjalankan)
12. [Keterbatasan dan cara memperbaikinya](#12-keterbatasan-dan-cara-memperbaikinya)
13. [Glosarium](#13-glosarium)

---

## 1. Masalah yang diselesaikan

Kamu punya PDF modul, misalnya 200 halaman, dan ingin bertanya "apa itu normalisasi database?". Ada dua kendala kalau seluruh PDF langsung diberikan ke LLM:

- LLM lokal kecil (`qwen2.5:3b`) punya jendela konteks terbatas dan tidak muat membaca 200 halaman.
- Meski muat, jawabannya lambat dan model mudah kehilangan fokus.

Solusinya adalah **RAG (Retrieval-Augmented Generation)**:

1. **Retrieval**: cari hanya bagian PDF yang relevan dengan pertanyaan.
2. **Augmented**: tempelkan bagian itu ke prompt sebagai konteks.
3. **Generation**: LLM menyusun jawaban dari konteks tersebut.

Pertanyaan kuncinya: bagaimana komputer tahu bagian mana yang relevan? Jawabannya adalah chunking, embedding, dan vector database.

## 2. Gambaran besar

**Saat upload PDF (sekali per file):**

```
PDF
 └─ ekstrak teks per halaman         (app/ingest.py: extract_pages)
     └─ potong jadi chunk            (app/ingest.py: chunk_text)
         └─ ubah tiap chunk jadi vektor   (app/store.py: _embed)
             └─ simpan di ChromaDB        (app/store.py: add_document)
```

**Saat bertanya:**

```
pertanyaan
 └─ ubah jadi vektor                 (app/store.py: _embed)
     └─ cari TOP_K chunk terdekat di ChromaDB   (app/store.py: search)
         └─ susun prompt: konteks + pertanyaan  (app/llm.py: build_prompt)
             └─ LLM menjawab                    (app/llm.py: generate)
                 └─ jawaban + sitasi halaman
```

Pembagian tugas:

| Komponen | Tugas | Analogi |
|---|---|---|
| `pypdf` | Mengambil teks dari PDF | Menyalin isi buku ke catatan |
| Chunking | Memotong teks jadi bagian kecil | Menandai paragraf |
| Model embedding (`multilingual-e5-small`) | Mengubah teks jadi angka yang mewakili makna | Pustakawan yang paham isi tiap paragraf |
| ChromaDB | Menyimpan vektor dan mencari yang terdekat | Rak indeks perpustakaan |
| LLM (Ollama / Claude) | Menyusun jawaban dari konteks | Mahasiswa yang menulis jawaban dari kutipan yang diberikan |

Penting: **model embedding dan LLM adalah dua model berbeda.** Model embedding hanya mengubah teks jadi angka dan tidak bisa menjawab. LLM hanya menjawab dari konteks yang diberikan dan tidak ikut mencari.

## 3. Ekstraksi teks PDF

`extract_pages` di `app/ingest.py` membaca PDF dengan `pypdf` dan mengembalikan daftar `(nomor_halaman, teks)`. Dua hal yang dilakukan:

- Spasi/tab berlebih dirapikan dan baris kosong beruntun dikurangi.
- Halaman tanpa teks dilewati. PDF hasil scan (gambar) tidak punya teks, sehingga seluruh halamannya terlewati. Kalau tidak ada teks sama sekali, upload ditolak dengan HTTP 422.

Nomor halaman dipertahankan dari sini sampai akhir supaya jawaban bisa menyebut sumbernya.

## 4. Chunking

### Apa itu chunk

Chunk adalah potongan kecil dari teks, sekitar satu paragraf. Teks 200 halaman dipotong jadi ratusan chunk, masing-masing maksimal `CHUNK_SIZE` karakter (default 800).

### Kenapa harus dipotong

- **Presisi pencarian.** Kalau satu unit adalah satu halaman penuh yang membahas banyak topik, vektornya jadi "rata-rata" semua topik dan tidak tajam. Unit yang lebih kecil punya makna lebih fokus.
- **Konteks hemat.** Hanya potongan relevan yang dikirim ke LLM, bukan seluruh dokumen.

### Overlap

`CHUNK_OVERLAP` (default 150) adalah jumlah karakter di akhir sebuah chunk yang diulang di awal chunk berikutnya:

```
chunk 0: [.............................. ujung kalimat penting]
chunk 1:                  [ujung kalimat penting ..............................]
                           ^^^^^^^^^^^^^^^^^^^^^ overlap
```

Tanpa overlap, kalimat yang jatuh tepat di batas potongan terbelah dan maknanya hilang dari kedua chunk. Dengan overlap, kalimat itu utuh di salah satunya.

### Cara kerja `chunk_text`

1. Ambil `CHUNK_SIZE` karakter dari posisi `start`.
2. Kalau belum akhir teks, geser batas mundur ke baris baru atau spasi terdekat (asalkan masih di paruh kedua chunk), supaya kata tidak terpotong.
3. Chunk berikutnya mulai dari `end - overlap`.
4. Ada validasi: `overlap` harus lebih kecil dari `size`.

Chunk dibuat **per halaman** (`build_chunks`), jadi satu chunk tidak pernah melintasi dua halaman. Tiap chunk membawa metadata `{"source": nama_file, "page": nomor_halaman}`.

### Trade-off ukuran chunk

| Chunk terlalu kecil | Chunk terlalu besar |
|---|---|
| Konteks terpotong, jawaban kurang lengkap | Vektor kabur, retrieval kurang presisi |
| Banyak chunk, pencarian lebih berat | Prompt panjang, LLM lebih lambat |

Tidak ada angka ajaib. Bandingkan 400 vs 800 vs 1200 dengan pertanyaan nyata (lihat bagian 12).

## 5. Embedding

### Apa itu embedding

Embedding mengubah teks menjadi **daftar angka (vektor)** yang mewakili *makna*-nya. Teks yang maknanya mirip menghasilkan vektor yang berdekatan; teks yang berbeda topik menghasilkan vektor yang berjauhan.

```
"Kucing makan ikan"    → [0.12, -0.80, 0.33, ...]   ← berdekatan
"Kucing sedang makan"  → [0.10, -0.78, 0.30, ...]   ←
"Harga saham naik"     → [-0.55, 0.20, 0.91, ...]   ← jauh
```

Ini berbeda dari pencarian kata kunci. "mobil" dan "kendaraan" dianggap dekat walau katanya berbeda.

### Model yang dipakai

`intfloat/multilingual-e5-small`:

- Menghasilkan vektor **384 dimensi** (384 angka per teks).
- Multibahasa, cocok untuk modul berbahasa Indonesia.
- Ukuran ~120 MB, diunduh sekali lalu berjalan offline.
- Dijalankan lokal lewat `sentence-transformers`, terpisah dari Ollama.

### Prefix `query:` dan `passage:`

Model keluarga E5 dilatih dengan awalan khusus:

- Teks dokumen (chunk) diberi `passage: `.
- Pertanyaan diberi `query: `.

Ini diatur di `config.py` (`PASSAGE_PREFIX`, `QUERY_PREFIX`). Kalau kamu mengganti model non-E5, kosongkan kedua prefix itu lewat `.env`.

### Normalisasi dan cosine similarity

Vektor dibuat dengan `normalize_embeddings=True`, artinya panjangnya disamakan jadi 1. Kemiripan diukur dengan **cosine similarity**, yaitu seberapa searah dua vektor:

- Mendekati 1: sangat mirip
- Mendekati 0: tidak berhubungan

ChromaDB dikonfigurasi `{"hnsw:space": "cosine"}` dan mengembalikan *jarak* (`dist`). Aplikasi mengubahnya jadi skor dengan `score = 1 - dist`.

## 6. ChromaDB

ChromaDB adalah **vector database**: tempat menyimpan vektor dan mencari yang terdekat dengan cepat tanpa membandingkan satu per satu. Di `app/store.py`, tiap chunk disimpan sebagai:

| Field | Isi |
|---|---|
| `id` | `"{nama_file}::{nomor_urut}"` |
| `documents` | Teks asli chunk |
| `embeddings` | Vektor 384 angka |
| `metadatas` | `{"source": ..., "page": ...}` |

Hal penting lain:

- **Persisten.** Dipakai lewat `PersistentClient(path=CHROMA_DIR)` (default `./data/chroma`). Setelah server restart, dokumen yang sudah di-upload tetap ada.
- **Upload ulang = ganti.** `add_document` memanggil `delete_document` dulu, sehingga file dengan nama sama menimpa isi lama.
- **Batch.** Chunk dimasukkan per 256 supaya aman untuk dokumen besar.
- **Hapus per dokumen.** `delete_document` memakai filter `where={"source": nama}`.
- Ubah `CHUNK_SIZE`/`CHUNK_OVERLAP` tidak mengubah data lama. **Upload ulang PDF** setelah mengubahnya.

## 7. Pencarian

`VectorStore.search(query, k)`:

1. Kalau koleksi kosong, kembalikan daftar kosong (API membalas "Belum ada dokumen").
2. Ubah pertanyaan jadi vektor (dengan prefix `query: `).
3. Minta ChromaDB `k` chunk terdekat (maksimal sejumlah chunk yang ada).
4. Kembalikan teks, sumber, halaman, dan skor tiap chunk.

`k` diambil dari `TOP_K` (default 5), atau dari `top_k` di body `/ask` (1 sampai 20).

**Catatan penting:** ChromaDB selalu mengembalikan `k` hasil terdekat, **walau tidak ada yang benar-benar relevan**. Lihat contoh "resep nasi goreng" di bagian 9.

## 8. Generasi jawaban oleh LLM

`app/llm.py` menyusun prompt dari chunk hasil pencarian:

```
KONTEKS:
[1] (modul.pdf, hlm 12)
<teks chunk 1>

[2] (modul.pdf, hlm 13)
<teks chunk 2>
...

PERTANYAAN:
<pertanyaan user>
```

Dikirim bersama **system prompt** yang mewajibkan LLM:

- Menjawab **hanya** dari konteks, tanpa pengetahuan luar.
- Kalau jawabannya tidak ada, menjawab `Tidak ditemukan di modul yang diunggah.`
- Memberi sitasi nomor konteks (`[1]`, `[2][3]`) untuk tiap klaim.
- Menjawab dalam bahasa yang sama dengan pertanyaan, singkat dan jelas.

`temperature` diset 0.2 supaya jawaban stabil dan tidak terlalu "kreatif". Provider dipilih lewat `LLM_PROVIDER`: `ollama` (lokal, `/api/chat`) atau `anthropic` (Claude API). Nomor `[n]` di jawaban dipetakan kembali ke nama file dan halaman, itulah sitasi yang tampil di UI.

## 9. Contoh nyata

Berikut hasil menjalankan `chunk_text` dan model embedding asli pada teks pendek tentang database. Ukuran chunk dikecilkan jadi 200 karakter dengan overlap 40 supaya kelihatan. Penyimpanan memakai ChromaDB di memori, bukan data aplikasi.

### Chunking

Teks terpotong jadi 5 chunk. Perhatikan kalimat yang diulang di batas:

```
[chunk 0] 'Normalisasi adalah ... Bentuk normal kedua (2NF) menghilangkan'
[chunk 1] 'Bentuk normal kedua (2NF) menghilangkan ketergantungan parsial ... Indeks pada database'
[chunk 2] 'tar kolom non-key.\n\nIndeks pada database dipakai untuk mempercepat ...'
[chunk 3] 'memperlambat operasi INSERT dan UPDATE.\n\nTransaksi adalah ...'
[chunk 4] 'Isolation, dan Durability menjamin data tetap benar meski terjadi kegagalan.'
```

### Embedding

Tiap chunk menjadi 384 angka, misalnya chunk 0 diawali `[0.063, -0.046, -0.073, -0.092, 0.068, ...]`. Angka satuannya tidak bermakna bagi manusia; yang dipakai adalah jarak antar vektor.

### Pencarian

| Pertanyaan | Chunk teratas | Skor | Relevan? |
|---|---|---|---|
| apa itu 3NF? | chunk 1 (berisi 3NF) | 0.868 | Ya |
| kenapa query jadi lambat? | chunk 2 (tentang indeks) | 0.813 | Ya |
| bagaimana jaga data tetap benar saat error? | chunk 4 (ACID, kegagalan) | 0.868 | Ya |
| resep nasi goreng | chunk 4 | 0.797 | **Tidak** |

Pelajaran dari tabel ini:

- **Pencarian berdasarkan makna.** "kenapa query jadi lambat" tidak memuat kata "indeks", tapi chunk indeks yang ditemukan.
- **Skor mentah tidak menandakan relevan atau tidak.** Pertanyaan tidak nyambung ("resep nasi goreng") tetap dapat 0.797, hampir sama dengan pertanyaan yang benar (0.813). Pada model E5 skor cosine cenderung menumpuk di rentang tinggi. Jangan memakai ambang tetap seperti "skor di bawah 0.8 berarti tidak relevan" tanpa mengujinya dulu.
- **Penjaga ada di system prompt**, bukan di retrieval. Model kecil kadang tetap mengarang, jadi cek sitasinya.

## 10. Konfigurasi `.env`

| Variabel | Default | Arti |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` (lokal) atau `anthropic` (Claude API) |
| `OLLAMA_URL` | `http://localhost:11434` | Alamat server Ollama |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model Ollama; harus sudah di-`pull`. Kalau RAM pas-pasan, coba `qwen2.5:1.5b` |
| `ANTHROPIC_API_KEY` | (kosong) | API key Claude; hanya dibutuhkan kalau `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Model Claude; diabaikan kalau pakai Ollama |
| `CHUNK_SIZE` | `800` | Karakter per chunk |
| `CHUNK_OVERLAP` | `150` | Karakter yang diulang antar chunk |
| `TOP_K` | `5` | Jumlah chunk yang dikirim ke LLM |

Variabel lain yang bisa di-override (opsional, lihat `app/config.py`): `EMBED_MODEL`, `QUERY_PREFIX`, `PASSAGE_PREFIX`, `CHROMA_DIR`, `UPLOAD_DIR`.

Jangan commit file `.env` yang berisi API key.

## 11. Cara menjalankan

Prasyarat: Python 3.10+ dan Ollama dengan model sudah ter-pull (`ollama pull qwen2.5:3b`).

```powershell
cd E:\miniproject\rag-modul
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Buka http://localhost:8000, upload PDF, lalu tanya. Dokumentasi API otomatis ada di http://localhost:8000/docs.

Pada run pertama, model embedding diunduh (~120 MB) sehingga startup lebih lama. Pastikan prompt terminal menampilkan `(.venv)` sebelum `pip install` supaya paket tidak masuk ke Python global.

Endpoint:

| Method | Path | Fungsi |
|---|---|---|
| POST | `/documents` | Upload PDF (form-data `file`) |
| GET | `/documents` | Daftar dokumen dan jumlah chunk |
| DELETE | `/documents/{name}` | Hapus dokumen dari ChromaDB dan disk |
| POST | `/ask` | `{"question": "...", "top_k": 5}` |
| GET | `/documents/{name}/suggestions` | 3 saran pertanyaan (cache; `?refresh=true` untuk membuat ulang) |
| GET | `/suggestions` | Semua saran yang tersimpan |
| GET | `/stats` | Jumlah chunk/vektor, dimensi, model, konfigurasi |

## 12. Keterbatasan dan cara memperbaikinya

| Masalah | Penyebab | Perbaikan |
|---|---|---|
| PDF scan ditolak | Tidak ada teks untuk diekstrak | Tambah OCR (`ocrmypdf`) sebagai fallback |
| Tabel dan rumus berantakan | `pypdf` hanya mengambil teks mentah | Ekstraktor lain atau parsing khusus tabel |
| Chunk yang salah ikut terambil | Embedding kurang tajam untuk istilah teknis atau kode | Hybrid search: gabungkan vector search dengan BM25 (`rank_bm25`) |
| Urutan hasil kurang tepat | Pencarian vektor hanya perkiraan makna | Reranker: ambil top-20, urutkan ulang dengan cross-encoder (`BAAI/bge-reranker-v2-m3`) jadi top-5 |
| Model mengarang | Model 1.5B–3B kurang patuh pada instruksi | Pakai model lebih besar atau Claude API; selalu cek sitasi |
| Follow-up ("jelaskan lebih detail") tidak dipahami | Tiap pertanyaan berdiri sendiri | Kirim riwayat chat ke prompt dan pencarian |

### Cara mengukur secara objektif

Menebak-nebak "rasanya lebih bagus" tidak bisa diandalkan. Buat **eval set**:

1. Siapkan ±20 pertanyaan beserta halaman yang berisi jawabannya.
2. Untuk tiap pertanyaan, jalankan `store.search` dan cek apakah halaman yang benar muncul di `TOP_K`.
3. Hitung **hit rate** = pertanyaan yang halaman benarnya masuk top-k, dibagi total pertanyaan.
4. Ubah `CHUNK_SIZE`, `CHUNK_OVERLAP`, atau `TOP_K`, upload ulang PDF, dan bandingkan hit rate-nya.

Dengan cara ini setiap perubahan (hybrid search, reranker, ukuran chunk) bisa dinilai dengan angka.

## 13. Glosarium

| Istilah | Arti |
|---|---|
| RAG | Retrieval-Augmented Generation: cari dokumen relevan dulu, baru minta LLM menjawab dari situ |
| Chunk | Potongan kecil teks (default ≤800 karakter) |
| Overlap | Bagian chunk yang diulang di chunk berikutnya agar kalimat di batas tidak hilang |
| Embedding / vektor | Daftar angka (di sini 384) yang mewakili makna sebuah teks |
| Cosine similarity | Ukuran kemiripan dua vektor berdasarkan arahnya; 1 = sangat mirip |
| Vector database | Penyimpanan vektor dengan pencarian "terdekat" yang cepat (ChromaDB) |
| Top-K | Jumlah chunk terdekat yang diambil untuk konteks |
| Prompt / system prompt | Teks yang dikirim ke LLM; system prompt berisi aturan perilakunya |
| Sitasi | Rujukan sumber (nama file dan halaman) pada tiap bagian jawaban |
| Hit rate | Persentase pertanyaan yang chunk benarnya muncul di top-K |
| Hybrid search | Gabungan pencarian vektor (makna) dan kata kunci (BM25) |
| Reranker | Model kedua yang mengurutkan ulang hasil pencarian agar lebih akurat |

"""Pemanggilan LLM: Ollama (lokal) atau Claude API."""
import httpx

from .config import ANTHROPIC_MODEL, LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL

SYSTEM_PROMPT = """Kamu asisten belajar untuk mahasiswa.
Aturan:
- Jawab HANYA berdasarkan KONTEKS yang diberikan. Jangan pakai pengetahuan di luar konteks.
- Kalau jawabannya tidak ada di konteks, katakan: "Tidak ditemukan di modul yang diunggah."
- Setiap klaim beri sitasi nomor konteks, contoh: [1] atau [2][3].
- Jawab dalam bahasa yang sama dengan pertanyaan, singkat dan jelas."""


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] ({h['source']}, hlm {h['page']})\n{h['text']}")
    return "\n\n".join(parts)


def build_prompt(question: str, hits: list[dict]) -> str:
    return f"KONTEKS:\n{build_context(hits)}\n\nPERTANYAAN:\n{question}"


def _ollama(prompt: str) -> str:
    r = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _anthropic(prompt: str) -> str:
    import anthropic  # import di sini supaya tidak wajib kalau pakai Ollama

    client = anthropic.Anthropic()  # baca ANTHROPIC_API_KEY dari environment
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def generate(question: str, hits: list[dict]) -> str:
    prompt = build_prompt(question, hits)
    if LLM_PROVIDER == "anthropic":
        return _anthropic(prompt)
    return _ollama(prompt)

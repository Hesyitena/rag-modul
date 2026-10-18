"""Pemanggilan LLM: Ollama (lokal) atau Claude API."""
import re

import httpx

from .config import ANTHROPIC_MODEL, LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL

SYSTEM_PROMPT = """Kamu asisten belajar untuk mahasiswa.
Aturan:
- Jawab HANYA berdasarkan KONTEKS yang diberikan. Jangan pakai pengetahuan di luar konteks.
- Kalau jawabannya tidak ada di konteks, katakan: "Tidak ditemukan di modul yang diunggah."
- Setiap klaim beri sitasi nomor konteks, contoh: [1] atau [2][3].
- Jawab dalam bahasa yang sama dengan pertanyaan, singkat dan jelas."""

SUGGEST_PROMPT = """Kamu membantu mahasiswa belajar dari modul kuliah.
Dari CUPLIKAN modul yang diberikan, tulis 3 pertanyaan pendek yang jawabannya ada di cuplikan itu.
Satu pertanyaan per baris. Tanpa nomor, tanpa kalimat pembuka atau penjelasan.
Gunakan bahasa yang sama dengan cuplikan."""


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] ({h['source']}, hlm {h['page']})\n{h['text']}")
    return "\n\n".join(parts)


def build_prompt(question: str, hits: list[dict]) -> str:
    return f"KONTEKS:\n{build_context(hits)}\n\nPERTANYAAN:\n{question}"


def _ollama(prompt: str, system: str) -> str:
    r = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _anthropic(prompt: str, system: str) -> str:
    import anthropic  # import di sini supaya tidak wajib kalau pakai Ollama

    client = anthropic.Anthropic()  # baca ANTHROPIC_API_KEY dari environment
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def _call(prompt: str, system: str) -> str:
    if LLM_PROVIDER == "anthropic":
        return _anthropic(prompt, system)
    return _ollama(prompt, system)


def generate(question: str, hits: list[dict]) -> str:
    return _call(build_prompt(question, hits), SYSTEM_PROMPT)


def parse_questions(raw: str) -> list[str]:
    """Ambil maks 3 pertanyaan dari output LLM; buang nomor/bullet dan baris yang bukan pertanyaan."""
    qs = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in raw.splitlines()]
    return [q for q in qs if q.endswith("?") and len(q) > 8][:3]


def suggest(passages: list[str]) -> list[str]:
    prompt = "CUPLIKAN:\n" + "\n\n---\n\n".join(p[:600] for p in passages)
    return parse_questions(_call(prompt, SUGGEST_PROMPT))


if __name__ == "__main__":
    assert parse_questions("Berikut pertanyaannya:\n1. Apa itu 2NF?\n- Apa itu ACID?\n\n3) Kenapa indeks mempercepat query?\nApa lagi?") == [
        "Apa itu 2NF?", "Apa itu ACID?", "Kenapa indeks mempercepat query?"]
    print("ok")

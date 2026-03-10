"""Constants and configuration for the RAG module."""

from pathlib import Path

# ── Model options ────────────────────────────────────────────────────────

EMBEDDING_MODELS = {
    # ── Ollama (requires `ollama pull <model>`) ──
    "Qwen3 Embedding 0.6B · MTEB #1 (recommended)": "ollama:qwen3-embedding:0.6b",
    "BGE-M3 567M · proven multilingual": "ollama:bge-m3",
    "EmbeddingGemma 300M · lightweight": "ollama:embeddinggemma",
    "Nomic Embed v2 MoE · fast multilingual": "ollama:nomic-embed-text-v2-moe",
    # ── HuggingFace (no Ollama needed, downloaded once) ──
    "multilingual-e5-small · no Ollama needed": "intfloat/multilingual-e5-small",
    "multilingual-e5-base · higher quality": "intfloat/multilingual-e5-base",
}

LLM_MODELS = {
    "Qwen 3.5 9B · latest, 256K ctx (recommended)": "qwen3.5:9b",
    "Qwen 3.5 4B · lightweight, 256K ctx": "qwen3.5:4b",
    "Qwen 3 8B · thinking mode": "qwen3:8b",
    "Qwen 3 4B · tiny, 256K ctx": "qwen3:4b",
    "Gemma 3 12B · 128K ctx": "gemma3:12b",
    "Gemma 3 4B · lightweight": "gemma3:4b",
    "Gemma 3 1B · ultra-light": "gemma3:1b",
}

# ── Retrieval parameters ─────────────────────────────────────────────────

RETRIEVAL_K = 20  # fetch more candidates for MMR
RETRIEVAL_FINAL_K = 10  # return after diversity filtering
MMR_LAMBDA = 0.7  # balance relevance (1.0) vs diversity (0.0)

# ── Chunking parameters ─────────────────────────────────────────────────

GAP_MINUTES = 150  # silence gap that splits conversations
MAX_CHUNK_TOKENS = 400
MAX_CHUNK_MESSAGES = 50
OVERLAP_MESSAGES = 5  # message-level overlap for context continuity

# ── Persistence ──────────────────────────────────────────────────────────

PERSIST_DIR = Path(".tglens_index")
META_FILE = PERSIST_DIR / "meta.json"

# ── System prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a friendly helpful assistant that answers questions about a user's Telegram inbox. You are allowed to be playful and answer with humor.

You will receive excerpts from real chat conversations retrieved by semantic \
search. Use ONLY the provided excerpts to answer. If the excerpts do not \
contain enough information, say so honestly — do not invent messages or facts.

Rules:
- Be concise. Cite chat names, dates, and people when relevant.
- The user's name is "{user_name}". Refer to them as "you" when describing their messages.
- Answer in the same language the user asks their question in.

Give detailed answers and use your deductive reasoning skills to connect the dots between different messages and chats. The more insights you can provide, the better!"""

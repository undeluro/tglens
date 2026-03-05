"""
RAG (Retrieval-Augmented Generation) module for tglens.
Enables querying chat histories using natural language.
Runs fully locally: sentence-transformers for embeddings, Ollama for LLM.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

if TYPE_CHECKING:
    from collections.abc import Generator

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

RETRIEVAL_K = 12  # fetch more candidates for MMR
RETRIEVAL_FINAL_K = 6  # return after diversity filtering
MMR_LAMBDA = 0.7  # balance relevance (1.0) vs diversity (0.0)

# ── Chunking parameters ─────────────────────────────────────────────────

GAP_MINUTES = 120  # silence gap that splits conversations
MAX_CHUNK_TOKENS = 400
MAX_CHUNK_MESSAGES = 50
OVERLAP_MESSAGES = 3  # message-level overlap for context continuity

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


# ═══════════════════════════════════════════════════════════════════════════
# Embeddings helpers
# ═══════════════════════════════════════════════════════════════════════════


class _E5Embeddings(HuggingFaceEmbeddings):
    """Wrapper that adds E5 query/passage prefixes for better retrieval."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


@st.cache_resource
def _get_embeddings(model_id: str):
    """Return an embeddings instance for the given model identifier."""
    if model_id.startswith("ollama:"):
        ollama_model = model_id[len("ollama:") :]
        return OllamaEmbeddings(model=ollama_model)

    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if "e5" in model_id.lower():
        return _E5Embeddings(
            model_name=model_id,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Chunking
# ═══════════════════════════════════════════════════════════════════════════


def _estimate_tokens(text: str) -> int:
    """Estimate token count. Russian text tokenizes ~1.8x word count,
    English ~1.3x. Use a blended estimate of 1.5x."""
    return max(1, int(len(text.split()) * 1.5))


def _format_message(row: pd.Series) -> str:
    """Format a single message row into a readable line."""
    ts = pd.to_datetime(row["datetime"]).strftime("%Y-%m-%d %H:%M")
    sender = row.get("from") or row.get("actor") or "Unknown"
    text = str(row.get("text", ""))
    return f"[{ts}] {sender}: {text}"


def _split_by_time_gap(
    group: pd.DataFrame, gap_minutes: int = GAP_MINUTES
) -> list[pd.DataFrame]:
    """Split a chat group into conversation segments by time gaps."""
    if group.empty:
        return []
    group = group.sort_values("datetime").reset_index(drop=True)
    dt = pd.to_datetime(group["datetime"])
    gaps = dt.diff() > pd.Timedelta(minutes=gap_minutes)
    segment_ids = gaps.cumsum()
    return [seg for _, seg in group.groupby(segment_ids) if len(seg) > 0]


def _segment_to_chunks(
    segment: pd.DataFrame,
    chat_name: str,
) -> list[Document]:
    """Convert a conversation segment into token-limited Document chunks
    with message-level overlap for context continuity."""
    messages = []
    for _, row in segment.iterrows():
        text = str(row.get("text", ""))
        if not text.strip():
            continue
        formatted = _format_message(row)
        messages.append(
            {
                "formatted": formatted,
                "tokens": _estimate_tokens(formatted),
                "row": row,
            }
        )
    if not messages:
        return []

    chunks = []
    i = 0
    while i < len(messages):
        chunk_msgs = []
        chunk_tokens = 0

        # Collect messages up to token/message limits
        j = i
        while j < len(messages):
            msg_tokens = messages[j]["tokens"]
            if chunk_tokens + msg_tokens > MAX_CHUNK_TOKENS and chunk_msgs:
                break
            if len(chunk_msgs) >= MAX_CHUNK_MESSAGES:
                break
            chunk_msgs.append(messages[j])
            chunk_tokens += msg_tokens
            j += 1

        if not chunk_msgs:
            i += 1
            continue

        # Build chunk content (no header — metadata is stored separately)
        content = "\n".join(m["formatted"] for m in chunk_msgs)

        # Extract metadata for filtered retrieval
        participants = list(
            {
                str(m["row"].get("from") or m["row"].get("actor") or "Unknown")
                for m in chunk_msgs
            }
        )
        first_dt = pd.to_datetime(chunk_msgs[0]["row"]["datetime"])
        last_dt = pd.to_datetime(chunk_msgs[-1]["row"]["datetime"])

        doc = Document(
            page_content=content,
            metadata={
                "chat_name": chat_name,
                "chat_id": str(chunk_msgs[0]["row"].get("chat_id", "")),
                "participants": ", ".join(sorted(participants)),
                "date_start": first_dt.isoformat(),
                "date_end": last_dt.isoformat(),
                "month": first_dt.strftime("%Y-%m"),
                "message_count": len(chunk_msgs),
            },
        )
        chunks.append(doc)

        # Advance with overlap: step back by OVERLAP_MESSAGES
        if j >= len(messages):
            break
        i = max(i + 1, j - OVERLAP_MESSAGES)

    return chunks


def _chunk_messages(df: pd.DataFrame, selected_chat_ids: set) -> list[Document]:
    """Chunk messages from selected chats into Documents."""
    mask = (
        df["chat_id"].isin(selected_chat_ids)
        & df["text"].notna()
        & (df["text"].astype(str).str.strip() != "")
        & (df["type"] == "message")
    )
    filtered = df[mask].copy()
    if filtered.empty:
        return []

    all_docs = []
    for (chat_id, chat_name), group in filtered.groupby(["chat_id", "chat_name"]):
        segments = _split_by_time_gap(group)
        for segment in segments:
            docs = _segment_to_chunks(segment, chat_name)
            all_docs.extend(docs)
    return all_docs


# ═══════════════════════════════════════════════════════════════════════════
# Vector store management
# ═══════════════════════════════════════════════════════════════════════════


def _get_df_hash(df: pd.DataFrame) -> str:
    """Compute a lightweight hash of key DataFrame columns."""
    key_cols = ["chat_id", "date_unixtime", "from_id"]
    cols = [c for c in key_cols if c in df.columns]
    raw = pd.util.hash_pandas_object(df[cols]).values.tobytes()
    return hashlib.md5(raw).hexdigest()


def _save_meta(df_hash: str, selected_chat_ids: list, embedding_model: str) -> None:
    """Persist index metadata for cache validation."""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(
        json.dumps(
            {
                "df_hash": df_hash,
                "selected_chat_ids": selected_chat_ids,
                "embedding_model": embedding_model,
                "timestamp": time.time(),
            }
        )
    )


def _load_meta() -> dict | None:
    """Load persisted metadata."""
    if META_FILE.exists():
        try:
            return json.loads(META_FILE.read_text())
        except Exception:
            return None
    return None


def _close_vector_store(vs: Chroma) -> None:
    """Properly close a Chroma vector store, releasing SQLite connections."""
    try:
        vs._client._system.stop()
        vs._client.clear_system_cache()
    except Exception:
        pass


def _load_existing_store(embedding_model_id: str) -> Chroma | None:
    """Try to load a persisted Chroma store from disk."""
    if not PERSIST_DIR.exists():
        return None
    # Don't load if built with a different embedding model (dimension mismatch)
    meta = _load_meta()
    if meta and meta.get("embedding_model") != embedding_model_id:
        return None
    try:
        embeddings = _get_embeddings(embedding_model_id)
        store = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings,
        )
        store._collection.count()
        return store
    except Exception:
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        return None


def build_vector_store(
    df: pd.DataFrame,
    selected_chat_ids: set,
    embedding_model_id: str,
    progress_callback=None,
) -> Chroma | None:
    """Build a Chroma vector store from chat messages."""
    docs = _chunk_messages(df, selected_chat_ids)
    if not docs:
        return None

    embeddings = _get_embeddings(embedding_model_id)

    # Close existing store if any, then wipe directory
    existing = st.session_state.get("rag_vector_store")
    if existing is not None:
        _close_vector_store(existing)
        st.session_state.pop("rag_vector_store", None)

    # Wipe persist dir; also clear Chroma's internal caches to avoid
    # stale SQLite connections referencing an old schema.
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    try:
        from chromadb.api.client import Client as _ChromaClient

        _ChromaClient.clear_system_cache()
    except Exception:
        pass
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    total_batches = (len(docs) + batch_size - 1) // batch_size

    vector_store = None
    for i in range(total_batches):
        batch = docs[i * batch_size : (i + 1) * batch_size]
        if progress_callback:
            progress_callback(i + 1, total_batches)

        for attempt in range(3):
            try:
                if vector_store is None:
                    vector_store = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=str(PERSIST_DIR),
                    )
                else:
                    vector_store.add_documents(batch)
                break
            except Exception as e:
                err = str(e).lower()
                if ("readonly" in err or "no such table" in err) and attempt < 2:
                    # Corrupted or locked DB — wipe and retry
                    if vector_store is not None:
                        _close_vector_store(vector_store)
                        vector_store = None
                    if PERSIST_DIR.exists():
                        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
                    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
                    time.sleep(0.5)
                    continue
                raise

    df_hash = _get_df_hash(df)
    _save_meta(df_hash, list(selected_chat_ids), embedding_model_id)
    return vector_store


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval & generation (streaming)
# ═══════════════════════════════════════════════════════════════════════════


def retrieve_context(
    vector_store: Chroma,
    query: str,
    chat_filter: str | None = None,
) -> list[Document]:
    """Retrieve relevant chunks using MMR for diversity."""
    search_kwargs = {
        "k": RETRIEVAL_FINAL_K,
        "fetch_k": RETRIEVAL_K,
        "lambda_mult": MMR_LAMBDA,
    }
    if chat_filter and chat_filter != "All chats":
        search_kwargs["filter"] = {"chat_name": chat_filter}

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )
    return retriever.invoke(query)


def stream_answer(
    docs: list[Document],
    query: str,
    user_name: str,
    llm_model: str,
) -> Generator[str, None, None]:
    """Stream an answer from the LLM given retrieved documents."""
    if not docs:
        yield "I couldn't find any relevant messages for that query."
        return

    # Build context with chunk metadata for grounding
    context_parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"[Excerpt {i} — {meta.get('chat_name', '?')}, {meta.get('date_start', '?')[:10]}]"
        context_parts.append(f"{header}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(user_name=user_name)),
        HumanMessage(
            content=(
                "## Retrieved chat excerpts\n\n"
                f"{context}\n\n"
                "---\n\n"
                f"## My question\n\n{query}"
            )
        ),
    ]

    llm = ChatOllama(model=llm_model, temperature=0.2, reasoning=False)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


# ═══════════════════════════════════════════════════════════════════════════
# User detection
# ═══════════════════════════════════════════════════════════════════════════


@st.cache_data
def _detect_user(df: pd.DataFrame) -> str:
    """Detect the current user's name from the DataFrame.
    Primary: sender of Saved Messages. Fallback: most frequent sender in
    private chats."""
    saved = df[df["chat_type"] == "saved_messages"]
    if not saved.empty:
        names = saved["from"].dropna()
        if not names.empty:
            return names.iloc[0]

    private = df[df["chat_type"] == "personal_chat"]
    if not private.empty:
        counts = private["from"].value_counts()
        if not counts.empty:
            return counts.index[0]
    return "User"


# ═══════════════════════════════════════════════════════════════════════════
# Example questions
# ═══════════════════════════════════════════════════════════════════════════

_QUESTION_TEMPLATES = [
    "Did {name} ever mention moving to another city?",
    "What restaurants or cafes did {name} recommend?",
    "Did {name} and I talk about any movies or books?",
    "Has {name} shared any links or articles with me?",
    "What did {name} complain about?",
    "Did {name} mention any trips or travel plans?",
    "What hobbies or interests did {name} bring up?",
    "Did {name} recommend any music or podcasts?",
]


def _build_example_questions(df: pd.DataFrame, user_name: str) -> list[str]:
    """Build example questions using real contact names from private chats."""
    import random

    private = df[df["chat_type"] == "personal_chat"]
    if private.empty:
        templates = random.sample(_QUESTION_TEMPLATES, 4)
        return [t.format(name="a friend") for t in templates]

    senders = private[private["from"] != user_name]["from"]
    senders = senders[
        senders.notna() & ~senders.str.strip().str.lower().isin(["unknown", ""])
    ]
    contacts = senders.value_counts().head(8).index.tolist()
    if not contacts:
        templates = random.sample(_QUESTION_TEMPLATES, 4)
        return [t.format(name="a friend") for t in templates]

    # Pick 4 random template+contact pairs
    templates = random.sample(_QUESTION_TEMPLATES, min(4, len(_QUESTION_TEMPLATES)))
    chosen_contacts = [random.choice(contacts) for _ in templates]

    return [t.format(name=c) for t, c in zip(templates, chosen_contacts)]


def _check_ollama() -> bool:
    """Return True if Ollama is reachable."""
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════


def render_rag_page(messages_df: pd.DataFrame) -> None:
    """Render the RAG chat page."""
    st.header("💬 Chat with Your Data")
    st.caption(
        "Ask questions about your Telegram history. "
        "Runs fully locally via Ollama + sentence-transformers."
    )

    user_name = _detect_user(messages_df)
    df_hash = _get_df_hash(messages_df)

    # ── Sidebar settings ─────────────────────────────────────────────────
    emb_options = list(EMBEDDING_MODELS.keys())
    llm_options = list(LLM_MODELS.keys())

    with st.sidebar:
        with st.expander("⚙️ RAG Settings", expanded=True):
            emb_label = st.selectbox(
                "Embedding model",
                options=emb_options,
                index=st.session_state.get("rag_emb_index", 0),
            )
            st.session_state["rag_emb_index"] = emb_options.index(emb_label)
            emb_model_id = EMBEDDING_MODELS[emb_label]

            llm_label = st.selectbox(
                "LLM model",
                options=llm_options,
                index=st.session_state.get("rag_llm_index", 0),
                help="Make sure you've pulled the model: `ollama pull <model>`",
            )
            st.session_state["rag_llm_index"] = llm_options.index(llm_label)
            llm_model_id = LLM_MODELS[llm_label]

            # Chat selection
            chat_options = (
                messages_df.groupby(["chat_id", "chat_name"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            chat_display = {
                f"{row['chat_name']} (ID: {row['chat_id']})": row["chat_id"]
                for _, row in chat_options.iterrows()
            }
            selected_display = st.multiselect(
                "Chats to index",
                options=list(chat_display.keys()),
                default=list(chat_display.keys()),
            )
            selected_ids = {chat_display[d] for d in selected_display}

            # Index actions
            st.divider()
            has_index = "rag_vector_store" in st.session_state

            if st.button(
                "🔨 Rebuild Index" if has_index else "🔨 Build Index",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["rag_build_requested"] = True

            if has_index and st.button("🗑️ Clear Index", use_container_width=True):
                _close_vector_store(st.session_state.rag_vector_store)
                del st.session_state["rag_vector_store"]
                st.session_state.pop("rag_df_hash", None)
                st.session_state.pop("rag_emb_model", None)
                shutil.rmtree(PERSIST_DIR, ignore_errors=True)
                st.rerun()

            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state["rag_chat_history"] = []
                st.rerun()

            # Index stats
            meta = _load_meta()
            if meta and "timestamp" in meta:
                from datetime import datetime, timezone

                built = datetime.fromtimestamp(meta["timestamp"], tz=timezone.utc)
                st.caption(
                    f"Index built: {built:%Y-%m-%d %H:%M UTC}\n\n"
                    f"Embedding: {meta.get('embedding_model', '?')}"
                )

    # ── Build index if requested ─────────────────────────────────────────
    if st.session_state.pop("rag_build_requested", False):
        if not selected_ids:
            st.error("Select at least one chat to index.")
            return

        bar = st.progress(0, text="Chunking and embedding messages…")

        def _on_progress(cur, total):
            bar.progress(cur / total, text=f"Embedding batch {cur}/{total}…")

        try:
            vs = build_vector_store(
                messages_df,
                selected_ids,
                emb_model_id,
                progress_callback=_on_progress,
            )
            if vs is None:
                st.error("No text messages found in the selected chats.")
                return
            st.session_state["rag_vector_store"] = vs
            st.session_state["rag_df_hash"] = df_hash
            st.session_state["rag_emb_model"] = emb_model_id
            bar.progress(1.0, text="Done!")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to build index: {e}")
            return

    # ── Load or validate index ───────────────────────────────────────────
    emb_match = st.session_state.get("rag_emb_model") == emb_model_id
    has_store = "rag_vector_store" in st.session_state
    meta = _load_meta()
    disk_hash_match = meta is not None and meta.get("df_hash") == df_hash

    if has_store and emb_match and st.session_state.get("rag_df_hash") == df_hash:
        vector_store = st.session_state["rag_vector_store"]
    elif has_store and not emb_match:
        # Dropdown changed but in-memory store uses old model — release it
        _close_vector_store(st.session_state["rag_vector_store"])
        del st.session_state["rag_vector_store"]
        st.session_state.pop("rag_df_hash", None)
        st.session_state.pop("rag_emb_model", None)
        # Try loading from disk (works if dropdown matches the on-disk model)
        existing = _load_existing_store(emb_model_id)
        if existing and disk_hash_match:
            st.session_state["rag_vector_store"] = existing
            st.session_state["rag_df_hash"] = df_hash
            st.session_state["rag_emb_model"] = emb_model_id
            vector_store = existing
        else:
            disk_model = meta.get("embedding_model", "") if meta else ""
            if disk_model and disk_model != emb_model_id:
                st.warning(
                    f"Index on disk was built with **{disk_model}**. "
                    "Select that model to reuse it, or **Rebuild Index** "
                    "with the current model."
                )
            else:
                st.info(
                    "👈 Open **RAG Settings** in the sidebar and click "
                    "**Build Index** to get started."
                )
            return
    else:
        existing = _load_existing_store(emb_model_id)
        if existing and disk_hash_match:
            st.session_state["rag_vector_store"] = existing
            st.session_state["rag_df_hash"] = df_hash
            st.session_state["rag_emb_model"] = emb_model_id
            vector_store = existing
        else:
            disk_model = meta.get("embedding_model", "") if meta else ""
            if disk_model and disk_model != emb_model_id:
                st.warning(
                    f"Index on disk was built with **{disk_model}**. "
                    "Select that model to reuse it, or **Rebuild Index** "
                    "with the current model."
                )
            else:
                st.info(
                    "👈 Open **RAG Settings** in the sidebar and click "
                    "**Build Index** to get started."
                )
            return

    chunk_count = vector_store._collection.count()
    st.caption(f"✅ Index ready — **{chunk_count:,}** chunks from your selected chats")

    # ── Chat history ─────────────────────────────────────────────────────
    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = []

    history = st.session_state["rag_chat_history"]

    # Seed greeting with example questions on first load
    if not history:
        examples = _build_example_questions(messages_df, user_name)
        history.append(
            {
                "role": "assistant",
                "content": (
                    "👋 Hi! I can answer questions about your Telegram chats. "
                    "Try one of these:"
                ),
                "examples": examples,
            }
        )

    # Render chat history
    for i, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Render clickable example question buttons
            if msg.get("examples"):
                cols = st.columns(len(msg["examples"]))
                for j, (col, q) in enumerate(zip(cols, msg["examples"])):
                    if col.button(q, use_container_width=True, key=f"rag_ex_{i}_{j}"):
                        st.session_state["rag_pending_query"] = q
                        st.rerun()
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        meta = src.metadata
                        st.caption(
                            f"**{meta.get('chat_name', '?')}** · "
                            f"{meta.get('date_start', '?')[:10]}"
                        )
                        st.code(src.page_content, language=None)

    # ── Chat input ───────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about your chats…")
    query = user_input or st.session_state.pop("rag_pending_query", None)

    if query:
        # Show user message
        with st.chat_message("user"):
            st.markdown(query)
        history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            if not _check_ollama():
                err = (
                    "⚠️ Cannot reach Ollama. Make sure it's installed and running:\n\n"
                    "```bash\n"
                    "# Install from https://ollama.com\n"
                    f"ollama pull {llm_model_id}\n"
                    "ollama serve\n"
                    "```"
                )
                st.warning(err)
                history.append({"role": "assistant", "content": err})
            else:
                try:
                    # Retrieve
                    with st.spinner("Searching…"):
                        docs = retrieve_context(vector_store, query)

                    # Stream answer
                    response = st.write_stream(
                        stream_answer(docs, query, user_name, llm_model_id)
                    )
                    if docs:
                        with st.expander("📄 Sources"):
                            for src in docs:
                                meta = src.metadata
                                st.caption(
                                    f"**{meta.get('chat_name', '?')}** · "
                                    f"{meta.get('date_start', '?')[:10]}"
                                )
                                st.code(src.page_content, language=None)

                    history.append(
                        {"role": "assistant", "content": response, "sources": docs}
                    )
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    history.append({"role": "assistant", "content": err})

"""
RAG (Retrieval-Augmented Generation) module for tglens.
Enables querying chat histories using natural language.
Runs fully locally: sentence-transformers for embeddings, Ollama for LLM.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# ── Model options ────────────────────────────────────────────────────────

EMBEDDING_MODELS = {
    "multilingual-e5-small (recommended)": "intfloat/multilingual-e5-small",
    "multilingual-e5-base (higher quality)": "intfloat/multilingual-e5-base",
    "all-MiniLM-L6-v2 (English only, fastest)": "sentence-transformers/all-MiniLM-L6-v2",
}

LLM_MODELS = {
    "Qwen 2.5 7B (recommended)": "qwen2.5:7b-instruct",
    "Qwen 2.5 14B (needs 32 GB RAM)": "qwen2.5:14b-instruct",
    "Qwen 2.5 3B (lightweight)": "qwen2.5:3b-instruct",
}

# ── Chunking parameters ─────────────────────────────────────────────────

GAP_MINUTES = 120  # silence gap that splits conversations
MAX_CHUNK_TOKENS = 400
MAX_CHUNK_MESSAGES = 50
OVERLAP_TOKENS = 40

# ── Persistence ──────────────────────────────────────────────────────────

PERSIST_DIR = Path(".tglens_index")
META_FILE = PERSIST_DIR / "meta.json"

# ── System prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an assistant that answers questions about a user's Telegram inbox.

Rules:
- Answer based ONLY on the provided chat excerpts.
- If the excerpts don't contain enough information, say so honestly.
- Be concise. Cite chat names, dates, and people when relevant.
- When quoting messages, use the exact text from the excerpts.
- The user's name is "{user_name}". Refer to them as "you" when describing their messages.
- Answer in the same language the user asks their question in."""

# ── Example questions for greeting ───────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "Who do I message the most?",
    "What did we talk about last week?",
    "Find messages about travel plans",
    "Summarize my recent conversations",
]


# ═════════════════════════════════════════════════════════════════════════
# E5 Embedding wrapper
# ═════════════════════════════════════════════════════════════════════════


class _E5Embeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings wrapper that adds E5 query/passage prefixes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════


def _get_df_hash(df: pd.DataFrame) -> str:
    """Stable hash of the DataFrame for cache invalidation."""
    content = pd.util.hash_pandas_object(df).values.tobytes()
    return hashlib.md5(content).hexdigest()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (words × 1.3)."""
    return int(len(text.split()) * 1.3)


@st.cache_resource
def _get_embeddings(model_id: str):
    """Load and cache the embedding model (persists across reruns)."""
    if model_id.startswith("intfloat/"):
        return _E5Embeddings(model_name=model_id)
    return HuggingFaceEmbeddings(model_name=model_id)


def _detect_user(df: pd.DataFrame) -> str:
    """Heuristic: the most frequent sender across private chats is the user."""
    private = df[df["chat_type"] == "personal_chat"]
    if private.empty:
        private = df
    counts = private["from"].dropna().value_counts()
    return counts.index[0] if not counts.empty else "Unknown"


def _check_ollama() -> bool:
    """Return True if Ollama is reachable."""
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


def _save_meta(df_hash: str, chat_ids: list, embedding_model: str) -> None:
    """Write index metadata sidecar for cache invalidation."""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(
        json.dumps(
            {
                "df_hash": df_hash,
                "chat_ids": sorted(int(c) for c in chat_ids),
                "embedding_model": embedding_model,
            }
        )
    )


def _load_meta() -> dict | None:
    """Read persisted index metadata, or None if not found."""
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return None


# ═════════════════════════════════════════════════════════════════════════
# Chunking
# ═════════════════════════════════════════════════════════════════════════


def _split_by_time_gap(group: pd.DataFrame, gap_minutes: int) -> list[pd.DataFrame]:
    """Split a sorted chat DataFrame into conversation segments by silence gaps."""
    if group.empty:
        return []

    segments: list[pd.DataFrame] = []
    cut_indices = [0]

    datetimes = group["datetime"].values
    for i in range(1, len(datetimes)):
        delta = (datetimes[i] - datetimes[i - 1]) / pd.Timedelta(minutes=1)
        if delta > gap_minutes:
            cut_indices.append(i)
    cut_indices.append(len(group))

    for start, end in zip(cut_indices, cut_indices[1:]):
        seg = group.iloc[start:end]
        if not seg.empty:
            segments.append(seg)

    return segments


def _format_message(row: pd.Series) -> str:
    sender = row.get("from") or row.get("actor") or "Unknown"
    time_str = (
        row["datetime"].strftime("%H:%M") if pd.notna(row.get("datetime")) else ""
    )
    return f"[{time_str}] {sender}: {row['text']}"


def _segment_to_chunks(
    segment: pd.DataFrame,
    chat_name: str,
    chat_id: str,
    chat_type: str,
) -> list[Document]:
    """Convert a conversation segment into one or more Document chunks."""
    msgs = [_format_message(row) for _, row in segment.iterrows()]
    msg_tokens = [_estimate_tokens(m) for m in msgs]

    date_start = str(segment["date_only"].iloc[0])
    date_end = str(segment["date_only"].iloc[-1])
    date_label = date_start if date_start == date_end else f"{date_start} → {date_end}"

    participants = segment["from"].dropna().unique().tolist()

    def _make_doc(lines: list[str], chunk_idx: int) -> Document:
        content = f"Chat: {chat_name} | Date: {date_label}\n\n" + "\n".join(lines)
        return Document(
            page_content=content,
            metadata={
                "chat_id": str(chat_id),
                "chat_name": str(chat_name),
                "chat_type": str(chat_type),
                "date_start": date_start,
                "date_end": date_end,
                "participants": ", ".join(participants[:10]),
                "message_count": len(lines),
                "chunk_index": chunk_idx,
            },
        )

    # Single-chunk fast path
    total_tokens = sum(msg_tokens)
    if total_tokens <= MAX_CHUNK_TOKENS and len(msgs) <= MAX_CHUNK_MESSAGES:
        return [_make_doc(msgs, 0)]

    # Token-aware sliding window
    chunks: list[Document] = []
    start = 0
    idx = 0

    while start < len(msgs):
        end = start + 1
        current_tokens = msg_tokens[start]

        while end < len(msgs):
            if current_tokens + msg_tokens[end] > MAX_CHUNK_TOKENS:
                break
            if (end - start) >= MAX_CHUNK_MESSAGES:
                break
            current_tokens += msg_tokens[end]
            end += 1

        chunks.append(_make_doc(msgs[start:end], idx))
        idx += 1

        if end >= len(msgs):
            break

        # Overlap: step back by ~OVERLAP_TOKENS worth of messages
        overlap_tokens = 0
        overlap_start = end
        while overlap_start > start and overlap_tokens < OVERLAP_TOKENS:
            overlap_start -= 1
            overlap_tokens += msg_tokens[overlap_start]
        start = max(overlap_start, start + 1)  # ensure progress

    return chunks


def _chunk_messages(df: pd.DataFrame, selected_chat_ids: set) -> list[Document]:
    """
    Chunk messages into Documents using conversation-aware splitting.

    Strategy:
        1. Filter to real text messages in selected chats
        2. Group by chat
        3. Within each chat, split on silence gaps (>2 h → new conversation)
        4. Each conversation segment → 1+ token-limited chunks with metadata
    """
    text_df = df[
        (df["text"].notna())
        & (df["text"].str.strip() != "")
        & (df["type"] == "message")
        & (df["chat_id"].isin(selected_chat_ids))
    ].copy()

    if text_df.empty:
        return []

    text_df = text_df.sort_values(["chat_id", "datetime"])
    documents: list[Document] = []

    for (chat_id, chat_name), group in text_df.groupby(["chat_id", "chat_name"]):
        chat_type = (
            group["chat_type"].iloc[0] if "chat_type" in group.columns else "unknown"
        )
        segments = _split_by_time_gap(group, GAP_MINUTES)
        for segment in segments:
            documents.extend(
                _segment_to_chunks(
                    segment, str(chat_name), str(chat_id), str(chat_type)
                )
            )

    return documents


# ═════════════════════════════════════════════════════════════════════════
# Vector store
# ═════════════════════════════════════════════════════════════════════════


def build_vector_store(
    df: pd.DataFrame,
    selected_chat_ids: set,
    embedding_model_id: str,
    progress_callback=None,
) -> Chroma | None:
    """Build a persistent ChromaDB vector store from messages."""
    documents = _chunk_messages(df, selected_chat_ids)
    if not documents:
        return None

    embeddings = _get_embeddings(embedding_model_id)

    # Wipe old index
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    batch_size = 256
    total_batches = (len(documents) + batch_size - 1) // batch_size

    first_batch = documents[:batch_size]
    if progress_callback:
        progress_callback(1, total_batches)

    vector_store = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        collection_name="tglens_messages",
        persist_directory=str(PERSIST_DIR),
    )

    for i in range(1, total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(documents))
        vector_store.add_documents(documents[start_idx:end_idx])
        if progress_callback:
            progress_callback(i + 1, total_batches)

    # Save metadata sidecar for cache invalidation
    _save_meta(
        df_hash=_get_df_hash(df),
        chat_ids=list(selected_chat_ids),
        embedding_model=embedding_model_id,
    )

    return vector_store


def _load_existing_store(embedding_model_id: str) -> Chroma | None:
    """Load a previously persisted vector store."""
    if not PERSIST_DIR.exists():
        return None
    embeddings = _get_embeddings(embedding_model_id)
    return Chroma(
        collection_name="tglens_messages",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )


# ═════════════════════════════════════════════════════════════════════════
# Query
# ═════════════════════════════════════════════════════════════════════════


def query_rag(
    vector_store: Chroma,
    query: str,
    user_name: str,
    llm_model: str,
    chat_filter: str | None = None,
    k: int = 8,
) -> tuple[str, list[Document]]:
    """Run a single-turn RAG query with optional chat-name filtering."""
    search_kwargs: dict = {"k": k}
    if chat_filter:
        search_kwargs["filter"] = {"chat_name": chat_filter}

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant messages found for your query.", []

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    llm = ChatOllama(model=llm_model, temperature=0.3)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(user_name=user_name)),
        HumanMessage(content=f"Chat excerpts:\n\n{context}\n\n---\nQuestion: {query}"),
    ]

    response = llm.invoke(messages)
    return response.content, docs


# ═════════════════════════════════════════════════════════════════════════
# Streamlit page
# ═════════════════════════════════════════════════════════════════════════


def render_rag_page(messages_df: pd.DataFrame) -> None:
    """Render the full "Chat with Data" page."""

    st.header("Chat with Your Data")
    st.caption(
        "Ask questions about your Telegram history · "
        "runs fully locally via Ollama + sentence-transformers"
    )

    user_name = _detect_user(messages_df)
    df_hash = _get_df_hash(messages_df)

    # ── Settings ─────────────────────────────────────────────────────────

    with st.expander("⚙️ Settings"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            embedding_choice = st.selectbox(
                "Embedding model",
                options=list(EMBEDDING_MODELS.keys()),
                index=0,
                help="Multilingual E5 models support English and Russian well.",
            )
        with col_m2:
            llm_choice = st.selectbox(
                "LLM model (Ollama)",
                options=list(LLM_MODELS.keys()),
                index=0,
                help="Make sure you've pulled the model: `ollama pull <model>`",
            )

        embedding_model_id = EMBEDDING_MODELS[embedding_choice]
        llm_model_id = LLM_MODELS[llm_choice]

        # Chat selection
        chat_info = (
            messages_df.groupby(["chat_id", "chat_name", "chat_type"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        chat_options = {
            f"{row['chat_name']}  ·  {row['chat_type']}  ·  {row['count']:,} msgs": row[
                "chat_id"
            ]
            for _, row in chat_info.iterrows()
        }
        selected_labels = st.multiselect(
            "Chats to index",
            options=list(chat_options.keys()),
            default=list(chat_options.keys()),
            help="Select which chats to include in the search index.",
        )
        selected_chat_ids = {chat_options[label] for label in selected_labels}

        st.divider()

        # Action buttons
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            build_clicked = st.button(
                "Build Index", type="primary", icon="🔨", use_container_width=True
            )
        with col_b:
            rebuild_clicked = st.button(
                "Rebuild Index", icon="🔄", use_container_width=True
            )
        with col_c:
            clear_clicked = st.button("Clear Chat", icon="🗑️", use_container_width=True)

    # ── Handle button actions ────────────────────────────────────────────

    if clear_clicked:
        st.session_state.rag_chat_history = []
        st.rerun()

    if rebuild_clicked:
        for key in ("rag_vector_store", "rag_df_hash", "rag_embedding_model"):
            st.session_state.pop(key, None)
        if PERSIST_DIR.exists():
            shutil.rmtree(PERSIST_DIR)
        build_clicked = True

    if build_clicked:
        if not selected_chat_ids:
            st.error("Select at least one chat to index.")
            return

        bar = st.progress(0, text="Chunking and embedding messages…")

        def _on_progress(cur, total):
            bar.progress(cur / total, text=f"Embedding batch {cur}/{total}…")

        try:
            vs = build_vector_store(
                messages_df,
                selected_chat_ids,
                embedding_model_id,
                progress_callback=_on_progress,
            )
            if vs is None:
                st.error("No text messages found in the selected chats.")
                return
            st.session_state.rag_vector_store = vs
            st.session_state.rag_df_hash = df_hash
            st.session_state.rag_embedding_model = embedding_model_id
            bar.progress(1.0, text="Done!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to build index: {e}")
        return

    # ── Resolve vector store (session state → disk → not found) ──────────

    hash_match = st.session_state.get("rag_df_hash") == df_hash
    model_match = st.session_state.get("rag_embedding_model") == embedding_model_id
    has_store = "rag_vector_store" in st.session_state

    if has_store and hash_match and model_match:
        vector_store = st.session_state.rag_vector_store
    else:
        meta = _load_meta()
        if (
            meta
            and meta.get("df_hash") == df_hash
            and meta.get("embedding_model") == embedding_model_id
        ):
            existing = _load_existing_store(embedding_model_id)
            if existing:
                st.session_state.rag_vector_store = existing
                st.session_state.rag_df_hash = df_hash
                st.session_state.rag_embedding_model = embedding_model_id
                vector_store = existing
            else:
                vector_store = None
        else:
            vector_store = None

    if vector_store is None:
        st.info(
            "Open **⚙️ Settings** above to select chats and **Build Index** "
            "before you can start chatting."
        )
        return

    chunk_count = vector_store._collection.count()
    st.caption(f"✅ Index ready — **{chunk_count:,}** chunks from your selected chats")

    # ── Initialize chat history ──────────────────────────────────────────

    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []

    # ── Resolve pending example query (from previous rerun) ──────────────

    pending_query: str | None = None
    if "_rag_pending_query" in st.session_state:
        pending_query = st.session_state.pop("_rag_pending_query")

    # ── Greeting (shown only when chat history is empty) ─────────────────

    if not st.session_state.rag_chat_history and pending_query is None:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi! I've indexed your chats and I'm ready to answer questions.\n\n"
                "Here are some things you can ask:"
            )

        cols = st.columns(2, gap="small")
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            with cols[i % 2]:
                if st.button(
                    example,
                    key=f"rag_example_{i}",
                    use_container_width=True,
                ):
                    st.session_state.rag_chat_history.append(
                        {"role": "user", "content": example}
                    )
                    st.session_state._rag_pending_query = example
                    st.rerun()

    # ── Render chat history ──────────────────────────────────────────────

    for msg in st.session_state.rag_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 Sources"):
                    for doc in msg["sources"]:
                        m = doc.metadata
                        st.caption(
                            f"**{m.get('chat_name')}** · {m.get('date_start')} · "
                            f"{m.get('message_count')} messages"
                        )
                        st.code(doc.page_content[:500], language=None)

    # ── Chat input ───────────────────────────────────────────────────────

    input_query = st.chat_input("Ask about your chats…")
    query = input_query or pending_query

    if query:
        # Add user message (skip if it was already added by example button)
        if input_query:
            st.session_state.rag_chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

        # Generate response
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
                st.session_state.rag_chat_history.append(
                    {"role": "assistant", "content": err}
                )
            else:
                with st.spinner("Searching & thinking…"):
                    try:
                        answer, sources = query_rag(
                            vector_store,
                            query,
                            user_name,
                            llm_model_id,
                        )
                        st.markdown(answer)
                        if sources:
                            with st.expander("📄 Sources"):
                                for doc in sources:
                                    m = doc.metadata
                                    st.caption(
                                        f"**{m.get('chat_name')}** · "
                                        f"{m.get('date_start')} · "
                                        f"{m.get('message_count')} messages"
                                    )
                                    st.code(doc.page_content[:500], language=None)
                        st.session_state.rag_chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )
                    except Exception as e:
                        err = f"Error: {e}"
                        st.error(err)
                        st.session_state.rag_chat_history.append(
                            {"role": "assistant", "content": err}
                        )

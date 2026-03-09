"""Streamlit UI for the RAG chat page."""

from __future__ import annotations

import random
import shutil
import time

import pandas as pd
import streamlit as st

from src.rag.config import EMBEDDING_MODELS, LLM_MODELS, PERSIST_DIR
from src.rag.retrieval import retrieve_context, stream_answer
from src.rag.vector_store import (
    build_vector_store,
    close_vector_store,
    get_df_hash,
    load_existing_store,
    load_meta,
)

# ── Example question templates ───────────────────────────────────────────

_QUESTION_TEMPLATES = [
    "Did {name} ever mention moving to another city?",
    "What restaurants or cafes did {name} recommend?",
    "Did {name} and I talk about any movies or books?",
    "Has {name} shared any links or articles with me?",
    "Did {name} mention any trips or travel plans?",
    "What hobbies or interests did {name} bring up?",
    "Did {name} recommend any music or podcasts?",
]


# ── Helpers ──────────────────────────────────────────────────────────────


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


def _build_example_questions(df: pd.DataFrame, user_name: str) -> list[str]:
    """Build example questions using real contact names from private chats."""
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


# ── Main page ────────────────────────────────────────────────────────────


def render_rag_page(messages_df: pd.DataFrame) -> None:
    """Render the RAG chat page."""
    st.header("💬 Chat with Your Data")
    st.caption(
        "Ask questions about your Telegram history. "
        "Runs fully locally via Ollama + sentence-transformers."
    )

    user_name = _detect_user(messages_df)
    df_hash = get_df_hash(messages_df)

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
                close_vector_store(st.session_state.rag_vector_store)
                del st.session_state["rag_vector_store"]
                st.session_state.pop("rag_df_hash", None)
                st.session_state.pop("rag_emb_model", None)
                shutil.rmtree(PERSIST_DIR, ignore_errors=True)
                st.rerun()

            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state["rag_chat_history"] = []
                st.rerun()

            # Index stats
            meta = load_meta()
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
            time.sleep(0.3)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to build index: {e}")
            return

    # ── Load or validate index ───────────────────────────────────────────
    emb_match = st.session_state.get("rag_emb_model") == emb_model_id
    has_store = "rag_vector_store" in st.session_state
    meta = load_meta()
    disk_hash_match = meta is not None and meta.get("df_hash") == df_hash

    if has_store and emb_match and st.session_state.get("rag_df_hash") == df_hash:
        vector_store = st.session_state["rag_vector_store"]
    elif has_store and not emb_match:
        # Dropdown changed but in-memory store uses old model — release it
        close_vector_store(st.session_state["rag_vector_store"])
        del st.session_state["rag_vector_store"]
        st.session_state.pop("rag_df_hash", None)
        st.session_state.pop("rag_emb_model", None)
        # Try loading from disk (works if dropdown matches the on-disk model)
        existing = load_existing_store(emb_model_id)
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
        existing = load_existing_store(emb_model_id)
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

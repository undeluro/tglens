"""ChromaDB vector store lifecycle — build, load, close, persist."""

from __future__ import annotations

import hashlib
import json
import shutil
import time

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma

from src.rag.chunking import chunk_messages
from src.rag.config import META_FILE, PERSIST_DIR
from src.rag.embeddings import get_embeddings


def get_df_hash(df: pd.DataFrame) -> str:
    """Compute a lightweight hash of key DataFrame columns."""
    key_cols = ["chat_id", "date_unixtime", "from_id"]
    cols = [c for c in key_cols if c in df.columns]
    raw = pd.util.hash_pandas_object(df[cols]).values.tobytes()
    return hashlib.md5(raw).hexdigest()


def save_meta(df_hash: str, selected_chat_ids: list, embedding_model: str) -> None:
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


def load_meta() -> dict | None:
    """Load persisted metadata."""
    if META_FILE.exists():
        try:
            return json.loads(META_FILE.read_text())
        except Exception:
            return None
    return None


def close_vector_store(vs: Chroma) -> None:
    """Properly close a Chroma vector store, releasing SQLite connections."""
    try:
        vs._client._system.stop()
        vs._client.clear_system_cache()
    except Exception:
        pass


def load_existing_store(embedding_model_id: str) -> Chroma | None:
    """Try to load a persisted Chroma store from disk."""
    if not PERSIST_DIR.exists():
        return None
    # Don't load if built with a different embedding model (dimension mismatch)
    meta = load_meta()
    if meta and meta.get("embedding_model") != embedding_model_id:
        return None
    try:
        embeddings = get_embeddings(embedding_model_id)
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
    docs = chunk_messages(df, selected_chat_ids)
    if not docs:
        return None

    embeddings = get_embeddings(embedding_model_id)

    # Close existing store if any, then wipe directory
    existing = st.session_state.get("rag_vector_store")
    if existing is not None:
        close_vector_store(existing)
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
                        close_vector_store(vector_store)
                        vector_store = None
                    if PERSIST_DIR.exists():
                        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
                    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
                    time.sleep(0.5)
                    continue
                raise

    df_hash = get_df_hash(df)
    save_meta(df_hash, list(selected_chat_ids), embedding_model_id)
    return vector_store

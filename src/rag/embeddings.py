"""Embedding model routing — Ollama, E5, and HuggingFace backends."""

from __future__ import annotations

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings


class _E5Embeddings(HuggingFaceEmbeddings):
    """Wrapper that adds E5 query/passage prefixes for better retrieval."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


@st.cache_resource
def get_embeddings(model_id: str):
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

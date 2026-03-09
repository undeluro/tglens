"""Retrieval and LLM streaming."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.rag.config import MMR_LAMBDA, RETRIEVAL_FINAL_K, RETRIEVAL_K, SYSTEM_PROMPT

if TYPE_CHECKING:
    from collections.abc import Generator


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

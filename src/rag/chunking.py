"""Message chunking — DataFrame to LangChain Documents."""

from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document

from src.rag.config import (
    GAP_MINUTES,
    MAX_CHUNK_MESSAGES,
    MAX_CHUNK_TOKENS,
    OVERLAP_MESSAGES,
)


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


def chunk_messages(df: pd.DataFrame, selected_chat_ids: set) -> list[Document]:
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

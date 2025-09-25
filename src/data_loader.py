"""
Telegram dump data loader module
Handles loading and preprocessing of Telegram JSON exports
"""

import json
import pandas as pd
from typing import Optional
import streamlit as st


@st.cache_data
def load_into_df(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load and process Telegram JSON export file into a DataFrame (cached)

    Args:
        uploaded_file: Streamlit uploaded file object containing Telegram JSON export

    Returns:
        pd.DataFrame: Processed messages with additional date/time columns, or None if failed
    """
    if uploaded_file is None:
        return None

    try:
        # Parse JSON file
        raw_data = json.load(uploaded_file)
    except json.JSONDecodeError:
        st.error(
            "Error reading JSON file. Please make sure it's a valid Telegram export."
        )
        return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

    try:
        # Extract messages from all chats
        chats = raw_data.get("chats", {}).get("list", [])
        all_messages = []

        for chat in chats:
            chat_name = chat.get("name", "Saved Messages")
            chat_type = chat.get("type", "unknown")
            chat_id = chat.get("id", 0)

            # Include all chat types for comprehensive analysis
            # Different tabs will filter as needed:
            # - General Overview,Contact Insights: personal_chat, saved_messages
            # - Group Insights: priv group, priv supergroup

            messages = chat.get("messages", [])
            for message in messages:
                # Extract text content
                text_content = _extract_text_content(message.get("text", ""))

                message_data = {
                    "chat_id": chat_id,
                    "chat_name": chat_name,
                    "chat_type": chat_type,
                    "message_id": message.get("id", 0),
                    "from": message.get("from", "Unknown"),
                    "from_id": message.get("from_id", ""),
                    "actor": message.get("actor", None),
                    "actor_id": message.get("actor_id", None),
                    "date": message.get("date", ""),
                    "date_unixtime": message.get("date_unixtime", 0),
                    "text": text_content,
                    "text_length": len(text_content) if text_content else 0,
                    "type": message.get("type", "message"),
                    "action": message.get("action", None),
                    "reply_to_message_id": message.get("reply_to_message_id", None),
                    "forwarded_from": message.get("forwarded_from", None),
                    "media_type": message.get("media_type", None),
                    "file": message.get("file", None),
                    "width": message.get("width", None),
                    "height": message.get("height", None),
                    "duration_seconds": message.get("duration_seconds", None),
                }
                all_messages.append(message_data)

        if not all_messages:
            st.error("No messages found in the uploaded file.")
            return None

        # Create DataFrame
        messages_df = pd.DataFrame(all_messages)

        # Add datetime processing
        if not messages_df.empty:
            # Convert date string to datetime
            messages_df["datetime"] = pd.to_datetime(messages_df["date"])

            # Add derived date columns
            messages_df["date_only"] = messages_df["datetime"].dt.date
            messages_df["year"] = messages_df["datetime"].dt.year
            messages_df["month"] = messages_df["datetime"].dt.month
            messages_df["day"] = messages_df["datetime"].dt.day
            messages_df["hour"] = messages_df["datetime"].dt.hour
            messages_df["weekday"] = messages_df["datetime"].dt.day_name()
            messages_df["month_name"] = messages_df["datetime"].dt.month_name()

        return messages_df

    except Exception as e:
        st.error(f"Error extracting messages: {str(e)}")
        return None


def _extract_text_content(text_field) -> str:
    """
    Extract text content from various text field formats

    Args:
        text_field: Text field from Telegram JSON (can be string, list, or dict)

    Returns:
        str: Extracted text content
    """
    if isinstance(text_field, str):
        return text_field
    elif isinstance(text_field, list):
        # Handle array of text objects
        text_parts = []
        for item in text_field:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        return "".join(text_parts)
    elif isinstance(text_field, dict) and "text" in text_field:
        return text_field["text"]
    else:
        return ""

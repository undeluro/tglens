"""
Telegram dump data loader module
Handles loading and preprocessing of Telegram JSON exports
"""

import json
import pandas as pd
from typing import Optional
import streamlit as st


class TelegramDataLoader:
    """Load and process Telegram JSON export data"""

    def __init__(self):
        self.raw_data = None

    def load_from_file(self, uploaded_file) -> bool:
        """
        Load data from uploaded JSON file
        Returns True if successful, False otherwise
        """
        try:
            self.raw_data = json.load(uploaded_file)
            return True
        except json.JSONDecodeError:
            st.error(
                "Error reading JSON file. Please make sure it's a valid Telegram export."
            )
            return False
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return False

    def extract_messages(self) -> Optional[pd.DataFrame]:
        """
        Extract and process messages from raw data into a DataFrame
        """
        if not self.raw_data:
            return None

        try:
            chats = self.raw_data.get("chats", {}).get("list", [])
            all_messages = []

            for chat in chats:
                chat_name = chat.get("name", "Saved Messages")
                chat_type = chat.get("type", "unknown")
                chat_id = chat.get("id", 0)

                # Only include personal chats and saved messages
                if chat_type != "personal_chat" and chat_type != "saved_messages":
                    continue

                messages = chat.get("messages", [])
                for message in messages:
                    # Extract text content
                    text_content = self._extract_text_content(message.get("text", ""))

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
            self.messages_df = pd.DataFrame(all_messages)

            # Process datetime directly here
            if not self.messages_df.empty:
                # Convert date string to datetime
                self.messages_df["datetime"] = pd.to_datetime(self.messages_df["date"])

                # Add derived date columns
                self.messages_df["date_only"] = self.messages_df["datetime"].dt.date
                self.messages_df["year"] = self.messages_df["datetime"].dt.year
                self.messages_df["month"] = self.messages_df["datetime"].dt.month
                self.messages_df["day"] = self.messages_df["datetime"].dt.day
                self.messages_df["hour"] = self.messages_df["datetime"].dt.hour
                self.messages_df["weekday"] = self.messages_df["datetime"].dt.day_name()
                self.messages_df["month_name"] = self.messages_df[
                    "datetime"
                ].dt.month_name()

            return self.messages_df

        except Exception as e:
            st.error(f"Error extracting messages: {str(e)}")
            return None

    def _extract_text_content(self, text_field) -> str:
        """
        Extract text content from various text field formats
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

    def _process_datetime(self):
        """
        Convert date strings to datetime objects and add derived columns
        """
        if self.messages_df is not None:
            # Convert date string to datetime
            self.messages_df["datetime"] = pd.to_datetime(self.messages_df["date"])

            # Add derived date columns
            self.messages_df["date_only"] = self.messages_df["datetime"].dt.date
            self.messages_df["year"] = self.messages_df["datetime"].dt.year
            self.messages_df["month"] = self.messages_df["datetime"].dt.month
            self.messages_df["day"] = self.messages_df["datetime"].dt.day
            self.messages_df["hour"] = self.messages_df["datetime"].dt.hour
            self.messages_df["weekday"] = self.messages_df["datetime"].dt.day_name()
            self.messages_df["month_name"] = self.messages_df[
                "datetime"
            ].dt.month_name()

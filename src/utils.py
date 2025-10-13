"""
Utility functions for tglens Telegram Analytics
"""

from typing import Optional, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import streamlit as st


def format_number(num: int) -> str:
    """Format large numbers with appropriate suffixes"""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def get_time_period_filter(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Filter dataframe by time period

    Args:
        df: Messages dataframe
        period: '7d', '30d', '90d', '1y', 'all'
    """
    if period == "all":
        return df

    now = df["datetime"].max()

    if period == "7d":
        start_date = now - timedelta(days=7)
    elif period == "30d":
        start_date = now - timedelta(days=30)
    elif period == "90d":
        start_date = now - timedelta(days=90)
    elif period == "1y":
        start_date = now - timedelta(days=365)
    else:
        return df

    return df[df["datetime"] >= start_date]


def create_activity_heatmap(df: pd.DataFrame, title: str = "Activity Heatmap"):
    """Create a heatmap showing activity by hour and day of week"""
    # Create hour-weekday combinations
    activity_data = (
        df.groupby(["weekday", "hour"]).size().reset_index(name="message_count")
    )

    # Create pivot table for heatmap
    heatmap_data = activity_data.pivot(
        index="weekday", columns="hour", values="message_count"
    ).fillna(0)

    # Reorder weekdays
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    heatmap_data = heatmap_data.reindex(weekday_order)

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=list(range(24)),
            y=weekday_order,
            colorscale="Viridis",
            showscale=True,
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>Hour: %{x}<br>Messages: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title, xaxis_title="Hour of Day", yaxis_title="Day of Week", height=400
    )

    return fig


def create_message_timeline(
    df: pd.DataFrame, chat_name: str = None, granularity: str = "day"
):
    """
    Create a timeline chart of messages

    Args:
        df: Messages dataframe
        chat_name: Specific chat to analyze (None for all)
        granularity: 'hour', 'day', 'week', 'month'
    """
    if chat_name:
        df = df[df["chat_name"] == chat_name]

    if granularity == "hour":
        df["period"] = df["datetime"].dt.floor("H")
        title_period = "Hour"
    elif granularity == "day":
        df["period"] = df["datetime"].dt.floor("D")
        title_period = "Day"
    elif granularity == "week":
        df["period"] = df["datetime"].dt.to_period("W").dt.start_time
        title_period = "Week"
    elif granularity == "month":
        df["period"] = df["datetime"].dt.to_period("M").dt.start_time
        title_period = "Month"
    else:
        df["period"] = df["datetime"].dt.floor("D")
        title_period = "Day"

    timeline_data = df.groupby("period").size().reset_index(name="message_count")

    fig = px.line(
        timeline_data,
        x="period",
        y="message_count",
        title=f"Messages per {title_period}" + (f" - {chat_name}" if chat_name else ""),
        labels={"period": "Date", "message_count": "Number of Messages"},
        markers=True,
    )

    fig.update_layout(height=400)
    return fig


@st.cache_data
def get_basic_stats(messages_df: pd.DataFrame) -> Dict:
    """Get basic statistics about the loaded data"""
    if messages_df is None or messages_df.empty:
        return {}

    calls_df = messages_df[messages_df["action"] == "phone_call"]

    return {
        "total_messages": len(messages_df),
        "total_chats": messages_df["chat_id"].nunique(),
        "date_range_start": messages_df["datetime"].min(),
        "date_range_end": messages_df["datetime"].max(),
        "total_days": (
            messages_df["datetime"].max() - messages_df["datetime"].min()
        ).days
        + 1,
        "total_text_length": messages_df["text_length"].sum(),
        "media_messages": messages_df["media_type"].notna().sum(),
        "total_calls": len(calls_df),
        "avg_call_duration": calls_df["duration_seconds"].mean()
        if len(calls_df) > 0 and "duration_seconds" in calls_df.columns
        else 0,
    }


@st.cache_data
def get_chat_summary(messages_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Generate summary statistics per chat"""
    if messages_df is None or messages_df.empty:
        return None

    chat_stats = (
        messages_df.groupby(["chat_id", "chat_name", "chat_type"])
        .agg(
            {
                "message_id": "count",
                "text_length": ["sum", "mean"],
                "from": "nunique",
                "datetime": ["min", "max"],
                "media_type": lambda x: x.notna().sum(),
            }
        )
        .round(2)
    )

    # Flatten column names
    chat_stats.columns = [
        "message_count",
        "total_text_length",
        "avg_text_length",
        "unique_senders",
        "first_message",
        "last_message",
        "media_count",
    ]

    chat_stats = chat_stats.reset_index()
    chat_stats["days_active"] = (
        chat_stats["last_message"] - chat_stats["first_message"]
    ).dt.days + 1
    chat_stats["messages_per_day"] = (
        chat_stats["message_count"] / chat_stats["days_active"]
    ).round(2)

    return chat_stats.sort_values("message_count", ascending=False)


def create_full_timeline(chat_df, start_date, end_date, chat_name):
    """Create a timeline chart that spans the full Telegram usage period"""
    import pandas as pd
    import plotly.express as px

    # Create a copy to avoid SettingWithCopyWarning
    chat_df_copy = chat_df.copy()

    # Group messages by day
    chat_df_copy["date"] = chat_df_copy["datetime"].dt.date
    daily_counts = chat_df_copy.groupby("date").size().reset_index(name="message_count")
    daily_counts["date"] = pd.to_datetime(daily_counts["date"])

    # Create a complete date range from start to end
    date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq="D")
    full_timeline = pd.DataFrame({"date": date_range})

    # Merge with actual message counts, filling missing days with 0
    full_timeline = full_timeline.merge(daily_counts, on="date", how="left")
    full_timeline["message_count"] = full_timeline["message_count"].fillna(0)

    # Create the plot
    fig = px.line(
        full_timeline,
        x="date",
        y="message_count",
        title=f"Daily Messages Timeline - {chat_name}",
        labels={"date": "Date", "message_count": "Number of Messages"},
        markers=True,
    )

    # Add hover information
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Messages: %{y}<extra></extra>")

    # Update layout
    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Number of Messages",
        showlegend=False,
    )

    # Add annotation about the full period
    total_days = (end_date - start_date).days + 1
    fig.add_annotation(
        text=f"Showing {total_days} days of Telegram history",
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )

    return fig


def create_word_cloud(chat_df, chat_name):
    """Create and display a word cloud from chat messages"""
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import re
        import streamlit as st

        # Get all text messages
        text_messages = chat_df[chat_df["text_length"] > 0]["text"].dropna()

        if text_messages.empty:
            st.info("No text messages available for word cloud generation")
            return

        # Combine all text
        all_text = " ".join(text_messages.astype(str))

        # Basic text cleaning
        # Remove URLs
        all_text = re.sub(r"http\S+|www\S+|https\S+", "", all_text, flags=re.MULTILINE)
        # Remove extra whitespace
        all_text = re.sub(r"\s+", " ", all_text)
        # Remove special characters but keep basic punctuation
        all_text = re.sub(r"[^\w\s\-.,!?]", " ", all_text)

        if len(all_text.strip()) == 0:
            st.info("No suitable text found for word cloud generation")
            return

        # Basic stop words (common words to exclude) - English and Russian
        stop_words = {
            # English stop words
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "this",
            "that",
            "these",
            "those",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "must",
            "shall",
            "not",
            "no",
            "yes",
            "ok",
            "okay",
            "just",
            "now",
            "so",
            "well",
            "like",
            "really",
            # Russian stop words
            "это",
            "что",
            "тот",
            "быть",
            "весь",
            "как",
            "она",
            "так",
            "его",
            "но",
            "да",
            "ты",
            "к",
            "у",
            "же",
            "вы",
            "за",
            "бы",
            "по",
            "только",
            "ее",
            "мне",
            "было",
            "вот",
            "от",
            "меня",
            "еще",
            "нет",
            "о",
            "из",
            "ему",
            "теперь",
            "когда",
            "даже",
            "ну",
            "вдруг",
            "ли",
            "если",
            "уже",
            "или",
            "ни",
            "был",
            "него",
            "до",
            "вас",
            "нибудь",
            "опять",
            "уж",
            "вам",
            "ведь",
            "там",
            "потом",
            "себя",
            "и",
            "в",
            "во",
            "не",
            "он",
            "на",
            "я",
            "с",
            "со",
            "а",
            "то",
            "все",
            "ей",
            "они",
            "где",
            "есть",
            "надо",
            "ней",
            "для",
            "мы",
            "тебя",
            "их",
            "чем",
            "была",
            "сам",
            "чтоб",
            "без",
            "будто",
            "чего",
            "раз",
            "тоже",
            "себе",
            "под",
            "будет",
            "ж",
            "тогда",
            "кто",
            "этот",
            "того",
            "потому",
            "этой",
            "над",
            "всех",
            "нас",
            "при",
            "были",
            "будем",
            "будут",
            "этого",
            "которой",
            "которые",
            "которых",
            "которому",
            "которая",
            "которое",
            "которую",
            "очень",
            "также",
            "кроме",
            "первый",
            "хорошо",
            "через",
            "можете",
            "знаю",
            "сказать",
            "какой",
            "нужно",
            "еще",
            "че",
            "чё",
        }

        # Create word cloud
        wordcloud = WordCloud(
            width=400,
            background_color="white",
            max_words=100,
            colormap="viridis",
            relative_scaling=0.5,
            min_font_size=10,
            stopwords=stop_words,
        ).generate(all_text)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="gaussian")
        ax.axis("off")
        ax.set_title(f"Most Common Words - {chat_name}", fontsize=16, pad=20)

        # Display in Streamlit
        st.pyplot(fig)
        plt.close()

        # Add some statistics
        word_count = len(all_text.split())
        unique_words = len(set(all_text.lower().split()))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Words", f"{word_count:,}")
        with col2:
            st.metric("Unique Words", f"{unique_words:,}")

    except ImportError:
        st.error(
            "WordCloud library not installed. Please install it with: pip install wordcloud"
        )
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")

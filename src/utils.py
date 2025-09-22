"""
Utility functions for tglens Telegram Analytics
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta


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

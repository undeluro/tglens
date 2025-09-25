"""
Visualizations and analytics for tglens Telegram Analytics
Handles both data analysis and chart generation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Optional
from src.utils import create_activity_heatmap


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


# renders tab 1
def render_general_overview(messages_df):
    """Render the General Overview tab"""
    st.header("General Overview of Private Chats")

    if messages_df is None or messages_df.empty:
        st.warning("No messages data available.")
        return

    # Filter to only private chats (personal_chat and saved_messages)
    private_chats_df = messages_df[
        messages_df["chat_type"].isin(["personal_chat", "saved_messages"])
    ].copy()

    if private_chats_df.empty:
        st.warning("No private chat messages found.")
        return

    # Get basic statistics
    stats = get_basic_stats(private_chats_df)
    chat_summary = get_chat_summary(private_chats_df)

    # Key Metrics Row
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            label="💬 Total Messages",
            value=f"{stats['total_messages']:,}",
            help="Total number of messages in all chats",
        )

    with col2:
        st.metric(
            label="📞 Total Calls",
            value=f"{stats['total_calls']:,}",
            help="Total number of voice/video calls",
        )

    with col3:
        st.metric(
            label="👥 Total Chats",
            value=f"{stats['total_chats']:,}",
            help="Number of different chats/conversations",
        )

    with col4:
        days_active = stats["total_days"]
        avg_messages_per_day = stats["total_messages"] / max(days_active, 1)
        avg_calls_per_day = stats["total_calls"] / max(days_active, 1)
        st.metric(
            label="📅 Messages/Day",
            value=f"{avg_messages_per_day:.1f}",
            help="Average messages per day",
        )

    with col5:
        st.metric(
            label="📅 Calls/Day",
            value=f"{avg_calls_per_day:.1f}",
            help="Average calls per day",
        )

    with col6:
        avg_duration = stats["avg_call_duration"]
        if avg_duration > 0:
            minutes = int(avg_duration // 60)
            seconds = int(avg_duration % 60)
            duration_display = f"{minutes}:{seconds:02d}"
        else:
            duration_display = "0:00"

        st.metric(
            label="⏱️ Avg Call Duration",
            value=duration_display,
            help="Average duration of voice/video calls",
        )

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top 10 Most Active Chats")
        if chat_summary is not None:
            top_chats = chat_summary.head(10)
            fig = px.bar(
                top_chats,
                x="message_count",
                y="chat_name",
                orientation="h",
                title="Messages per Chat",
                labels={
                    "message_count": "Number of Messages",
                    "chat_name": "Chat Name",
                },
                color="message_count",
                color_continuous_scale="viridis",
            )
            fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("📞 Calls Distribution")
        if private_chats_df is not None:
            call_type_counts = (
                private_chats_df[private_chats_df["action"] == "phone_call"]
                .groupby("chat_name")
                .size()
                .reset_index(name="count")
            )
            fig = px.pie(
                call_type_counts,
                values="count",
                names="chat_name",
                title="Distribution of Calls per Contact",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

    # Media and Call Analysis
    col1, col2 = st.columns(2)

    with col1:
        # Video and Voice messages count per contact
        media_df = private_chats_df[
            private_chats_df["media_type"].isin(["video_message", "voice_message"])
        ]
        if not media_df.empty:
            media_counts = (
                media_df.groupby(["chat_name", "media_type"])
                .size()
                .reset_index(name="count")
            )

            if not media_counts.empty:
                # Sort by total count per contact
                contact_totals = (
                    media_counts.groupby("chat_name")["count"]
                    .sum()
                    .sort_values(ascending=False)
                )
                media_counts = (
                    media_counts.set_index("chat_name")
                    .loc[contact_totals.index]
                    .reset_index()
                )

                fig = px.bar(
                    media_counts,
                    x="chat_name",
                    y="count",
                    color="media_type",
                    title="Video & Voice Messages per Contact (Sorted by Total)",
                    labels={
                        "chat_name": "Contact",
                        "count": "Message Count",
                        "media_type": "Media Type",
                    },
                    color_discrete_map={
                        "video_message": "#FF6B6B",
                        "voice_message": "#4ECDC4",
                    },
                )
                fig.update_layout(height=400, xaxis_tickangle=45)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No video or voice messages found")
        else:
            st.info("No video or voice messages found")

    with col2:
        # Average call duration per contact
        calls_df = private_chats_df[private_chats_df["action"] == "phone_call"]
        if not calls_df.empty and "duration_seconds" in calls_df.columns:
            call_duration = (
                calls_df.groupby("chat_name")["duration_seconds"].mean().reset_index()
            )
            call_duration["duration_minutes"] = call_duration["duration_seconds"] / 60
            call_duration = call_duration.sort_values(
                "duration_minutes", ascending=False
            ).head(10)

            if not call_duration.empty:
                fig = px.bar(
                    call_duration,
                    x="chat_name",
                    y="duration_minutes",
                    title="Average Call Duration per Contact (Top 10)",
                    labels={
                        "chat_name": "Contact",
                        "duration_minutes": "Avg Duration (minutes)",
                    },
                    color="duration_minutes",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(height=400, xaxis_tickangle=45)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No call duration data found")
        else:
            st.info("No call data found")

    # Date Range
    st.subheader("📅 Activity Timeline")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**First Message:** {stats['date_range_start'].strftime('%B %d, %Y')}")
    with col2:
        st.info(f"**Last Message:** {stats['date_range_end'].strftime('%B %d, %Y')}")
    with col3:
        start_date = stats["date_range_start"]
        end_date = stats["date_range_end"]

        years = end_date.year - start_date.year
        months = end_date.month - start_date.month
        days = end_date.day - start_date.day

        # Adjust for negative days
        if days < 0:
            months -= 1
            days += (
                start_date.replace(month=start_date.month % 12 + 1, day=1)
                - pd.Timedelta(days=1)
            ).day

        # Adjust for negative months
        if months < 0:
            years -= 1
            months += 12

        duration_parts = []
        if years > 0:
            duration_parts.append(f"{years} year{'s' if years != 1 else ''}")
        if months > 0:
            duration_parts.append(f"{months} month{'s' if months != 1 else ''}")
        if days > 0:
            duration_parts.append(f"and {days} day{'s' if days != 1 else ''}")

        duration_str = " ".join(duration_parts) if duration_parts else "Same day"
        st.info(f"**Have been using Telegram for:** {duration_str}")

    # Messages over time

    # Group by month for better visualization
    private_chats_df["year_month"] = private_chats_df["datetime"].dt.to_period("M")
    monthly_counts = (
        private_chats_df.groupby("year_month").size().reset_index(name="message_count")
    )
    monthly_counts["year_month_str"] = monthly_counts["year_month"].astype(str)

    fig = px.line(
        monthly_counts,
        x="year_month_str",
        y="message_count",
        title="Messages per Month",
        labels={"year_month_str": "Month", "message_count": "Number of Messages"},
        markers=True,
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")

    # Activity by hour of day
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⏰ Activity by Hour of Day")
        hourly_activity = (
            private_chats_df.groupby("hour").size().reset_index(name="message_count")
        )
        fig = px.bar(
            hourly_activity,
            x="hour",
            y="message_count",
            title="Messages by Hour",
            labels={"hour": "Hour of Day", "message_count": "Number of Messages"},
            color="message_count",
            color_continuous_scale="plasma",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("📅 Activity by Day of Week")
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        daily_activity = (
            private_chats_df.groupby("weekday")
            .size()
            .reindex(weekday_order)
            .reset_index(name="message_count")
        )
        daily_activity["weekday"] = daily_activity.index.map(lambda x: weekday_order[x])

        fig = px.bar(
            daily_activity,
            x="weekday",
            y="message_count",
            title="Messages by Day of Week",
            labels={"weekday": "Day of Week", "message_count": "Number of Messages"},
            color="message_count",
            color_continuous_scale="viridis",
        )
        fig.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

    # Activity Heatmap
    st.subheader("🔥 Activity Heatmap")
    st.plotly_chart(create_activity_heatmap(private_chats_df), width="stretch")

    # Additional stats
    st.subheader("📝 Content Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="📝 Total Characters",
            value=f"{stats['total_text_length']:,}",
            help="Total characters in all messages",
        )

    with col2:
        avg_message_length = stats["total_text_length"] / max(
            stats["total_messages"], 1
        )
        st.metric(
            label="📏 Avg Message Length",
            value=f"{avg_message_length:.1f}",
            help="Average characters per message",
        )

    with col3:
        st.metric(
            label="🖼️ Media Messages",
            value=f"{stats['media_messages']:,}",
            help="Messages containing media (photos, videos, etc.)",
        )

    # Chat summary table
    if chat_summary is not None:
        st.subheader("📋 Chat Summary Table")

        # Display options
        display_df = chat_summary.head().copy()
        display_df["first_message"] = display_df["first_message"].dt.strftime(
            "%Y-%m-%d"
        )
        display_df["last_message"] = display_df["last_message"].dt.strftime("%Y-%m-%d")

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config={
                "chat_name": "Chat Name",
                "chat_type": "Type",
                "message_count": st.column_config.NumberColumn("Messages", format="%d"),
                "unique_senders": st.column_config.NumberColumn("Users", format="%d"),
                "avg_text_length": st.column_config.NumberColumn(
                    "Avg Length", format="%.1f"
                ),
                "messages_per_day": st.column_config.NumberColumn(
                    "Msgs/Day", format="%.2f"
                ),
                "first_message": "First Message",
                "last_message": "Last Message",
                "days_active": st.column_config.NumberColumn(
                    "Days Active", format="%d"
                ),
            },
        )


# renders tab 2
def render_contact_analysis(messages_df):
    """Render the Contact Analysis tab with full timeline coverage"""

    if messages_df is None or messages_df.empty:
        st.warning("No messages data available.")
        return

    # Filter to only private chats (personal_chat and saved_messages)
    private_chats_df = messages_df[
        messages_df["chat_type"].isin(["personal_chat", "saved_messages"])
    ].copy()

    if private_chats_df.empty:
        st.warning("No private chat messages found.")
        return

    st.header("Contact Insights")

    # Get unique combinations of chat name and ID to handle duplicate names
    chat_options = private_chats_df[["chat_name", "chat_id"]].drop_duplicates()

    # Create display names that show both name and ID for disambiguation
    chat_display_options = []
    chat_lookup = {}

    for _, row in chat_options.iterrows():
        chat_name = row["chat_name"]
        chat_id = row["chat_id"]
        display_name = f"{chat_name} (ID: {chat_id})"
        chat_display_options.append(display_name)
        chat_lookup[display_name] = (chat_name, chat_id)

    if len(chat_display_options) == 0:
        st.warning("No valid chats found in the data.")
        return

    # Contact selector with name + ID display
    selected_display = st.selectbox(
        "Select a chat", options=chat_display_options, key="contact_selector"
    )

    if selected_display:
        # Get the actual chat name and ID from the selection
        selected_chat_name, selected_chat_id = chat_lookup[selected_display]

        # Filter messages for the selected chat using both name and ID for accuracy
        chat_df = private_chats_df[
            (private_chats_df["chat_name"] == selected_chat_name)
            & (private_chats_df["chat_id"] == selected_chat_id)
        ]

        if chat_df.empty:
            st.warning("No messages found for this chat.")
            return

        # Basic stats for the chat
        total_messages = len(chat_df)
        total_calls = len(chat_df[chat_df["action"] == "phone_call"])
        calls_df = chat_df[chat_df["action"] == "phone_call"]
        avg_call_duration = (
            calls_df["duration_seconds"].mean()
            if len(calls_df) > 0 and "duration_seconds" in calls_df.columns
            else 0
        )
        total_media = chat_df["media_type"].notna().sum()
        voice_messages = len(chat_df[chat_df["media_type"] == "voice_message"])
        video_messages = len(chat_df[chat_df["media_type"] == "video_message"])

        # Key Metrics
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(label="💬 Total Messages", value=f"{total_messages:,}")

        with col2:
            st.metric(label="📞 Total Calls", value=f"{total_calls:,}")

        with col3:
            st.metric(label="🎵 Voice Messages", value=f"{voice_messages:,}")

        with col4:
            st.metric(label="🎥 Video Messages", value=f"{video_messages:,}")

        with col5:
            st.metric(label="🖼️ Media Messages", value=f"{total_media:,}")

        # Additional detailed metrics
        st.subheader("📈 Detailed Analytics")

        # Calculate additional stats
        first_message_date = chat_df["datetime"].min()
        last_message_date = chat_df["datetime"].max()
        conversation_days = (last_message_date - first_message_date).days + 1
        avg_messages_per_day = (
            total_messages / conversation_days if conversation_days > 0 else 0
        )

        # Character statistics
        total_chars = chat_df["text_length"].sum()
        avg_message_length = total_chars / total_messages if total_messages > 0 else 0

        # Activity patterns
        most_active_hour = (
            chat_df["hour"].mode().iloc[0]
            if not chat_df["hour"].mode().empty
            else "N/A"
        )
        most_active_day = (
            chat_df["weekday"].mode().iloc[0]
            if not chat_df["weekday"].mode().empty
            else "N/A"
        )

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric(
                label="📅 Conversation Span",
                value=f"{conversation_days} days",
                help="Total days between first and last message",
            )

        with col2:
            st.metric(
                label="📊 Avg Messages/Day",
                value=f"{avg_messages_per_day:.1f}",
                help="Average messages per day in this conversation",
            )

        with col3:
            if avg_call_duration > 0:
                minutes = int(avg_call_duration // 60)
                seconds = int(avg_call_duration % 60)
                duration_display = f"{minutes}:{seconds:02d}"
            else:
                duration_display = "0:00"
            st.metric(
                label="⏱️ Avg Call Duration",
                value=duration_display,
                help="Average duration of calls in this conversation",
            )

        with col4:
            st.metric(
                label="📝 Avg Message Length",
                value=f"{avg_message_length:.0f} chars",
                help="Average characters per message",
            )

        with col5:
            st.metric(
                label="⏰ Most Active Hour",
                value=f"{most_active_hour}:00" if most_active_hour != "N/A" else "N/A",
                help="Hour of day with most messages",
            )

        with col6:
            st.metric(
                label="📅 Most Active Day",
                value=most_active_day,
                help="Day of week with most messages",
            )

        # Create full timeline spanning entire Telegram usage period
        st.subheader("📅 Message Timeline")

        # Get the full date range from all messages
        full_start_date = private_chats_df["datetime"].min()
        full_end_date = private_chats_df["datetime"].max()

        # Create timeline with full date range
        chat_timeline = create_full_timeline(
            chat_df, full_start_date, full_end_date, selected_chat_name
        )
        st.plotly_chart(chat_timeline, width="stretch")

        # Participant Activity Distribution
        has_message_participants = (
            "from" in chat_df.columns and not chat_df["from"].isna().all()
        )
        has_call_participants = (
            "actor" in chat_df.columns and not chat_df["actor"].isna().all()
        )

        if has_message_participants or has_call_participants:
            st.subheader("👥 Participant Activity Distribution")

            col1, col2 = st.columns(2)

            with col1:
                # Calculate character counts per participant
                if has_message_participants:
                    participant_stats = (
                        chat_df[chat_df["from"].notna()]
                        .groupby("from")
                        .agg({"text_length": "sum", "message_id": "count"})
                        .reset_index()
                    )

                    participant_stats.columns = [
                        "participant",
                        "total_characters",
                        "message_count",
                    ]

                    if not participant_stats.empty:
                        # Character distribution pie chart
                        fig_chars = px.pie(
                            participant_stats,
                            values="total_characters",
                            names="participant",
                            title="Character Distribution by Participant",
                            color_discrete_sequence=px.colors.qualitative.Set3,
                        )
                        fig_chars.update_traces(
                            textposition="inside", textinfo="percent+label"
                        )
                        fig_chars.update_layout(height=400)
                        st.plotly_chart(fig_chars, width="stretch")
                    else:
                        st.info("No participant data available for messaging analysis")
                else:
                    st.info(
                        "No regular messages found - this chat only contains calls/service messages"
                    )

            with col2:
                # Call Patterns Analysis
                if has_call_participants:
                    call_data = chat_df[chat_df["action"] == "phone_call"]

                    if not call_data.empty:
                        # Filter out null/empty values and calculate calls per participant
                        valid_call_data = call_data[
                            call_data["actor"].notna() & (call_data["actor"] != "")
                        ]

                        if not valid_call_data.empty:
                            call_stats = (
                                valid_call_data["actor"].value_counts().reset_index()
                            )
                            call_stats.columns = ["participant", "call_count"]

                            # Call distribution pie chart with consistent colors
                            fig_calls = px.pie(
                                call_stats,
                                values="call_count",
                                names="participant",
                                title="Call Initiation by Participant",
                                color_discrete_sequence=px.colors.qualitative.Set3,
                            )
                            fig_calls.update_traces(
                                textposition="inside", textinfo="percent+label"
                            )
                            fig_calls.update_layout(height=400)
                            st.plotly_chart(fig_calls, width="stretch")
                        else:
                            st.info("All call records have unknown/empty initiators")
                    else:
                        st.info("No call data found for this conversation")
                else:
                    st.info("No call participants found in this conversation")
        else:
            st.info(
                "No participant information available for activity distribution analysis"
            )

        # Content Analysis in two columns
        col1, col2 = st.columns(2)

        with col1:
            # Media type breakdown
            if total_media > 0:
                st.subheader("🎭 Media Content Analysis")
                media_types = chat_df[chat_df["media_type"].notna()][
                    "media_type"
                ].value_counts()

                if not media_types.empty:
                    fig_media = px.pie(
                        values=media_types.values,
                        names=media_types.index,
                        title="Media Types Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    fig_media.update_traces(
                        textposition="inside", textinfo="percent+label"
                    )
                    fig_media.update_layout(height=400)
                    st.plotly_chart(fig_media, width="stretch")
            else:
                st.subheader("🎭 Media Content Analysis")
                st.info("No media messages found in this conversation")

        with col2:
            # Word count distribution (if we have text data)
            if "text_length" in chat_df.columns and chat_df["text_length"].sum() > 0:
                st.subheader("📝 Message Length Distribution")

                # Create bins for message lengths
                bins = [0, 10, 50, 100, 200, 500, float("inf")]
                labels = [
                    "Very Short (0-10)",
                    "Short (11-50)",
                    "Medium (51-100)",
                    "Long (101-200)",
                    "Very Long (201-500)",
                    "Extremely Long (500+)",
                ]

                chat_df_copy = chat_df.copy()
                chat_df_copy["length_category"] = pd.cut(
                    chat_df_copy["text_length"], bins=bins, labels=labels, right=True
                )
                length_dist = (
                    chat_df_copy["length_category"].value_counts().sort_index()
                )

                fig_length = px.bar(
                    x=length_dist.index,
                    y=length_dist.values,
                    title="Message Length Categories",
                    labels={"x": "Message Length Category", "y": "Number of Messages"},
                    color=length_dist.values,
                    color_continuous_scale="blues",
                )
                fig_length.update_layout(height=400, xaxis_tickangle=45)
                st.plotly_chart(fig_length, width="stretch")
            else:
                st.subheader("📝 Message Length Distribution")
                st.info("No text length data available")

        # Activity heatmap
        st.subheader("🔥 Activity Heatmap")
        heatmap_fig = create_activity_heatmap(chat_df)
        st.plotly_chart(heatmap_fig, width="stretch")

        # Word Cloud
        st.subheader("☁️ Word Cloud")
        create_word_cloud(chat_df, selected_chat_name)


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


def render_group_insights(messages_df):
    """Render the Group Insights tab"""

    if messages_df is None or messages_df.empty:
        st.warning("No messages data available.")
        return

    # Filter for group chats only (include all group types)
    group_messages = messages_df[
        messages_df["chat_type"].isin(["private_group", "private_supergroup"])
    ].copy()

    if group_messages.empty:
        st.warning("No group chats found in your data.")
        st.info(
            "This analysis only works with group conversations (not private chats)."
        )
        return

    st.header("Group Insights")

    # Get unique combinations of chat name and ID to handle duplicate names
    chat_options = group_messages[["chat_name", "chat_id"]].drop_duplicates()

    # Create display names that show both name and ID for disambiguation
    chat_display_options = []
    chat_lookup = {}

    for _, row in chat_options.iterrows():
        chat_name = row["chat_name"]
        chat_id = row["chat_id"]
        display_name = f"{chat_name} (ID: {chat_id})"
        chat_display_options.append(display_name)
        chat_lookup[display_name] = (chat_name, chat_id)

    if len(chat_display_options) == 0:
        st.warning("No valid group chats found in the data.")
        return

    # Group selector with name + ID display
    selected_display = st.selectbox(
        "Select a group", options=chat_display_options, key="group_selector"
    )

    if selected_display:
        # Get the actual chat name and ID from the selection
        selected_chat_name, selected_chat_id = chat_lookup[selected_display]

        # Filter messages for the selected group using both name and ID for accuracy
        chat_df = group_messages[
            (group_messages["chat_name"] == selected_chat_name)
            & (group_messages["chat_id"] == selected_chat_id)
        ]

        if chat_df.empty:
            st.warning("No messages found for this group.")
            return

        # Basic stats for the group
        total_messages = len(chat_df)
        calls_df = chat_df[chat_df["action"] == "phone_call"]
        total_calls = len(calls_df)
        avg_call_duration = (
            calls_df["duration_seconds"].mean()
            if len(calls_df) > 0 and "duration_seconds" in calls_df.columns
            else 0
        )
        # Media statistics - calculate once
        media_df = chat_df[chat_df["media_type"].notna()]
        total_media = len(media_df)
        voice_messages = len(media_df[media_df["media_type"] == "voice_message"])
        video_messages = len(media_df[media_df["media_type"] == "video_message"])
        unique_participants = chat_df["from"].nunique()

        # Key Metrics
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric(label="👥 Participants", value=f"{unique_participants:,}")

        with col2:
            st.metric(label="💬 Total Messages", value=f"{total_messages:,}")

        with col3:
            st.metric(label="📞 Total Calls", value=f"{total_calls:,}")

        with col4:
            st.metric(label="🎵 Voice Messages", value=f"{voice_messages:,}")

        with col5:
            st.metric(label="🎥 Video Messages", value=f"{video_messages:,}")

        with col6:
            st.metric(label="🖼️ Media Messages", value=f"{total_media:,}")

        # Additional Key Metrics row
        st.subheader("📈 Detailed Analytics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            # Calculate conversation span
            first_message = chat_df["datetime"].min()
            last_message = chat_df["datetime"].max()
            total_days = (last_message - first_message).days + 1
            avg_messages_per_day = total_messages / max(total_days, 1)
            st.metric(
                label="📅 Messages/Day",
                value=f"{avg_messages_per_day:.1f}",
                help=f"Conversation span: {total_days} days",
            )

        with col2:
            # Most active hour
            if "hour" in chat_df.columns:
                hourly_activity = chat_df.groupby("hour").size()
                most_active_hour = (
                    hourly_activity.idxmax() if not hourly_activity.empty else 0
                )
                st.metric(
                    label="🕐 Most Active Hour",
                    value=f"{most_active_hour}:00",
                    help="Hour with most messages",
                )
            else:
                st.metric(label="🕐 Most Active Hour", value="N/A")

        with col3:
            # Most active weekday
            if "weekday" in chat_df.columns:
                daily_activity = chat_df.groupby("weekday").size()
                most_active_day = (
                    daily_activity.idxmax() if not daily_activity.empty else "Unknown"
                )
                st.metric(
                    label="📆 Most Active Day",
                    value=most_active_day,
                    help="Day of week with most messages",
                )
            else:
                st.metric(label="📆 Most Active Day", value="N/A")

        with col4:
            # Average message length
            avg_message_length = (
                chat_df["text_length"].mean() if len(chat_df) > 0 else 0
            )
            st.metric(
                label="📏 Avg Message Length",
                value=f"{avg_message_length:.1f}",
                help="Average characters per message",
            )

        with col5:
            # Average call duration
            if avg_call_duration > 0:
                minutes = int(avg_call_duration // 60)
                seconds = int(avg_call_duration % 60)
                duration_display = f"{minutes}:{seconds:02d}"
            else:
                duration_display = "0:00"
            st.metric(
                label="⏱️ Avg Call Duration",
                value=duration_display,
                help="Average duration of group calls",
            )

        # Create full timeline spanning entire period
        st.subheader("📅 Message Timeline")

        # Get the full date range from chat messages
        full_start_date = chat_df["datetime"].min()
        full_end_date = chat_df["datetime"].max()

        # Create timeline with full date range
        group_timeline = create_full_timeline(
            chat_df, full_start_date, full_end_date, selected_chat_name
        )
        st.plotly_chart(group_timeline, width="stretch")

        col1, col2 = st.columns(2)

        with col1:
            # Participant Activity Distribution
            st.subheader("👥 Participant Activity Distribution")
            # Message count per participant (pie chart)
            has_message_participants = (
                "from" in chat_df.columns and not chat_df["from"].isna().all()
            )

            if has_message_participants:
                message_stats = (
                    chat_df[chat_df["from"].notna()]
                    .groupby("from")
                    .size()
                    .reset_index(name="message_count")
                    .sort_values("message_count", ascending=False)
                    .head(10)
                )

                if not message_stats.empty:
                    fig = px.pie(
                        message_stats,
                        values="message_count",
                        names="from",
                        title="Participant Activity (by Messages)",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    fig.update_traces(textposition="inside", textinfo="percent+label")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No participant data available")
            else:
                st.info("No participant data available")

        with col2:
            # Media Analysis
            st.subheader("🎭 Media Content Analysis")
            if total_media > 0:
                media_df = chat_df[chat_df["media_type"].notna()]
                media_counts = media_df["media_type"].value_counts().reset_index()
                media_counts.columns = ["media_type", "count"]

                fig = px.pie(
                    media_counts,
                    values="count",
                    names="media_type",
                    title="Media Types Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No media messages found in this group")

        # Activity Heatmap
        st.subheader("🔥 Activity Heatmap")
        st.plotly_chart(create_activity_heatmap(chat_df), width="stretch")

        # Word Cloud
        st.subheader("☁️ Word Cloud")
        create_word_cloud(chat_df, selected_chat_name)

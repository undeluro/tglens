"""
tglens - Telegram Analytics Dashboard
A Streamlit app for analyzing Telegram dumps
"""

import streamlit as st

# ── Page functions ───────────────────────────────────────────────────────


def analytics_page():
    """Analytics page — overview, contact insights, and group insights."""
    from src.visualizations import (
        render_general_overview,
        render_contact_analysis,
        render_group_insights,
    )

    messages = st.session_state.get("messages_df")

    if messages is None:
        st.html("""
        <div style="text-align: center;">
            <h1>Welcome to tglens! 👋</h1>
            <p>Data is processed locally and never leaves your browser.</p>
        </div>
        """)

        st.subheader("🚀 Getting Started")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **Step 1: Export Data**
            - Open Telegram Desktop (not mobile, not macOS, not web)
                - If you use macOS you can install Telegram Desktop alongside the native version
            - Go to Settings → Advanced and scroll down
            - Export Telegram data in JSON format
            """)
        with col2:
            st.markdown("""
            **Step 2: Upload File**
            - Use the file uploader in the sidebar
            - Select your exported JSON file (usually `result.json`)
            - Wait for processing to complete
            """)
        with col3:
            st.markdown("""
            **Step 3: Explore**
            - View general overview
            - Analyze chat patterns
            - Discover insights per contact
            """)

        return

    private_messages = messages[
        messages["chat_type"].isin(["personal_chat", "saved_messages"])
    ].copy()
    group_messages = messages[
        messages["chat_type"].isin(["private_group", "private_supergroup"])
    ].copy()

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 General Overview",
            "🧑‍💻 Contact Insights",
            "👥 Group Insights",
        ]
    )

    with tab1:
        render_general_overview(private_messages)

    with tab2:
        render_contact_analysis(private_messages)

    with tab3:
        render_group_insights(group_messages)


def rag_page():
    """Chat with Data page — RAG-powered Q&A over Telegram history."""
    from src.rag import render_rag_page

    messages = st.session_state.get("messages_df")
    if messages is None:
        st.header("Chat with Your Data")
        st.caption(
            "Ask questions about your Telegram history · runs fully locally via Ollama"
        )
        st.info("👈 Upload a Telegram JSON export in the sidebar to get started.")
        return

    render_rag_page(messages)


# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="tglens - Telegram Analytics",
    page_icon="📊",
    layout="wide",
)

# ── Navigation ───────────────────────────────────────────────────────────

pg = st.navigation(
    [
        st.Page(analytics_page, title="Analytics", icon="📊", default=True),
        st.Page(rag_page, title="Chat with Data", icon="💬"),
    ]
)

# ── Sidebar: file upload ─────────────────────────────────────────────────

with st.sidebar:
    from src.data_loader import load_into_df

    uploaded_file = st.file_uploader(
        "📁 Choose your Telegram JSON export file",
        type="json",
        help="Upload the JSON file exported from Telegram Desktop",
    )

    if uploaded_file is not None:
        with st.spinner("🔄 Loading and processing data..."):
            messages = load_into_df(uploaded_file)

        if messages is not None and not messages.empty:
            st.session_state.messages_df = messages

            # Balloons only on first upload / file change
            if (
                "last_uploaded_file" not in st.session_state
                or st.session_state.last_uploaded_file != uploaded_file
            ):
                st.session_state.last_uploaded_file = uploaded_file
                st.balloons()

                private_count = len(
                    messages[
                        messages["chat_type"].isin(["personal_chat", "saved_messages"])
                    ]
                )
                group_count = len(
                    messages[
                        messages["chat_type"].isin(
                            ["private_group", "private_supergroup"]
                        )
                    ]
                )

                st.toast(
                    f"✅ Loaded {len(messages):,} messages! "
                    f"({private_count:,} private, {group_count:,} group)"
                )
        else:
            st.error(
                "Something went wrong. Please ensure you uploaded a valid JSON export."
            )
            st.session_state.pop("messages_df", None)
    else:
        st.session_state.pop("messages_df", None)

# ── Run selected page ────────────────────────────────────────────────────

pg.run()

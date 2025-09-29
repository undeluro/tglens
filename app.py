"""
tglens - Telegram Analytics Dashboard
A Streamlit app for analyzing Telegram dumps
"""

import streamlit as st

from src.data_loader import load_into_df
from src.visualizations import (
    render_general_overview,
    render_contact_analysis,
    render_group_insights,
)


def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="tglens - Telegram Analytics",
        page_icon="📊",
        layout="wide",
    )


def main():
    """Main application function"""
    setup_page()

    uploaded_file = st.file_uploader(
        "📁 Choose your Telegram JSON export file",
        type="json",
        help="Upload the JSON file exported from Telegram Desktop",
    )

    # Main content area
    if uploaded_file is None:
        st.html("""
        <div style="text-align: center;">
            <h1>Welcome to tglens! 👋</h1>
            <p>Data is processed locally and never leaves your browser.</p>
        </div>
        """)

        # Instructions
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
            - Use the file uploader above
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

    # Load and process data
    with st.spinner("🔄 Loading and processing your Telegram data..."):
        messages = load_into_df(uploaded_file)

    if messages is None or messages.empty:
        st.error(
            "Something went wrong(see above). Please ensure you uploaded a valid JSON export."
        )
        return

    private_messages = messages[
        messages["chat_type"].isin(["personal_chat", "saved_messages"])
    ].copy()
    group_messages = messages[
        messages["chat_type"].isin(["private_group", "private_supergroup"])
    ].copy()

    # Show balloons only when new file is loaded (compare file objects)
    if (
        "last_uploaded_file" not in st.session_state
        or st.session_state.last_uploaded_file != uploaded_file
    ):
        st.session_state.last_uploaded_file = uploaded_file
        st.balloons()

        private_count = len(private_messages)
        group_count = len(group_messages)

        st.toast(
            f"✅ Successfully loaded {len(messages):,} messages! ({private_count:,} private, {group_count:,} group)"
        )

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 General Overview",
            "🧑‍💻 Contact Insights",
            "👥 Group Insights",
        ]
    )

    with tab1:
        render_general_overview(private_messages)
        if "initial_rerun_done" not in st.session_state:
            st.session_state.initial_rerun_done = True
            st.rerun()  # почему то без этого костыля(или это не костыль) после каждого рерана кидает в конец первого таба, взято с форума стримлита

    with tab2:
        render_contact_analysis(private_messages)

    with tab3:
        render_group_insights(group_messages)


if __name__ == "__main__":
    main()

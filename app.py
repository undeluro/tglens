"""
tglens - Telegram Analytics Dashboard
A Streamlit app for analyzing Telegram dumps
"""

import streamlit as st

from src.data_loader import TelegramDataLoader
from src.visualizations import render_general_overview, render_contact_analysis, render_group_insights


def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="tglens - Telegram Analytics",
        page_icon="📊",
        layout="wide",
    )


def load_into_df(uploaded_file):
    """Load and process the uploaded Telegram data"""
    if uploaded_file is None:
        return None

    # Use session state to cache processed data and prevent re-processing
    if (
        "messages_df" not in st.session_state
        or st.session_state.get("last_file_name") != uploaded_file.name
    ):
        with st.spinner("🔄 Loading and processing your Telegram data..."):
            data_loader = TelegramDataLoader()

            if data_loader.load_from_file(uploaded_file):
                messages_df = data_loader.extract_messages()
                if messages_df is not None:
                    st.session_state.messages_df = messages_df
                    st.session_state.last_file_name = uploaded_file.name
                    st.balloons()  # Only show balloons on successful new load
                    return messages_df
                else:
                    return None
            else:
                return None

    # Return cached data without balloons
    return st.session_state.messages_df


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
            - Select your exported JSON file
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
    messages = load_into_df(uploaded_file)

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 General Overview",
            "👥 Contact Insights", 
            "👥 Group Insights",
        ]
    )

    with tab1:
        render_general_overview(messages)
        if "initial_rerun_done" not in st.session_state:
            st.session_state.initial_rerun_done = True
            st.rerun()

    with tab2:
        render_contact_analysis(messages)

    with tab3:
        render_group_insights(messages)


if __name__ == "__main__":
    main()

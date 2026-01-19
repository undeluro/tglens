# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**tglens** is a Telegram export analytics tool that provides insights into messaging behavior. It's a Streamlit web application that runs entirely locally - user data never leaves their computer. The app analyzes Telegram JSON export files to generate statistics, visualizations, and insights about private chats and group conversations.

## Core Commands

### Running the Application
```bash
uv run streamlit run app.py
```
This starts the Streamlit web server. Users upload their Telegram JSON export file through the browser interface.

### Development Environment
```bash
# Install dependencies with dev tools
uv sync --all-groups
```

### Linting
```bash
# Run ruff linter
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## Architecture

### Entry Point
`app.py` - Main Streamlit application that:
- Sets up page configuration
- Handles file upload interface
- Splits data into private chats and group chats
- Renders three main tabs (General Overview, Contact Insights, Group Insights)

### Data Layer
`src/data_loader.py` - Handles loading and preprocessing of Telegram JSON exports:
- `load_into_df()` - Main cached function that parses JSON and returns a pandas DataFrame
- Extracts messages from all chats in the export
- Handles complex text field formats (string, list, dict)
- Adds derived datetime columns (date_only, year, month, day, hour, weekday, month_name)
- Preserves all chat types for filtering in different tabs

**Key DataFrame columns:**
- `chat_id`, `chat_name`, `chat_type` - Chat identification
- `from`, `from_id`, `actor`, `actor_id` - Message/action participants
- `date`, `datetime`, `date_unixtime` - Timing information
- `text`, `text_length` - Message content
- `action` - Service messages (e.g., "phone_call", "group_call")
- `media_type`, `file`, `duration_seconds` - Media information
- Derived: `date_only`, `year`, `month`, `day`, `hour`, `weekday`, `month_name`

### Visualization Layer
`src/visualizations.py` - Contains three main render functions for the three tabs:
- `render_general_overview()` - Tab 1: Aggregated statistics across all private chats
- `render_contact_analysis()` - Tab 2: Deep dive into individual private chat (@st.fragment for performance)
- `render_group_insights()` - Tab 3: Group chat analysis (@st.fragment for performance)

Each render function receives **pre-filtered data** from `app.py` and is responsible for:
- Computing statistics
- Creating Plotly visualizations
- Displaying metrics with `st.metric()`, charts with `st.plotly_chart()`
- Handling edge cases (empty data, missing fields)

### Utilities Layer
`src/utils.py` - Shared utility functions:
- `get_basic_stats()` - Cached function computing aggregate statistics
- `get_chat_summary()` - Cached function generating per-chat summary table
- `create_activity_heatmap()` - Hour-of-day × day-of-week heatmap using Plotly
- `create_full_timeline()` - Timeline chart spanning entire Telegram usage period with zero-filled gaps
- `create_word_cloud()` - WordCloud generation with bilingual stop words (English + Russian)
- `get_time_period_filter()` - Filter DataFrame by time period (7d, 30d, etc.)

## Data Flow

1. User uploads JSON file → `app.py` receives upload
2. `load_into_df()` parses and caches the DataFrame
3. `app.py` filters into `private_messages` and `group_messages` based on `chat_type`
4. Each tab receives its respective filtered DataFrame
5. Tabs use utility functions for statistics and visualizations
6. Streamlit caching (`@st.cache_data`) prevents redundant computation

## Chat Type Handling

The codebase filters chats by `chat_type`:
- **Private chats**: `"personal_chat"`, `"saved_messages"` → Tabs 1 & 2
- **Group chats**: `"private_group"`, `"private_supergroup"` → Tab 3

## Key Implementation Patterns

### Caching Strategy
- `@st.cache_data` on `load_into_df()` - Caches by file object, reloads only when file changes
- `@st.cache_data` on utility functions (`get_basic_stats`, `get_chat_summary`) - Caches by DataFrame hash
- `@st.fragment()` on tab render functions - Isolates tab reruns for performance

### Duplicate Chat Name Handling
Both Contact Insights and Group Insights tabs handle duplicate chat names by:
1. Creating display options with format: `"{chat_name} (ID: {chat_id})"`
2. Filtering using both `chat_name` AND `chat_id` to ensure accuracy

### Timeline Visualization
Uses `create_full_timeline()` to show complete Telegram usage history:
- Accepts full date range (start_date, end_date) from parent DataFrame
- Creates continuous date range
- Fills missing days with zero message counts
- Shows context of when conversations occurred relative to entire usage period

### Text Content Extraction
`_extract_text_content()` handles multiple Telegram JSON text formats:
- Plain string: `"text"`
- Array of objects: `[{"text": "foo"}, {"text": "bar"}]`
- Dict: `{"text": "content"}`

## Technology Stack

- **Python 3.13+** - Required minimum version
- **uv** - Package manager (preferred over pip/poetry)
- **Streamlit** - Web framework for the analytics interface
- **Pandas** - Data manipulation and analysis
- **Plotly** - Interactive charting library
- **WordCloud** - Text visualization
- **Matplotlib** - Backend for WordCloud rendering
- **Ruff** - Linter and formatter (in dev dependencies)

## Project Structure

```
tglens-dev/
├── app.py                 # Main Streamlit application entry point
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # JSON parsing and DataFrame creation
│   ├── visualizations.py  # Three tab render functions
│   └── utils.py           # Shared statistics and visualization utilities
├── pyproject.toml         # Project metadata and dependencies (uv format)
├── uv.lock                # Locked dependency versions
└── .streamlit/
    └── config.toml        # Custom Streamlit theme
```

## Development Guidelines

### When Adding New Visualizations
1. Add utility functions to `src/utils.py` if reusable across tabs
2. Add tab-specific logic directly in `src/visualizations.py` render functions
3. Use Plotly for interactive charts, Matplotlib only when necessary (e.g., WordCloud)
4. Always handle empty/missing data gracefully with `st.info()` or `st.warning()`

### When Modifying Data Loading
1. Changes to `data_loader.py` affect cached data - users must re-upload files
2. Preserve all chat types in the DataFrame; filtering happens in `app.py`
3. Keep text extraction logic in `_extract_text_content()` to handle format variations

### When Working with Streamlit
- Use `st.fragment()` decorator for expensive tab renders to isolate reruns
- Use `st.cache_data` for data processing functions
- Check `st.session_state` to track state across reruns (see balloons/toast logic in `app.py`)
- Use `st.spinner()` for long-running operations

### Code Style
- Use Ruff for linting and formatting (configured in dev dependencies)
- Follow existing patterns: type hints where helpful, docstrings for public functions
- Keep visualization logic in `visualizations.py`, not in utility functions

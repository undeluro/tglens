# tglens - Telegram export analytics

Gain valuable insights into your Telegram chats and uncover trends in your messaging behavior.

## Usage
~~‼️ **WARNING** This app will be hosted on Streamlit Community Cloud. BUT I whould definitely NOT recommend you upload any of your private info to a server you don't fully trust and know how it works(please don't do that). Even though Streamlit claims they store your data only in RAM, so it's deleted right after the end of session(which I think is probably true). Anyway, you will see a warning when using the hosted version of the app.~~

~~Obviously, the easiest workaround is to run app locally and it's **really easy** even if you haven't done any of this things before.~~

*After some thinking, it was decided to leave the app local only.*

### 🔒 Privacy first
Your data never leaves your computer. Everything runs locally.

### Step 0: Prerequisites
- **Python 3.13+** 
- **Git** 
- **uv** - Python package manager ([install here](https://docs.astral.sh/uv/getting-started/installation/))

### Step 1: Get your data
1. Open **Telegram Desktop** (not mobile, not web version, not MacOS native). You need cross-platform Qt version.
> **Note for macOS users**: You can install Telegram Desktop alongside the native version from the [official download page](https://desktop.telegram.org/)
2. Go to Settings → Advanced(and sroll down) → Export Telegram data
3. Choose JSON format and export your chats(we need only personal chats)
4. Wait for the export to complete

### Step 2: Clone the repo
```bash
git clone https://github.com/undeluro/tglens.git && cd tglens
```

### Step 3: Run the app
```bash
uv run streamlit run app.py
```

That's it.


## 🛠️ What's under the hood?

- **Streamlit** - Beautiful web interface
- **Plotly** - Interactive charts and graphs  
- **Pandas** - Data processing powerhouse
- **WordCloud** - Word visualization

## 🤝 Contributing

Found a bug? Have an idea? Feel free to open an issue or submit a pull request.

---

*Made with ❤️ for curious minds who want to understand their digital conversations better.* 
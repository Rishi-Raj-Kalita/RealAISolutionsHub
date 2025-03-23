# AI-Powered LinkedIn Summarization Podcast 🎙️

## Overview
This project automates the process of fetching, summarizing, and converting LinkedIn posts into an audio podcast. It allows users to stay updated with the most valuable LinkedIn insights without having to manually scroll through posts. The generated podcast summarizes the top 5 LinkedIn posts for the day, making it perfect for listening during a commute or free time.

# Note : 
This is not the official API for linkedin, use at your own risk.

## Features
- ✅ Fetches the top 5 recent LinkedIn posts from multiple users
- ✅ Summarizes the key points using a local Llama3.1 model
- ✅ Selects the best 5 posts for the day
- ✅ Converts the summary into a high-quality podcast
- ✅ Runs 100% locally for privacy and efficiency

## Tech Stack
- **LangGraph** → For agent orchestration
- **ChatOllama + Llama3.1** → Local LLM processing
- **ElevenLabs** → High-quality text-to-speech conversion

## How It Works
1. Input public LinkedIn profile IDs.
2. The system fetches recent posts from these profiles.
3. Summarization is done using Llama3.1.
4. The best 5 posts are selected.
5. The text is converted into a podcast using ElevenLabs.

## Installation & Setup
### Prerequisites
- Python 3.8+
- Ollama installed with Llama3.1
- ElevenLabs API key

### Steps
1. **Clone the repository**
   ```bash
    git clone https://github.com/Rishi-Raj-Kalita/RealAISolutionsHub.git
    cd RealAISolutionsHub/LinkedCastAI
   ```

2. **Run the script**
   ```bash
   streamlit run App.py
   ```

## Contributing
Feel free to raise an issue or submit a pull request if you have improvements or feature suggestions!
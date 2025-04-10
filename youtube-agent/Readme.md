```markdown
# 🎥 YouTube Transcript RAG App

This Python project extracts transcripts from YouTube videos, splits the content into chunks, embeds them into a vector database, and allows users to ask natural language questions about the video using Google's Gemini LLM.

---

## 📌 Features

- Extracts and processes YouTube video transcripts.
- Splits transcript text into manageable chunks.
- Embeds chunks using HuggingFace sentence transformers.
- Stores embeddings in a Chroma vector database.
- Uses Google's Gemini 2.0 Flash LLM to answer questions based on transcript content.
- Command-line interface to interact with the video content.

---

## 🧱 Project Structure

```bash
├── main.py                 # Main application logic
├── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/youtube-transcript-rag.git
cd youtube-transcript-rag
```

### 2. Install Requirements

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

> **Note:** Create a `requirements.txt` file with the following content:
```txt
langchain
langchain-google-genai
chromadb
sentence-transformers
```

### 3. Set Your API Key

Make sure you have access to **Google Generative AI (Gemini)**.
Replace the following in `main.py`:
```python
YOUTUBE_URL = "video URL"              # Replace with your video link
GOOGLE_API_KEY = "Your_google_AI_API"  # Replace with your Gemini API key
```

### 4. Run the App

```bash
python main.py
```

---

## 🛠️ How It Works

1. **Transcript Extraction**  
   Loads the YouTube transcript using `YoutubeLoader`.

2. **Chunking & Embedding**  
   Splits transcript into 1000-character chunks with 200-character overlap. Embeds each chunk using HuggingFace models.

3. **Vector Store**  
   Stores embeddings in a Chroma DB for similarity search.

4. **LLM Integration**  
   Gemini 2.0 Flash LLM responds to queries using only the retrieved transcript data.

5. **Interactive CLI**  
   Ask questions in real-time; type `exit` to quit.

---

## 📦 Example Usage

```
❓ Your question: What is the video mainly about?

📢 Answer:
The video discusses...
```

---

## 🧠 Tech Stack

- **LangChain** - Framework for building LLM-powered apps.
- **ChromaDB** - Local vector database.
- **HuggingFace** - Embeddings with sentence-transformers.
- **Google Generative AI** - Gemini LLM for response generation.
- **YouTube Transcript API** - Via LangChain loaders.

---

## 🔒 License

MIT License © 2025 SHIVAM
```


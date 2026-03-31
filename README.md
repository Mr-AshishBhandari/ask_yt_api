# YouTube Transcript Q&A with LangChain & HuggingFace

This project provides a FastAPI-based service that allows you to ingest YouTube video transcripts and ask questions about them using an AI assistant. The AI uses **HuggingFace language models** and **Chroma vector database** for semantic retrieval of transcript content.

---

## Features

- Fetches YouTube video transcripts automatically using `youtube_transcript_api`.
- Splits transcripts into manageable chunks for embeddings.
- Generates embeddings using HuggingFace models.
- Stores embeddings in a **Chroma** vector database.
- Allows question-answering over transcripts using **LangChain** chains.
- Fully contained in a **FastAPI** web service with `/ingest` and `/ask` endpoints.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Mr-AshishBhandari/ask_yt_api
cd <repo_directory>
```

2.Create and activate a virtual environment:
```bash 
uv init
uv venv
uv sync
```

3.Create a .env file in the project root with the following variables:
```bash
HUGGINGFACEHUB_ACCESS_TOKEN=<your_huggingface_token>
tenant=<your_chroma_tenant>
database=<your_chroma_database>
chroma_api_key=<your_chroma_api_key>
```

4.Start the FastAPI server
```bash
uvicorn main:app --reload
```
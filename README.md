# 📄 RAG-Based Document Assistant

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDFs and ask questions based on their content. The system retrieves relevant document chunks and generates accurate answers using Groq-powered LLMs.

---
## Live Demo

**Deployed Application:**  
[RAG Based Document Assistant – Streamlit App](https://rag-based-document-assistant.streamlit.app/)

---

## Features

* Upload multiple PDF documents
* Ask questions about uploaded documents
* Semantic search using vector embeddings
* Fast responses using Groq (LLaMA 3)
* Source citation (file name + page number)
* Hallucination control (answers only from context)
* Optimized retrieval (top-k relevant chunks)

---

## Architecture

```
User Query
     ↓
Streamlit UI
     ↓
Document Loader (PDFs)
     ↓
Text Splitter (Chunking)
     ↓
Embeddings (Sentence Transformers)
     ↓
FAISS Vector Store
     ↓
Retriever (Top-k chunks)
     ↓
Prompt Template
     ↓
Groq LLM (LLaMA 3)
     ↓
Answer + Source Citations
```

---

## Tech Stack

* **Frontend:** Streamlit
* **LLM:** Groq (LLaMA 3)
* **Framework:** LangChain (LCEL)
* **Embeddings:** Sentence Transformers
* **Vector DB:** FAISS
* **Language:** Python

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-based-document-assistant.git
cd rag-based-document-assistant
```

---

### 2. Create virtual environment

```bash
python -m venv rag_venv
source rag_venv/bin/activate   # Mac/Linux
rag_venv\Scripts\activate      # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add API Key

Create a `.env` file in root:

```env
GROQ_API_KEY=your_api_key_here
```

---

### 5. Run the app

```bash
streamlit run app.py
```

---

## Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to Streamlit Cloud
3. Deploy app using `app.py`
4. Add secret:

```toml
GROQ_API_KEY="your_api_key_here"
```

---

## Project Structure

```
rag-based-document-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── rag/
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── rag_chain.py
│
└── uploads/
```

---

## How It Works

1. PDFs are uploaded via Streamlit UI
2. Documents are split into chunks
3. Chunks are converted into embeddings
4. Stored in FAISS vector database
5. User query is matched with relevant chunks
6. Retrieved context is passed to Groq LLM
7. Answer is generated with source references

---

## Guardrails

* Model is restricted to provided context
* Returns fallback if answer not found
* Prevents hallucinated responses

---

## Example Queries

* "What is the leave policy?"
* "Summarize the uploaded document"
* "What are workplace safety rules?"

---

## Future Improvements

* Conversational memory
* Hybrid search (BM25 + vector)
* UI enhancements
* Multi-language support

---
## Author

Tanya Mishra

---

⭐ If you like this project, give it a star!

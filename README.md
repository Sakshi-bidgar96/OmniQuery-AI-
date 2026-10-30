# ✨ OmniQuery AI – All-in-One AI Knowledge Assistant

## 📌 Project Overview

OmniQuery AI is an AI-powered knowledge assistant developed using Python and Streamlit. It helps users get answers to general questions, search the web, and retrieve information from uploaded PDF documents.

The project uses Retrieval-Augmented Generation (RAG) and Knowledge Graph technology to improve information retrieval and provide context-aware answers.

## 🚀 Features

- 🤖 AI-powered chatbot
- 📄 PDF document upload and question answering
- 🔍 Online web search
- 🧠 Knowledge Graph integration using Neo4j
- 📚 Document processing and retrieval using ChromaDB
- 💬 Chat history management
- ⚡ Interactive user interface using Streamlit

## 🛠️ Technologies Used

- Python
- Streamlit
- Google Gemini API
- ChromaDB
- Neo4j
- PyPDF
- DuckDuckGo Search
- SQLite

## 📂 Project Structure

```text
OmniQuery-AI/
│
├── app.py
├── document_loader.py
├── graph_store.py
├── requirements.txt
├── test_gemini.py
├── test_neo4j.py
├── graph-rag-assistant/
├── my chat bot/
├── src/
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sakshi-bidgar96/OmniQuery-AI-.git
```

### 2. Open Project Folder

```bash
cd OmniQuery-AI-
```

### 3. Create Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure API Key

Create a `.env` file and add your Google Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

Never upload your actual API key to GitHub.

### 7. Run the Application

```bash
streamlit run app.py
```

## 🎯 Project Objectives

- To develop an intelligent AI-based question-answering system.
- To provide answers from uploaded PDF documents.
- To integrate web search for additional information.
- To organize information using Knowledge Graph technology.
- To create a user-friendly AI assistant.

## 👩‍💻 Developer

**Sakshi Dhondiram Bidgar**

GitHub: [Sakshi-bidgar96](https://github.com/Sakshi-bidgar96)

## 📌 Conclusion

OmniQuery AI combines Artificial Intelligence, RAG, and Knowledge Graph technology to create an intelligent assistant for information retrieval and question answering.

---

⭐ If you find this project interesting, consider giving it a star!

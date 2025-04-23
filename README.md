# PDF Question Answering App with LLM Integration

This project is a Flask-based web application that allows users to upload PDF documents and ask questions based on the content. It uses LangChain, FAISS for vector storage, and a locally hosted LLaMA-based language model for accurate, document-aware responses.

## Features

-  Upload a PDF and extract its text.
-  Chunk and embed the text using `sentence-transformers`.
-  Store embeddings in a FAISS vector database.
-  Query the document using a conversational retrieval chain with LLaMA.cpp.
-  Ask questions and receive precise, context-based answers.

---

## 🛠 Tech Stack

- **Flask** – Web server
- **LangChain** – LLM chains and memory
- **FAISS** – Vector database
- **LLaMA.cpp** – Local inference of a quantized LLM
- **PyPDF2** – PDF parsing
- **sentence-transformers** – Embedding generation
- **HuggingFace Embeddings** – For semantic search
- **CORS** – Cross-origin requests handling

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pdf-qa-app.git
cd pdf-qa-app

# Create and activate virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

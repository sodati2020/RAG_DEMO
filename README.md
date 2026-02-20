## Simple RAG Demo

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using OpenAI.

Flow:

1. Load Kaiser policy PDFs
2. Extract paragraphs and tables using pdfplumber
3. Generate embeddings for document chunks
4. Embed the user question
5. Retrieve top similar chunks using cosine similarity
6. Send retrieved context to GPT-4o
7. Return grounded answer

This prevents hallucination and ensures answers come directly from the policy documents.

# pick a folder:
cd ~/projects

# clone (replace with your repo URL)
git clone https://github.com/sodati2020/RAG_DEMO.git
cd RAG_DEMO
# create venv
python3 -m venv .venv

# activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
Then create .env and run
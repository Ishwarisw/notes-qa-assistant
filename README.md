# Notes Q&A Assistant

A command-line tool that answers questions from a PDF using RAG
(Retrieval-Augmented Generation). It answers only from your document and
says so when the answer is not there.

## How it works
1. `reader.py` extracts text from the PDF.
2. `chunker.py` cleans the text and splits it into overlapping chunks.
3. `embed_retriever.py` turns chunks and questions into embeddings and finds the closest chunks.
4. `answer.py` sends those chunks and the question to a Gemini model, with retries and model fallback.

## Setup
```bash
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
python -m pip install -r requirements.txt
```
Create a `.env` file containing `GEMINI_API_KEY=your-key`. Put your PDF in the
project folder as `sample.pdf`, then run:
```bash
python answer.py
```

## Tests
```bash
python -m pytest -v
```

## What I learned
I started with TF-IDF keyword search. It failed on "How do I handle biased
hiring?" because my notes say "Recruitment AI favors male candidates", with
no shared words. Switching to embeddings fixed this, and the tool even handled a
typo ("baised"). Cleaning junk lines (bullets, stray numbers) and chunking
by lines also improved retrieval.

## Limitations
- Works on text PDFs only (no scanned images).
- Retrieves from one PDF at a time.
- Quality depends on chunking and the similarity threshold (`MIN_SCORE`).

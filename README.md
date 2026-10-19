# Notes Q&A Assistant

A command-line tool that answers questions from a PDF using RAG
(Retrieval-Augmented Generation). It answers only from your document, shows
which parts of the document it used, and says so when the answer is not there.

## Demo
```text
$ python answer.py sample.pdf
Loaded embeddings from cache.
Loaded 13 chunks from sample.pdf.

Ask a question (or type 'quit'): What is fairness?

Equitable treatment of individuals/groups; mitigate bias.

Sources:
  - 0.70 | AI Ethics & Trustworthy AI – Complete Study Guide
  - 0.68 | Protected attributes Characteristics to avoid for fairness Race, gender , age
```

## Features
- Works on any text PDF: `python answer.py your-file.pdf`
- Embedding-based search, so it finds relevant text even without matching words
- Shows the source chunks and similarity scores for each answer
- Admits when the answer is not in the document
- Caches embeddings, so later runs start instantly and use fewer API calls
- Retries temporary API errors and falls back to other models

## How it works
1. `reader.py` extracts text from the PDF.
2. `chunker.py` cleans the text and splits it into chunks.
3. `embed_retriever.py` turns chunks and questions into embeddings and finds the closest chunks.
4. `cache.py` saves embeddings to disk so they are not recomputed.
5. `answer.py` sends the best chunks and the question to a Gemini model.

## Setup
```bash
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
python -m pip install -r requirements.txt
```
Create a `.env` file containing `GEMINI_API_KEY=your-key`, then run:
```bash
python answer.py your-file.pdf
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
by lines also improved retrieval. When the API returned 503 errors, I added
retries and model fallback.

## Limitations
- Works on text PDFs only (no scanned images).
- Handles one PDF at a time.
- Quality depends on chunking and the similarity threshold (`MIN_SCORE`).

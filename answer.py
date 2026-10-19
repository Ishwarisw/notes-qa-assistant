import os
import sys
import time
from dotenv import load_dotenv
from google import genai

from reader import read_pdf
from chunker import chunk_by_lines
from embed_retriever import search_embeddings
from cache import get_index

MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
DEFAULT_PDF = "sample.pdf"
TOP_K = 4
MIN_SCORE = 0.40
NOT_FOUND = "I could not find this in the notes."
RETRY_WORDS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded")


def with_retry(func, attempts=4):
    """Call func(); on temporary errors wait and try again."""
    delay = 2
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as e:
            temporary = any(word in str(e) for word in RETRY_WORDS)
            if not temporary or attempt == attempts:
                raise
            print(f"  (busy, retrying in {delay}s... attempt {attempt}/{attempts})")
            time.sleep(delay)
            delay *= 2


def generate_answer(client, prompt):
    """Try each model in turn until one works."""
    last_error = None
    for model in MODELS:
        try:
            response = with_retry(
                lambda: client.models.generate_content(model=model, contents=prompt)
            )
            return response.text
        except Exception as e:
            last_error = e
            print(f"  (model {model} failed, trying next one)")
    raise last_error


def build_prompt(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    return (
        "You are a study assistant. Answer the question using ONLY the "
        "context below.\n"
        f'If the answer is not in the context, reply exactly: "{NOT_FOUND}"\n'
        "Keep the answer short and clear.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


def answer_question(question, chunks, matrix, client):
    results = with_retry(
        lambda: search_embeddings(client, question, chunks, matrix, top_k=TOP_K)
    )
    relevant = [(chunk, score) for chunk, score in results if score >= MIN_SCORE]
    if not relevant:
        return NOT_FOUND, []
    prompt = build_prompt(question, [chunk for chunk, _ in relevant])
    answer = generate_answer(client, prompt)
    sources = [f"{score:.2f} | {chunk.splitlines()[0]}" for chunk, score in relevant]
    return answer, sources


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    if not os.path.exists(pdf_path):
        raise SystemExit(f"PDF not found: {pdf_path}")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Check your .env file.")

    client = genai.Client(api_key=api_key)
    chunks = chunk_by_lines(read_pdf(pdf_path))
    matrix = with_retry(lambda: get_index(client, chunks))
    print(f"Loaded {len(chunks)} chunks from {pdf_path}.")

    while True:
        question = input("\nAsk a question (or type 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        try:
            answer, sources = answer_question(question, chunks, matrix, client)
            print("\n" + answer)
            if sources and answer.strip() != NOT_FOUND:
                print("\nSources:")
                for s in sources:
                    print("  -", s)
        except Exception as e:
            print("Something went wrong:", e)


if __name__ == "__main__":
    main()

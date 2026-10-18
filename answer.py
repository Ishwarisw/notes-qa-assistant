import os
from dotenv import load_dotenv
from google import genai

from reader import read_pdf
from chunker import chunk_by_lines
from retriever import build_index, search

MODEL = "gemini-flash-latest"
PDF_PATH = "sample.pdf"
TOP_K = 13
MIN_SCORE = 0.05
NOT_FOUND = "I could not find this in the notes."


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


def answer_question(question, chunks, vectorizer, matrix, client):
    results = search(question, chunks, vectorizer, matrix, top_k=TOP_K)
    relevant = [chunk for chunk, score in results if score >= MIN_SCORE]
    if not relevant:
        return NOT_FOUND
    prompt = build_prompt(question, relevant)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Check your .env file.")

    client = genai.Client(api_key=api_key)
    chunks = chunk_by_lines(read_pdf(PDF_PATH))
    vectorizer, matrix = build_index(chunks)
    print(f"Loaded {len(chunks)} chunks from {PDF_PATH}.")

    while True:
        question = input("\nAsk a question (or type 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        try:
            print("\n" + answer_question(question, chunks, vectorizer, matrix, client))
        except Exception as e:
            print("Something went wrong:", e)


if __name__ == "__main__":
    main()

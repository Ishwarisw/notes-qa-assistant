import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"


def embed_texts(client, texts, task_type):
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    vectors = np.array([e.values for e in response.embeddings], dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def build_embedding_index(client, chunks):
    return embed_texts(client, chunks, "RETRIEVAL_DOCUMENT")


def search_embeddings(client, question, chunks, matrix, top_k=4):
    q = embed_texts(client, [question], "RETRIEVAL_QUERY")[0]
    scores = matrix @ q
    best = scores.argsort()[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in best]


if __name__ == "__main__":
    from reader import read_pdf
    from chunker import chunk_by_lines

    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    chunks = chunk_by_lines(read_pdf("sample.pdf"))
    matrix = build_embedding_index(client, chunks)

    questions = [
        "What is fairness?",
        "How do I handle biased hiring?",
        "What protects student data?",
    ]
    for question in questions:
        print("\nQ:", question)
        for chunk, score in search_embeddings(client, question, chunks, matrix, top_k=3):
            print(f"  {score:.3f} | {chunk.splitlines()[0]}")

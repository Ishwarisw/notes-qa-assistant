import sys
from reader import read_pdf
from chunker import chunk_by_lines
from retriever import build_index, search

question = " ".join(sys.argv[1:])
chunks = chunk_by_lines(read_pdf("sample.pdf"))
vectorizer, matrix = build_index(chunks)

for chunk, score in search(question, chunks, vectorizer, matrix, top_k=4):
    first_line = chunk.splitlines()[0]
    print(f"{score:.3f} | {first_line}")

def chunk_text(text, chunk_size=500, overlap=100):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    from reader import read_pdf

    text = read_pdf("sample.pdf")
    chunks = chunk_text(text)
    print("Number of chunks:", len(chunks))
    print("--- First chunk ---")
    print(chunks[0])
    print("--- Second chunk ---")
    print(chunks[1])

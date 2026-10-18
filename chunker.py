import re


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


def clean_lines(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # skip lines that contain only bullets, digits, dots or spaces
        if re.fullmatch(r"[•\d\.\s]+", line):
            continue
        lines.append(line)
    return lines


def chunk_by_lines(text, max_chars=500, overlap_lines=1):
    chunks = []
    current = []
    size = 0
    for line in clean_lines(text):
        if size + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            current = current[-overlap_lines:] if overlap_lines else []
            size = sum(len(l) + 1 for l in current)
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


if __name__ == "__main__":
    from reader import read_pdf

    chunks = chunk_by_lines(read_pdf("sample.pdf"))
    print("Number of chunks:", len(chunks))
    print("--- First chunk ---")
    print(chunks[0])
    print("--- Second chunk ---")
    print(chunks[1])

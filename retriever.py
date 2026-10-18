from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_index(chunks):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def search(question, chunks, vectorizer, matrix, top_k=3):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix).flatten()
    best = scores.argsort()[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in best]


if __name__ == "__main__":
    from reader import read_pdf
    from chunker import chunk_text

    text = read_pdf("sample.pdf")
    chunks = chunk_text(text)
    vectorizer, matrix = build_index(chunks)

    question = "What is robustness in AI?"
    for chunk, score in search(question, chunks, vectorizer, matrix):
        print(f"Score: {score:.3f}")
        print(chunk)
        print("-----")

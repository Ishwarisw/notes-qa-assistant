import hashlib
import os
import numpy as np

from embed_retriever import EMBED_MODEL, build_embedding_index

CACHE_DIR = ".cache"


def get_index(client, chunks):
    """Load saved embeddings if the same chunks were embedded before."""
    fingerprint = EMBED_MODEL + "\n" + "\n".join(chunks)
    key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"{key}.npy")

    if os.path.exists(path):
        print("Loaded embeddings from cache.")
        return np.load(path)

    matrix = build_embedding_index(client, chunks)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, matrix)
    print("Embeddings created and saved to cache.")
    return matrix

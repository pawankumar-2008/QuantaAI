from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rag_search(
    query: str,
    docs: list[dict[str, Any]],
    k: int = 3,
) -> list[dict[str, Any]]:
    if not docs:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    document_texts = [document["text"] for document in docs]
    vectors = vectorizer.fit_transform([*document_texts, query])
    similarities = cosine_similarity(
        vectors[-1],
        vectors[:-1],
    )[0]
    return [
        docs[index] | {"score": float(similarities[index])}
        for index in similarities.argsort()[::-1][:k]
        if similarities[index] > 0.05
    ]

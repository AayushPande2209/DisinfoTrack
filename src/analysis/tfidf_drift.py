from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_drift(canonical: str, variant: str) -> float:
    """Compute TF-IDF cosine drift between two claim strings.

    Drift is defined as (1 - cosine_similarity) * 100, expressed as a
    percentage. A score of 0 means identical vocabulary; 100 means no
    shared terms at all.

    Short or empty claims get a drift of 0.0 rather than crashing.
    """
    canonical = canonical.strip()
    variant = variant.strip()
    if not canonical or not variant:
        return 0.0

    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        tfidf = vec.fit_transform([canonical, variant])
        sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        return round(float((1.0 - sim) * 100), 2)
    except ValueError:
        # Vectorizer can fail when vocabulary is empty after tokenisation
        return 0.0


def batch_drift(pairs: list[tuple[str, str]]) -> list[float]:
    """Compute drift for a list of (canonical, variant) string pairs."""
    return [compute_tfidf_drift(c, v) for c, v in pairs]

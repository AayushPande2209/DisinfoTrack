from typing import Optional
import numpy as np

from .semantic_similarity import encode_claims, compute_similarity
from .triple_extraction import extract_triple


def _embed_component(text: Optional[str]) -> Optional[np.ndarray]:
    """Encode a single string component; returns None if text is None or empty."""
    if not text:
        return None
    return encode_claims([text])[0]


def _triple_similarity(
    triple_a: tuple[Optional[str], Optional[str], Optional[str]],
    triple_b: tuple[Optional[str], Optional[str], Optional[str]],
) -> Optional[float]:
    """Compute average cosine similarity across matched triple components.

    Skips components where either side is None. Returns None if no components
    can be compared.
    """
    scores = []
    for comp_a, comp_b in zip(triple_a, triple_b):
        if comp_a is None or comp_b is None:
            continue
        emb_a = _embed_component(comp_a)
        emb_b = _embed_component(comp_b)
        if emb_a is not None and emb_b is not None:
            scores.append(float(compute_similarity(emb_a, emb_b)))

    return float(np.mean(scores)) if scores else None


def _confidence_label(score: float, threshold: float) -> str:
    distance = abs(score - threshold)
    if distance > 0.15:
        return "HIGH"
    if distance > 0.08:
        return "MEDIUM"
    return "LOW"


def compare_claims(
    claim_a: str,
    claim_b: str,
    threshold: float = 0.75,
    embedding_weight: float = 0.60,
    triple_weight: float = 0.40,
) -> dict:
    """Compare two claims and return a verdict with component scores.

    Args:
        claim_a: First claim text.
        claim_b: Second claim text.
        threshold: Combined score cutoff for EQUIVALENT verdict. Configurable
            because optimal value depends on human-labeled calibration data.
        embedding_weight: Weight for full-sentence embedding similarity.
        triple_weight: Weight for triple-component similarity.

    Returns:
        Dict with keys: embedding_similarity, triple_similarity,
        combined_score, verdict, confidence.
    """
    if not claim_a or not claim_b:
        return {
            "embedding_similarity": 0.0,
            "triple_similarity": None,
            "combined_score": 0.0,
            "verdict": "DIFFERENT",
            "confidence": "HIGH",
        }

    emb_a, emb_b = encode_claims([claim_a, claim_b])
    embedding_sim = float(compute_similarity(emb_a, emb_b))

    triple_a = extract_triple(claim_a)
    triple_b = extract_triple(claim_b)
    triple_sim = _triple_similarity(triple_a, triple_b)

    if triple_sim is not None:
        combined = embedding_weight * embedding_sim + triple_weight * triple_sim
    else:
        combined = embedding_sim

    verdict = "EQUIVALENT" if combined >= threshold else "DIFFERENT"
    confidence = _confidence_label(combined, threshold)

    return {
        "embedding_similarity": round(embedding_sim, 4),
        "triple_similarity": round(triple_sim, 4) if triple_sim is not None else None,
        "combined_score": round(combined, 4),
        "verdict": verdict,
        "confidence": confidence,
    }

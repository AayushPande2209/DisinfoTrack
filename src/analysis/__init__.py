from .semantic_similarity import encode_claims, compute_similarity
from .triple_extraction import extract_triple
from .claim_equivalence import compare_claims

__all__ = [
    "encode_claims",
    "compute_similarity",
    "extract_triple",
    "compare_claims",
]

import re
from typing import Optional

import numpy as np
import spacy

_nlp = spacy.load("en_core_web_sm")

# ── Word / phrase lists ────────────────────────────────────────────────────

_FORMAL_WORDS: frozenset[str] = frozenset({
    # Medical / scientific terms
    "vaccine", "pathogen", "efficacy", "immunity", "transmission", "mortality",
    # Research language
    "study", "research", "evidence", "data", "correlation",
    # Hedging language
    "suggests", "indicates", "may", "could",
})

# Multi-word first so partial matches aren't double-counted.  Single words
# come last and are matched as whole tokens.
_AUTHORITY_PHRASES: list[str] = [
    "scientists say",
    "studies show",
    "according to",
    "experts",
    "researchers",
    "government",
    "cdc",
    "who",
]

_EMOTION_WORDS: frozenset[str] = frozenset({
    "kill", "destroy", "deadly", "dangerous", "poison",
    "hoax", "lie", "conspiracy", "evil", "corrupt",
    "fake", "hidden", "secret", "agenda",
})

# Regex for numbers / percentages
_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?%?\b')


def extract_features(text: str) -> dict[str, float]:
    """Extract four linguistic features from a claim string.

    Features are each normalised to [0, 1] by dividing by token count.
    Returns zeros for empty / very short input rather than crashing.

    Returns:
        Dict with keys: formality, specificity, authority, emotion.
    """
    if not text or not text.strip():
        return {"formality": 0.0, "specificity": 0.0, "authority": 0.0, "emotion": 0.0}

    doc = _nlp(text.lower())
    tokens = [t.text for t in doc if not t.is_space]
    n = len(tokens)
    if n == 0:
        return {"formality": 0.0, "specificity": 0.0, "authority": 0.0, "emotion": 0.0}

    # ── Formality ──────────────────────────────────────────────────────────
    formal_count = sum(1 for t in tokens if t in _FORMAL_WORDS)
    formality = min(formal_count / n, 1.0)

    # ── Specificity ────────────────────────────────────────────────────────
    # spaCy NER on the original-case doc for better entity recall
    doc_orig = _nlp(text)
    ner_count = len(doc_orig.ents)
    num_count = len(_NUMBER_RE.findall(text))
    specificity = min((ner_count + num_count) / n, 1.0)

    # ── Authority ──────────────────────────────────────────────────────────
    # Count how many distinct authority phrases appear in the lowercased text
    lowered = text.lower()
    authority_hits = sum(1 for phrase in _AUTHORITY_PHRASES if phrase in lowered)
    authority = min(authority_hits / n, 1.0)

    # ── Emotion ────────────────────────────────────────────────────────────
    emotion_count = sum(1 for t in tokens if t in _EMOTION_WORDS)
    emotion = min(emotion_count / n, 1.0)

    return {
        "formality": round(formality, 6),
        "specificity": round(specificity, 6),
        "authority": round(authority, 6),
        "emotion": round(emotion, 6),
    }


def feature_vector(text: str) -> np.ndarray:
    """Return features as a 1-D numpy array: [formality, specificity, authority, emotion]."""
    feats = extract_features(text)
    return np.array([feats["formality"], feats["specificity"],
                     feats["authority"], feats["emotion"]], dtype=float)


FEATURE_NAMES: list[str] = ["formality", "specificity", "authority", "emotion"]

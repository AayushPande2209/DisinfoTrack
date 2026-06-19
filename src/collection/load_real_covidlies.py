import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CSV = _PROJECT_ROOT / "COVIDLIES_real.csv"


def load_covidlies_csv(filepath: Path | str = _DEFAULT_CSV) -> pd.DataFrame:
    """Load COVIDLies dataset from CSV file.

    Returns a DataFrame with columns: claim_id, anchor_claim, variant_text,
    platform, category, origin, reliability_score.
    """
    df = pd.read_csv(filepath)
    df["anchor_claim"] = df["anchor_claim"].str.strip()
    df["variant_text"] = df["variant_text"].str.strip()
    return df


def get_variant_pairs(df: pd.DataFrame) -> list[dict]:
    """Return a list of (anchor_claim, variant_text) dicts with metadata."""
    pairs = []
    for _, row in df.iterrows():
        pairs.append({
            "claim_id": row["claim_id"],
            "canonical_claim": row["anchor_claim"],
            "variant_claim": row["variant_text"],
            "category": row["category"],
            "platform": row["platform"],
        })
    return pairs

import random
from pathlib import Path
from typing import Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"
_DEFAULT_OUTPUT = _DATA_DIR / "annotation_task.csv"

MAX_PAIRS = 100


def _sample_with_category_balance(pairs: list[dict], n: int) -> list[dict]:
    """Sample up to n pairs while maximising category representation.

    Iterates through categories round-robin until n pairs are collected or
    the pool is exhausted.
    """
    from collections import defaultdict

    by_category: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_category[p["category"]].append(p)

    # Shuffle within each category for randomness
    rng = random.Random(42)
    for cat_pairs in by_category.values():
        rng.shuffle(cat_pairs)

    # Round-robin across sorted categories so output is deterministic
    categories = sorted(by_category.keys())
    selected: list[dict] = []
    indices = {cat: 0 for cat in categories}

    while len(selected) < n:
        added_any = False
        for cat in categories:
            idx = indices[cat]
            if idx < len(by_category[cat]):
                selected.append(by_category[cat][idx])
                indices[cat] += 1
                added_any = True
                if len(selected) >= n:
                    break
        if not added_any:
            break

    return selected


def generate_annotation_csv(
    df: pd.DataFrame,
    output_path: Path = _DEFAULT_OUTPUT,
    max_pairs: int = MAX_PAIRS,
    random_seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate a CSV of claim pairs ready for human annotation.

    Args:
        df: COVIDLies DataFrame from load_covidlies_csv().
        output_path: Where to save the annotation CSV.
        max_pairs: Maximum rows to include (sampled with category balance).
        random_seed: Seed passed to random for reproducibility.

    Returns:
        The annotation DataFrame that was saved.
    """
    if random_seed is not None:
        random.seed(random_seed)

    all_pairs = []
    for _, row in df.iterrows():
        all_pairs.append({
            "claim_id": row["claim_id"],
            "canonical_claim": str(row["anchor_claim"]).strip(),
            "variant_claim": str(row["variant_text"]).strip(),
            "category": row["category"],
        })

    sampled = _sample_with_category_balance(all_pairs, max_pairs)

    annotation_rows = []
    for pair in sampled:
        annotation_rows.append({
            "claim_id": pair["claim_id"],
            "canonical_claim": pair["canonical_claim"],
            "variant_claim": pair["variant_claim"],
            "category": pair["category"],
            "annotator_label": "",       # same / different
            "annotator_confidence": "",  # 1-3
            "notes": "",
        })

    annotation_df = pd.DataFrame(annotation_rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_df.to_csv(output_path, index=False)
    print(f"Saved {len(annotation_df)} pairs to {output_path}")

    _print_summary(annotation_df)
    return annotation_df


def _print_summary(df: pd.DataFrame) -> None:
    """Print a category breakdown of the annotation task."""
    print("\nPairs per category:")
    counts = df["category"].value_counts()
    for cat, count in counts.items():
        print(f"  {cat}: {count}")
    print(f"  TOTAL: {len(df)}")

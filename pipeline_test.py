"""Phase 2 end-to-end pipeline test.

Loads 5 example pairs from COVIDLies, runs text cleaning, then claim_equivalence
on each pair and prints the full result dict. Also generates the annotation CSV.
"""

import sys
import traceback
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.collection.load_real_covidlies import load_covidlies_csv, get_variant_pairs
from src.preprocessing.text_cleaning import clean_text
from src.analysis.claim_equivalence import compare_claims
from src.evaluation.human_annotation import generate_annotation_csv


def run_pipeline_test() -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading COVIDLies dataset...")
    df = load_covidlies_csv()
    print(f"Loaded {len(df)} rows | {df['category'].nunique()} categories\n")

    pairs = get_variant_pairs(df)

    # Pick 5 pairs spread across different categories for variety
    seen_categories: set[str] = set()
    sample: list[dict] = []
    for pair in pairs:
        cat = pair["category"]
        if cat not in seen_categories:
            sample.append(pair)
            seen_categories.add(cat)
        if len(sample) == 5:
            break
    # Fallback: just take first 5 if categories exhausted early
    if len(sample) < 5:
        sample = pairs[:5]

    # ── Claim equivalence on 5 pairs ────────────────────────────────────────
    print("=" * 60)
    print("Running claim_equivalence on 5 example pairs\n")

    for i, pair in enumerate(sample, start=1):
        print(f"--- Pair {i} | category: {pair['category']} ---")
        canonical = clean_text(pair["canonical_claim"])
        variant = clean_text(pair["variant_claim"])

        print(f"  Canonical : {canonical[:120]}{'...' if len(canonical) > 120 else ''}")
        print(f"  Variant   : {variant[:120]}{'...' if len(variant) > 120 else ''}")

        try:
            result = compare_claims(canonical, variant)
            print(f"  embedding_similarity : {result['embedding_similarity']}")
            print(f"  triple_similarity    : {result['triple_similarity']}")
            print(f"  combined_score       : {result['combined_score']}")
            print(f"  verdict              : {result['verdict']}")
            print(f"  confidence           : {result['confidence']}")
        except Exception:
            print("  ERROR during compare_claims:")
            traceback.print_exc()

        print()

    # ── Generate annotation CSV ──────────────────────────────────────────────
    print("=" * 60)
    print("Generating annotation task CSV...")
    try:
        generate_annotation_csv(df)
    except Exception:
        print("ERROR generating annotation CSV:")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Phase 2 pipeline test complete.")


if __name__ == "__main__":
    run_pipeline_test()

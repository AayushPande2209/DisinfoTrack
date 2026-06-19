"""Phase 3 end-to-end pipeline test: category-specific drift prediction.

Loads COVIDLies, trains the category classifier (with LOOCV evaluation),
then for 5 diverse claim pairs: extracts linguistic features, predicts
category, computes observed TF-IDF drift, and runs anomaly analysis.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.collection.load_real_covidlies import load_covidlies_csv, get_variant_pairs
from src.preprocessing.text_cleaning import clean_text
from src.analysis.tfidf_drift import compute_tfidf_drift
from src.analysis.prediction.feature_extractor import extract_features, FEATURE_NAMES
from src.analysis.prediction.category_classifier import (
    train_and_save_classifier,
    predict_category,
    get_feature_importances,
)
from src.analysis.prediction.drift_predictor import analyze_claim_pair


def _truncate(text: str, width: int = 55) -> str:
    return text if len(text) <= width else text[:width - 3] + "..."


def run() -> None:
    # ── 1. Load dataset ───────────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 3 — Category-Specific Drift Prediction")
    print("=" * 70)
    print("\nLoading COVIDLies dataset...")
    df = load_covidlies_csv()
    print(f"Loaded {len(df)} rows | {df['claim_id'].nunique()} unique claims\n")

    # ── 2. Train classifier (LOOCV) ───────────────────────────────────────
    print("=" * 70)
    print("Training category classifier (LOOCV)")
    print("=" * 70)
    try:
        train_and_save_classifier(df)
    except Exception:
        print("ERROR during classifier training:")
        traceback.print_exc()
        return

    # ── 3. Pick 5 diverse pairs ───────────────────────────────────────────
    pairs = get_variant_pairs(df)
    seen: set[str] = set()
    sample: list[dict] = []
    for p in pairs:
        cat = p["category"]
        # Skip pairs where canonical == variant (data artifacts)
        if p["canonical_claim"].strip() == p["variant_claim"].strip():
            continue
        if cat not in seen:
            sample.append(p)
            seen.add(cat)
        if len(sample) == 5:
            break
    if len(sample) < 5:
        # Fallback: fill from the front, skipping exact duplicates
        for p in pairs:
            if p not in sample and p["canonical_claim"].strip() != p["variant_claim"].strip():
                sample.append(p)
            if len(sample) == 5:
                break

    # ── 4. Per-pair analysis table ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Claim-pair analysis (5 examples)")
    print("=" * 70)

    for i, pair in enumerate(sample, start=1):
        canonical = clean_text(pair["canonical_claim"])
        variant   = clean_text(pair["variant_claim"])
        true_cat  = pair["category"]

        print(f"\n--- Pair {i} | true category: {true_cat} ---")
        print(f"  Canonical : {_truncate(canonical)}")
        print(f"  Variant   : {_truncate(variant)}")

        try:
            # Feature extraction
            feats = extract_features(canonical)
            print(f"  Features  : formality={feats['formality']:.3f}  "
                  f"specificity={feats['specificity']:.3f}  "
                  f"authority={feats['authority']:.3f}  "
                  f"emotion={feats['emotion']:.3f}")

            # Observed drift
            observed = compute_tfidf_drift(canonical, variant)

            # Full analysis
            result = analyze_claim_pair(canonical, variant, observed)

            pred_cat   = result["category"]
            confidence = result["confidence"]
            exp_min    = result["expected_min"]
            exp_max    = result["expected_max"]
            obs        = result["observed_drift"]
            anomalous  = result["is_anomalous"]
            severity   = result["anomaly_severity"]

            match_marker = "✓" if pred_cat == true_cat else "✗"
            print(f"  Predicted : {pred_cat} ({confidence:.2%} conf) {match_marker}")
            print(f"  Drift     : observed={obs:.1f}%  expected={exp_min:.0f}–{exp_max:.0f}%")
            print(f"  Anomaly   : {'YES — ' + severity if anomalous else 'No'}")

        except Exception:
            print("  ERROR:")
            traceback.print_exc()

    # ── 5. Feature importances ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Feature importances (trained model)")
    print("=" * 70)
    try:
        importances = get_feature_importances()
        for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 50)
            print(f"  {name:<12}: {imp:.4f}  {bar}")
    except Exception:
        print("ERROR fetching feature importances:")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Phase 3 pipeline test complete.")
    print("=" * 70)


if __name__ == "__main__":
    run()

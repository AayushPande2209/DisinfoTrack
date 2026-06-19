from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
_DEFAULT_ANNOTATION = _DATA_DIR / "annotation_task.csv"


def _binary_metrics(
    y_true: list[int], y_pred: list[int]
) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Compute Fleiss' kappa for a ratings matrix (subjects x raters).

    Each cell is a category index (0 or 1 for binary annotation). Handles
    missing values encoded as -1 by excluding those rater-subject pairs.
    """
    n_subjects, n_raters = ratings_matrix.shape
    n_categories = 2  # same=1, different=0

    # Count agreements per subject
    counts = np.zeros((n_subjects, n_categories))
    valid_raters_per_subject = np.zeros(n_subjects)

    for i in range(n_subjects):
        for j in range(n_raters):
            val = ratings_matrix[i, j]
            if val >= 0:
                counts[i, int(val)] += 1
                valid_raters_per_subject[i] += 1

    n = valid_raters_per_subject  # per-subject rater counts
    N = n_subjects

    # Proportion of all assignments to each category
    total_assignments = n.sum()
    if total_assignments == 0:
        return 0.0
    p_j = counts.sum(axis=0) / total_assignments

    # Per-subject agreement
    p_i = np.zeros(N)
    for i in range(N):
        ni = n[i]
        if ni <= 1:
            p_i[i] = 0.0
        else:
            p_i[i] = (np.sum(counts[i] * (counts[i] - 1))) / (ni * (ni - 1))

    P_bar = p_i.mean()
    P_e_bar = np.sum(p_j ** 2)

    if P_e_bar == 1.0:
        return 1.0
    return (P_bar - P_e_bar) / (1.0 - P_e_bar)


def evaluate_model(
    annotation_path: Path = _DEFAULT_ANNOTATION,
    output_dir: Path = _OUTPUTS_DIR,
    thresholds: Optional[list[float]] = None,
) -> dict:
    """Validate claim_equivalence predictions against human labels.

    Loads the annotation CSV (must have annotator_label column filled in),
    runs claim_equivalence on the same pairs, computes metrics across a
    threshold sweep, plots a threshold curve, and prints the best threshold.

    Args:
        annotation_path: Path to the completed annotation CSV.
        output_dir: Directory where threshold_curve.png is saved.
        thresholds: List of thresholds to evaluate. Defaults to 0.60–0.90
            in steps of 0.05.

    Returns:
        Dict with 'best_threshold', 'best_metrics', and 'all_results'.
    """
    import matplotlib.pyplot as plt
    from src.analysis.claim_equivalence import compare_claims
    from src.preprocessing.text_cleaning import clean_text

    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.60, 0.91, 0.05)]

    annotation_path = Path(annotation_path)
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}\n"
            "Run human_annotation.generate_annotation_csv() first and fill in labels."
        )

    df = pd.read_csv(annotation_path)

    labeled = df[df["annotator_label"].isin(["same", "different"])].copy()
    if labeled.empty:
        raise ValueError(
            "No labeled rows found. Fill in the 'annotator_label' column "
            "with 'same' or 'different' before running evaluation."
        )

    y_true = (labeled["annotator_label"] == "same").astype(int).tolist()

    print(f"Computing model predictions for {len(labeled)} labeled pairs...")
    combined_scores = []
    for _, row in labeled.iterrows():
        result = compare_claims(
            clean_text(str(row["canonical_claim"])),
            clean_text(str(row["variant_claim"])),
            threshold=0.75,  # threshold here only affects verdict field, not score
        )
        combined_scores.append(result["combined_score"])

    all_results = []
    best_f1 = -1.0
    best_threshold = thresholds[0]
    best_metrics: dict = {}

    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in combined_scores]
        metrics = _binary_metrics(y_true, y_pred)
        metrics["threshold"] = t
        all_results.append(metrics)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = t
            best_metrics = metrics

    # Fleiss' kappa if multiple annotator columns present
    annotator_cols = [c for c in df.columns if c.startswith("annotator_label")]
    if len(annotator_cols) > 1:
        label_map = {"same": 1, "different": 0}
        matrix = np.full((len(labeled), len(annotator_cols)), -1, dtype=float)
        for j, col in enumerate(annotator_cols):
            for i, val in enumerate(labeled[col]):
                if val in label_map:
                    matrix[i, j] = label_map[val]
        kappa = _fleiss_kappa(matrix)
        print(f"\nFleiss' kappa (inter-annotator agreement): {kappa:.4f}")
    else:
        kappa = None

    # Threshold curve plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "threshold_curve.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xs = [r["threshold"] for r in all_results]

    axes[0].plot(xs, [r["f1"] for r in all_results], marker="o", label="F1")
    axes[0].plot(xs, [r["precision"] for r in all_results], marker="s", label="Precision")
    axes[0].plot(xs, [r["recall"] for r in all_results], marker="^", label="Recall")
    axes[0].axvline(best_threshold, color="red", linestyle="--", label=f"Best ({best_threshold})")
    axes[0].set_xlabel("Threshold")
    axes[0].set_title("Precision / Recall / F1 vs Threshold")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(xs, [r["accuracy"] for r in all_results], marker="o", color="purple")
    axes[1].axvline(best_threshold, color="red", linestyle="--")
    axes[1].set_xlabel("Threshold")
    axes[1].set_title("Accuracy vs Threshold")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nThreshold curve saved to {plot_path}")

    print(f"\nBest threshold: {best_threshold}")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  F1:        {best_metrics['f1']:.4f}")

    return {
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "all_results": all_results,
        "fleiss_kappa": kappa,
    }

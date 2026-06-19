from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from .feature_extractor import feature_vector, FEATURE_NAMES

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODEL_PATH = _PROJECT_ROOT / "models" / "category_classifier.pkl"

# Cached after first load so repeated calls to predict_category don't re-read
# the file every time.
_CACHE: Optional[dict] = None


def _build_training_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Deduplicate by claim_id, extract features, encode labels.

    Returns (X, y_encoded, label_encoder).
    """
    unique = (
        df.drop_duplicates("claim_id")[["claim_id", "anchor_claim", "category"]]
        .dropna(subset=["anchor_claim", "category"])
        .reset_index(drop=True)
    )

    X = np.array([feature_vector(str(row["anchor_claim"])) for _, row in unique.iterrows()])
    le = LabelEncoder()
    y = le.fit_transform(unique["category"].values)
    return X, y, le


def train_and_save_classifier(df: pd.DataFrame) -> RandomForestClassifier:
    """Train a Random Forest on the 62 unique canonical claims and save to disk.

    Uses Leave-One-Out CV to report honest per-class and overall accuracy.
    The final model is trained on the full dataset before saving.

    Args:
        df: COVIDLies DataFrame from load_covidlies_csv().

    Returns:
        Trained RandomForestClassifier fit on all data.
    """
    X, y, le = _build_training_data(df)
    n_samples = len(X)
    print(f"Training on {n_samples} unique canonical claims across {len(le.classes_)} categories")
    print(f"Class distribution:")
    for cls, count in zip(*np.unique(y, return_counts=True)):
        print(f"  {le.classes_[cls]}: {count}")

    # ── LOOCV ─────────────────────────────────────────────────────────────
    print(f"\nRunning Leave-One-Out CV ({n_samples} folds)...")
    loo = LeaveOneOut()
    loo_preds: list[int] = []
    loo_truths: list[int] = []

    for train_idx, test_idx in loo.split(X):
        clf_fold = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )
        clf_fold.fit(X[train_idx], y[train_idx])
        loo_preds.append(int(clf_fold.predict(X[test_idx])[0]))
        loo_truths.append(int(y[test_idx][0]))

    overall_acc = accuracy_score(loo_truths, loo_preds)
    print(f"\nLOOCV Overall accuracy: {overall_acc:.3f}")
    print("\nPer-class LOOCV accuracy:")
    # Compute per-class manually so we can show it clearly
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = [t == cls_idx for t in loo_truths]
        if not any(mask):
            continue
        cls_preds = [p for t, p in zip(loo_truths, loo_preds) if t == cls_idx]
        cls_correct = sum(p == cls_idx for p in cls_preds)
        cls_total = len(cls_preds)
        print(f"  {cls_name:<28}: {cls_correct}/{cls_total} = {cls_correct/cls_total:.2f}")

    print("\nFull classification report (LOOCV):")
    print(classification_report(
        loo_truths, loo_preds,
        target_names=le.classes_,
        zero_division=0,
    ))

    # ── Train on full dataset ──────────────────────────────────────────────
    clf_final = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    clf_final.fit(X, y)

    # ── Feature importances ────────────────────────────────────────────────
    importances = clf_final.feature_importances_
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:<12}: {imp:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf_final, "le": le}, _MODEL_PATH)
    print(f"\nModel saved to {_MODEL_PATH}")

    # Warm the in-process cache
    global _CACHE
    _CACHE = {"clf": clf_final, "le": le}

    return clf_final


def _load_model() -> dict:
    """Load model from disk (once) or return in-process cache."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {_MODEL_PATH}. "
            "Call train_and_save_classifier(df) first."
        )
    _CACHE = joblib.load(_MODEL_PATH)
    return _CACHE


def predict_category(claim_text: str) -> tuple[str, float]:
    """Predict the category of a claim using the saved Random Forest.

    Returns:
        (category_label, confidence_probability)
    """
    bundle = _load_model()
    clf: RandomForestClassifier = bundle["clf"]
    le: LabelEncoder = bundle["le"]

    vec = feature_vector(claim_text).reshape(1, -1)
    proba = clf.predict_proba(vec)[0]
    pred_idx = int(np.argmax(proba))
    return le.classes_[pred_idx], float(proba[pred_idx])


def get_feature_importances() -> dict[str, float]:
    """Return feature name → importance mapping from the saved model."""
    bundle = _load_model()
    clf: RandomForestClassifier = bundle["clf"]
    return dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist()))

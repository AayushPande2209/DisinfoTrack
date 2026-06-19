from typing import Optional

from .category_classifier import predict_category

# Expected TF-IDF drift ranges (%) per category, derived from Phase 1 analysis.
# Values are (min_pct, max_pct). Claims whose observed drift falls outside the
# max are flagged as anomalous.
DRIFT_RANGES: dict[str, tuple[float, float]] = {
    "Medical Misinformation": (65.0, 75.0),
    "Conspiracy Theories":    (74.0, 78.0),
    "Combative Efforts":      (75.0, 78.0),
    # Remaining categories: conservative estimates pending Phase 1 calibration
    "Miscellaneous":          (60.0, 75.0),
    "Government":             (70.0, 78.0),
    "Accidental leakage":     (60.0, 70.0),
    "Statistics":             (65.0, 75.0),
}

_DEFAULT_RANGE: tuple[float, float] = (60.0, 80.0)


def predict_drift(claim_text: str) -> tuple[float, float]:
    """Return the expected (min, max) drift range for a claim.

    Predicts the claim's category first, then looks up the corresponding
    drift range from DRIFT_RANGES.

    Returns:
        (expected_min_pct, expected_max_pct)
    """
    category, _ = predict_category(claim_text)
    return DRIFT_RANGES.get(category, _DEFAULT_RANGE)


def _anomaly_severity(observed: float, expected_max: float) -> str:
    overshoot = observed - expected_max
    if overshoot <= 0:
        return "NONE"
    if overshoot <= 5.0:
        return "MILD"
    return "HIGH"


def analyze_claim_pair(
    canonical: str,
    variant: str,
    observed_drift: float,
) -> dict:
    """Predict expected drift for a claim pair and flag anomalies.

    Args:
        canonical: The original / source claim text.
        variant: The variant / mutated claim text.
        observed_drift: TF-IDF drift score (0-100) already computed by the caller.

    Returns:
        Dict with keys:
            category, confidence, expected_min, expected_max,
            observed_drift, is_anomalous, anomaly_severity
    """
    category, confidence = predict_category(canonical)
    exp_min, exp_max = DRIFT_RANGES.get(category, _DEFAULT_RANGE)
    severity = _anomaly_severity(observed_drift, exp_max)
    is_anomalous = severity != "NONE"

    return {
        "category":        category,
        "confidence":      round(confidence, 4),
        "expected_min":    exp_min,
        "expected_max":    exp_max,
        "observed_drift":  round(observed_drift, 2),
        "is_anomalous":    is_anomalous,
        "anomaly_severity": severity,
    }

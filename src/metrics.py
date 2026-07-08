"""
Governance metrics:
  δ  — Manifold Drift (Frobenius norm distance in activation space)
  B  — Oversight Burden (fraction of uncertain predictions)
  H  — Mean prediction entropy
  E_sys — System Efficacy (accuracy gain per unit oversight burden)
"""

import numpy as np
from scipy import stats
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_drift(baseline_hidden: np.ndarray, fused_hidden: np.ndarray) -> float:
    """
    Normalised Frobenius norm distance between activation matrices.
    δ = ||H_fused - H_baseline||_F / ||H_baseline||_F
    """
    diff = fused_hidden - baseline_hidden
    norm_diff = np.linalg.norm(diff, "fro")
    norm_base = np.linalg.norm(baseline_hidden, "fro")
    return float(norm_diff / (norm_base + 1e-8))


def compute_performance(logits: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    """Accuracy and macro-F1 from logits."""
    preds = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(true_labels, preds)),
        "f1_macro": float(f1_score(true_labels, preds, average="macro", zero_division=0)),
    }


def compute_oversight_burden(
    logits: np.ndarray,
    confidence_threshold: float = 0.70,
) -> Dict[str, float]:
    """
    Oversight burden as the fraction of predictions where max confidence
    falls below `confidence_threshold` — those require human review.

    Also returns mean Shannon entropy of the output distribution.
    """
    probs = softmax(logits, axis=1)
    max_conf = probs.max(axis=1)

    uncertain = (max_conf < confidence_threshold).mean()

    # Shannon entropy per prediction, then averaged
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    mean_entropy = float(entropy.mean())

    return {
        "oversight_burden": float(uncertain),
        "mean_entropy": mean_entropy,
        "mean_confidence": float(max_conf.mean()),
    }


def compute_esys(
    accuracy: float,
    oversight_burden: float,
    baseline_accuracy: float = 0.50,
) -> float:
    """
    E_sys = accuracy_gain / (1 + oversight_burden)
    Accuracy gain is measured relative to random-chance baseline.
    """
    gain = max(accuracy - baseline_accuracy, 0.0)
    return float(gain / (1.0 + oversight_burden))


# ---------------------------------------------------------------------------
# Per-condition result bundle
# ---------------------------------------------------------------------------

def build_condition_metrics(
    condition_id: int,
    cartridge_ids: List[int],
    baseline_hidden: np.ndarray,
    fused_hidden: np.ndarray,
    logits: np.ndarray,
    true_labels: np.ndarray,
    baseline_accuracy: float,
    cfg,
) -> Dict:
    drift = compute_drift(baseline_hidden, fused_hidden)
    perf = compute_performance(logits, true_labels)
    burden = compute_oversight_burden(logits, cfg.confidence_threshold)
    esys = compute_esys(perf["accuracy"], burden["oversight_burden"])  # always 0.50 random-chance baseline
    acc_drop = max(baseline_accuracy - perf["accuracy"], 0.0)

    return {
        "condition": condition_id,
        "n_cartridges": len(cartridge_ids),
        "cartridge_ids": cartridge_ids,
        "delta": round(drift, 4),
        "accuracy": round(perf["accuracy"], 4),
        "f1_macro": round(perf["f1_macro"], 4),
        "accuracy_drop": round(acc_drop, 4),
        "oversight_burden": round(burden["oversight_burden"], 4),
        "mean_entropy": round(burden["mean_entropy"], 4),
        "mean_confidence": round(burden["mean_confidence"], 4),
        "e_sys": round(esys, 4),
    }


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def run_statistical_tests(conditions: List[Dict]) -> Dict:
    """
    Pearson correlation (δ → accuracy_drop),
    OLS regression,
    Bonferroni-corrected pairwise t-tests on accuracy groups.
    """
    deltas = np.array([c["delta"] for c in conditions])
    acc_drops = np.array([c["accuracy_drop"] for c in conditions])
    burdens = np.array([c["oversight_burden"] for c in conditions])
    esys_vals = np.array([c["e_sys"] for c in conditions])

    r_delta_acc, p_delta_acc = pearsonr(deltas, acc_drops) if len(deltas) > 2 else (np.nan, np.nan)
    r_delta_burden, p_delta_burden = pearsonr(deltas, burdens) if len(deltas) > 2 else (np.nan, np.nan)

    # OLS: accuracy_drop ~ delta
    if len(deltas) > 2:
        slope, intercept, r_value, p_ols, se = stats.linregress(deltas, acc_drops)
        r_squared = r_value ** 2
    else:
        slope, intercept, r_squared, p_ols, se = np.nan, np.nan, np.nan, np.nan, np.nan

    # Bonferroni t-tests: compare condition 1 vs each subsequent condition
    n_comparisons = max(len(conditions) - 1, 1)
    bonferroni_results = []
    if len(conditions) > 1:
        base_acc = conditions[0]["accuracy"]
        # We compare the accuracy of baseline vs each other condition
        # Since we have point estimates (not distributions), we use the burden
        # distributions as a proxy variance indicator (acknowledged limitation)
        for c in conditions[1:]:
            # Approximate z-test: difference in proportion (accuracy treated as proportion)
            n = 150  # eval sample size per condition
            p1, p2 = base_acc, c["accuracy"]
            pooled = (p1 + p2) / 2
            se_diff = np.sqrt(2 * pooled * (1 - pooled) / n) + 1e-8
            z = (p1 - p2) / se_diff
            p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
            p_corrected = min(p_raw * n_comparisons, 1.0)
            bonferroni_results.append({
                "condition": c["condition"],
                "delta_accuracy": round(p1 - p2, 4),
                "z_stat": round(z, 4),
                "p_raw": round(p_raw, 6),
                "p_bonferroni": round(p_corrected, 6),
                "significant": p_corrected < 0.05,
            })

    # Spearman rank correlation (δ vs E_sys — non-parametric)
    rho_delta_esys, p_rho = (
        stats.spearmanr(deltas, esys_vals) if len(deltas) > 2 else (np.nan, np.nan)
    )

    return {
        "pearson_delta_accdrop": {"r": round(float(r_delta_acc), 4), "p": round(float(p_delta_acc), 6)},
        "pearson_delta_burden": {"r": round(float(r_delta_burden), 4), "p": round(float(p_delta_burden), 6)},
        "ols_delta_accdrop": {
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "r_squared": round(float(r_squared), 4),
            "p": round(float(p_ols), 6),
        },
        "spearman_delta_esys": {"rho": round(float(rho_delta_esys), 4), "p": round(float(p_rho), 6)},
        "bonferroni_pairwise": bonferroni_results,
    }

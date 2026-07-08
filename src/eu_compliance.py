"""
Maps empirical metrics to EU AI Act (Regulation EU 2024/1689) compliance zones.

Three-zone framework:
  Zone 1 — Compliant      : δ, accuracy_drop, and oversight_burden within thresholds
  Zone 2 — Caution        : at least one metric in the intermediate range
  Zone 3 — Non-Compliant  : at least one metric exceeds upper threshold

Article mapping:
  Art. 9  — Risk management system       → δ threshold (detectable drift = unmanaged risk)
  Art. 13 — Transparency                 → oversight_burden (humans cannot assess what they cannot see)
  Art. 14 — Human oversight              → effective oversight fails when burden > threshold
  Art. 15 — Accuracy and robustness      → accuracy_drop threshold
"""

from typing import Dict, List


EU_AI_ACT_ARTICLES = {
    "art_9": {
        "title": "Risk management system",
        "metric": "delta",
        "description": "Activation drift (δ) indicates accumulation of unmonitored risks across cartridges.",
    },
    "art_13": {
        "title": "Transparency and provision of information",
        "metric": "mean_entropy",
        "description": "High output entropy reduces operator ability to understand system behaviour.",
    },
    "art_14": {
        "title": "Human oversight",
        "metric": "oversight_burden",
        "description": "Oversight burden quantifies the fraction of decisions requiring human review.",
        "recital": "Recital 84 — operators must be able to 'immediately identify' when the system is not performing as intended.",
    },
    "art_15": {
        "title": "Accuracy, robustness and cybersecurity",
        "metric": "accuracy_drop",
        "description": "Accuracy degradation relative to the single-cartridge baseline.",
    },
}


def classify_zone(condition: Dict, cfg) -> Dict:
    """
    Assign a compliance zone and per-article assessment to one experimental condition.
    """
    delta = condition["delta"]
    acc_drop = condition["accuracy_drop"]
    burden = condition["oversight_burden"]

    def zone_for(val, z1_max, z2_max):
        if val <= z1_max:
            return 1
        elif val <= z2_max:
            return 2
        return 3

    art9_zone  = zone_for(delta,    cfg.zone1_delta_max,    cfg.zone2_delta_max)
    art14_zone = zone_for(burden,   cfg.zone1_burden_max,   cfg.zone2_burden_max)
    art15_zone = zone_for(acc_drop, cfg.zone1_acc_drop_max, cfg.zone2_acc_drop_max)

    # Overall zone is the worst across articles
    overall_zone = max(art9_zone, art14_zone, art15_zone)

    zone_labels = {1: "Compliant", 2: "Caution", 3: "Non-Compliant"}

    return {
        "condition": condition["condition"],
        "n_cartridges": condition["n_cartridges"],
        "overall_zone": overall_zone,
        "overall_label": zone_labels[overall_zone],
        "articles": {
            "art_9":  {"zone": art9_zone,  "label": zone_labels[art9_zone],  "value": delta},
            "art_14": {"zone": art14_zone, "label": zone_labels[art14_zone], "value": burden},
            "art_15": {"zone": art15_zone, "label": zone_labels[art15_zone], "value": acc_drop},
        },
        "remediation": _remediation_guidance(overall_zone, art9_zone, art14_zone, art15_zone),
    }


def _remediation_guidance(overall, art9, art14, art15) -> List[str]:
    """Return actionable remediation steps per violated article."""
    steps = []
    if art9 >= 2:
        steps.append(
            "Art. 9 — Implement continuous drift monitoring with δ logged per inference batch; "
            "trigger risk review when δ exceeds 0.15."
        )
    if art14 >= 2:
        steps.append(
            "Art. 14 — Reduce active cartridge count or increase confidence threshold to restore "
            "effective human oversight capacity; consider automatic routing of uncertain predictions."
        )
    if art15 >= 2:
        steps.append(
            "Art. 15 — Accuracy degradation exceeds acceptable bounds; re-evaluate cartridge "
            "combination or implement domain-gating to restrict cartridge scope."
        )
    if overall == 1:
        steps.append("System operating within EU AI Act compliance bounds.")
    return steps


def classify_all_zones(conditions: List[Dict], cfg) -> List[Dict]:
    return [classify_zone(c, cfg) for c in conditions]


def derive_pilot_thresholds(pilot_conditions: List[Dict]) -> Dict:
    """
    Derive compliance zone thresholds from a pilot subset of conditions
    (n=1 through n=4) without seeing the validation conditions (n=5–7).

    Strategy:
      - δ: Zone 1 upper bound = midpoint between n=1 and n=2 values.
           Zone 2 upper bound = 75% of pilot δ_max.
      - B(n): retain conventional 0.20 / 0.40 (operationally motivated
              by Art. 14 review capacity, not data-derived).
      - acc_drop: retain conventional 0.05 / 0.15 (regulatory language
                  on "appropriate accuracy" implies < 5 pp degradation).

    Returns a dict of threshold values to substitute into ExperimentConfig.
    """
    deltas = sorted(c["delta"] for c in pilot_conditions)
    non_zero = [d for d in deltas if d > 0]

    if len(non_zero) >= 1:
        delta_min_nonzero = non_zero[0]
        delta_max_pilot = max(non_zero)
        # Zone 1 boundary: halfway between baseline (0) and first non-zero delta
        z1_delta = round(delta_min_nonzero / 2, 3)
        # Zone 2 boundary: 75% of pilot maximum
        z2_delta = round(delta_max_pilot * 0.75, 3)
    else:
        z1_delta = 0.05
        z2_delta = 0.10

    return {
        "zone1_delta_max": z1_delta,
        "zone2_delta_max": z2_delta,
        "zone1_burden_max": 0.20,
        "zone2_burden_max": 0.40,
        "zone1_acc_drop_max": 0.05,
        "zone2_acc_drop_max": 0.15,
        "derived_from_n_conditions": len(pilot_conditions),
        "pilot_delta_range": [min(deltas), max(deltas)],
    }


def find_empirical_thresholds(conditions: List[Dict], cfg=None) -> Dict:
    """
    Identify the cartridge count at which each compliance zone transitions.
    These become the paper's empirically-derived threshold values.
    """
    # Default thresholds (used when cfg not provided)
    z1_delta  = cfg.zone1_delta_max    if cfg else 0.05
    z2_delta  = cfg.zone2_delta_max    if cfg else 0.10
    z1_burden = cfg.zone1_burden_max   if cfg else 0.20
    z2_burden = cfg.zone2_burden_max   if cfg else 0.40
    z1_drop   = cfg.zone1_acc_drop_max if cfg else 0.05
    z2_drop   = cfg.zone2_acc_drop_max if cfg else 0.15

    # Find first condition where each metric exits Zone 1
    delta_vals = [(c["n_cartridges"], c["delta"]) for c in conditions]
    burden_vals = [(c["n_cartridges"], c["oversight_burden"]) for c in conditions]
    acc_drop_vals = [(c["n_cartridges"], c["accuracy_drop"]) for c in conditions]

    def first_above(vals, threshold):
        for n, v in vals:
            if v > threshold:
                return n
        return None

    # Empirical breakpoints — use config-calibrated thresholds
    delta_at_15pct = first_above(delta_vals, z1_delta)
    delta_at_30pct = first_above(delta_vals, z2_delta)
    burden_at_20pct = first_above(burden_vals, z1_burden)
    burden_at_40pct = first_above(burden_vals, z2_burden)
    acc_drop_at_5pct = first_above(acc_drop_vals, z1_drop)
    acc_drop_at_15pct = first_above(acc_drop_vals, z2_drop)

    # E_sys peak
    esys_vals = [(c["n_cartridges"], c["e_sys"]) for c in conditions]
    if esys_vals:
        peak_n, peak_esys = max(esys_vals, key=lambda x: x[1])
    else:
        peak_n, peak_esys = None, None

    return {
        "art9_caution_at_n_cartridges": delta_at_15pct,
        "art9_noncompliant_at_n_cartridges": delta_at_30pct,
        "art14_caution_at_n_cartridges": burden_at_20pct,
        "art14_noncompliant_at_n_cartridges": burden_at_40pct,
        "art15_caution_at_n_cartridges": acc_drop_at_5pct,
        "art15_noncompliant_at_n_cartridges": acc_drop_at_15pct,
        "esys_peak": {"n_cartridges": peak_n, "e_sys": peak_esys},
        "note": (
            "Cartridge counts at which each EU AI Act article's threshold is first exceeded. "
            "These are the paper's empirically derived compliance boundaries."
        ),
    }


def generate_compliance_table(zones: List[Dict]) -> str:
    """Pretty-print the compliance zone table for the paper."""
    header = (
        f"{'N':>2}  {'δ Art.9':>10}  {'Burden Art.14':>14}  "
        f"{'AccDrop Art.15':>15}  {'Overall':>12}"
    )
    rows = [header, "-" * len(header)]
    for z in zones:
        row = (
            f"{z['n_cartridges']:>2}  "
            f"{z['articles']['art_9']['label']:>10}  "
            f"{z['articles']['art_14']['label']:>14}  "
            f"{z['articles']['art_15']['label']:>15}  "
            f"{z['overall_label']:>12}"
        )
        rows.append(row)
    return "\n".join(rows)

"""
Fill PAPER.md placeholders with real results from the dual-method experiment.
Run after experiment.py has completed successfully.

Usage:
    python fill_paper.py
"""

import json
import os
import sys
import re


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt(val, decimals=4):
    if val is None:
        return "N/A"
    return f"{float(val):.{decimals}f}"


def fmt_pct(val):
    if val is None:
        return "N/A"
    return f"{float(val)*100:.1f}%"


def zone_label(z):
    return {"Compliant": "Zone 1", "Caution": "Zone 2", "Non-Compliant": "Zone 3"}.get(z, z)


def condition_row(c, z, include_zone=True):
    """Format one table row for a single condition."""
    row = (f"{c['n_cartridges']} | "
           f"{fmt(c['delta'])} | "
           f"{fmt(c['accuracy'])} | "
           f"{fmt(c['accuracy_drop'])} | "
           f"{fmt(c['oversight_burden'])} | "
           f"{fmt(c['e_sys'])}")
    if include_zone:
        row += f" | {zone_label(z['overall_label'])}"
    return row


def burden_profile_summary(conditions):
    """Describe B(n) profile in prose (used for TIES method comparison)."""
    burdens = [(c["n_cartridges"], c["oversight_burden"]) for c in conditions]
    max_b = max(b for _, b in burdens)
    n_above_40 = sum(1 for _, b in burdens if b > 0.40)
    n_above_20 = sum(1 for _, b in burdens if b > 0.20)
    first_above_20 = next((n for n, b in burdens if b > 0.20), None)
    first_above_40 = next((n for n, b in burdens if b > 0.40), None)
    if max_b < 0.20:
        return f"remains below the Zone 1 threshold (max B(n) = {fmt(max_b)}) across all conditions"
    elif max_b < 0.40:
        return (f"exceeds the Zone 1/2 boundary at n={first_above_20} "
                f"(max B(n) = {fmt(max_b)}) but stays below the Zone 2/3 boundary throughout")
    else:
        return (f"exceeds Zone 1 at n={first_above_20} and Zone 2 at n={first_above_40} "
                f"(max B(n) = {fmt(max_b)} at n={max(n for n, b in burdens if b == max_b)})")


def build_replacements(results_dir):
    conditions_ta   = load_json(os.path.join(results_dir, "conditions_ta.json"))
    conditions_ties = load_json(os.path.join(results_dir, "conditions_ties.json"))
    stats_ta        = load_json(os.path.join(results_dir, "statistics_ta.json"))
    stats_ties      = load_json(os.path.join(results_dir, "statistics_ties.json"))
    compliance_ta   = load_json(os.path.join(results_dir, "compliance_ta.json"))
    compliance_ties = load_json(os.path.join(results_dir, "compliance_ties.json"))
    pilot_thresh    = load_json(os.path.join(results_dir, "pilot_thresholds.json"))

    zones_ta   = compliance_ta["zones"]
    zones_ties = compliance_ties["zones"]
    thresh_ta  = compliance_ta["empirical_thresholds"]
    thresh_ties= compliance_ties["empirical_thresholds"]

    repl = {}

    # --- Pilot thresholds ---
    repl["PLACEHOLDER_PILOT_Z1"] = fmt(pilot_thresh["zone1_delta_max"], 3)
    repl["PLACEHOLDER_PILOT_Z2"] = fmt(pilot_thresh["zone2_delta_max"], 3)

    # --- TA pilot rows (n=1–4, no zone column) ---
    for i in range(1, 5):
        c = conditions_ta[i-1]
        z = zones_ta[i-1]
        repl[f"PLACEHOLDER_TA_C{i}_ROW"] = condition_row(c, z, include_zone=False)

    # --- TA full rows (n=1–7, with zone) ---
    for i in range(1, 8):
        c = conditions_ta[i-1]
        z = zones_ta[i-1]
        repl[f"PLACEHOLDER_TA_C{i}_FULL"] = condition_row(c, z, include_zone=True)
        repl[f"PLACEHOLDER_TA_C{i}_DELTA"]   = fmt(c["delta"])
        repl[f"PLACEHOLDER_TA_C{i}_ACC"]     = fmt(c["accuracy"])
        repl[f"PLACEHOLDER_TA_C{i}_DROP"]    = fmt(c["accuracy_drop"])
        repl[f"PLACEHOLDER_TA_C{i}_BURDEN"]  = fmt(c["oversight_burden"])
        repl[f"PLACEHOLDER_TA_C{i}_F1"]      = fmt(c["f1_macro"])

    # --- TIES full rows ---
    for i in range(1, 8):
        c = conditions_ties[i-1]
        z = zones_ties[i-1]
        repl[f"PLACEHOLDER_TIES_C{i}_FULL"] = condition_row(c, z, include_zone=True)
        repl[f"PLACEHOLDER_TIES_C{i}_DELTA"]   = fmt(c["delta"])
        repl[f"PLACEHOLDER_TIES_C{i}_ACC"]     = fmt(c["accuracy"])

    # --- TA statistics ---
    p_ta = stats_ta["pearson_delta_accdrop"]
    ols_ta = stats_ta["ols_delta_accdrop"]
    sp_ta  = stats_ta.get("spearman_delta_esys", {})
    repl["PLACEHOLDER_TA_PEARSON_R"]     = fmt(p_ta["r"], 3)
    repl["PLACEHOLDER_TA_PEARSON_P"]     = (fmt(p_ta["p"], 4) if p_ta["p"] >= 0.0001
                                             else f"{p_ta['p']:.2e}")
    repl["PLACEHOLDER_TA_R2"]            = fmt(ols_ta["r_squared"], 3)
    repl["PLACEHOLDER_TA_SLOPE"]         = fmt(ols_ta["slope"], 4)
    repl["PLACEHOLDER_TA_SPEARMAN_RHO"]  = fmt(sp_ta.get("rho"), 3)
    repl["PLACEHOLDER_TA_SPEARMAN_P"]    = (fmt(sp_ta["p"], 4)
                                             if sp_ta.get("p") and sp_ta["p"] >= 0.0001
                                             else f"{sp_ta.get('p', float('nan')):.2e}")

    # --- TIES statistics ---
    p_ties  = stats_ties["pearson_delta_accdrop"]
    ols_ties = stats_ties["ols_delta_accdrop"]
    repl["PLACEHOLDER_TIES_PEARSON_R"] = fmt(p_ties["r"], 3)
    repl["PLACEHOLDER_TIES_PEARSON_P"] = (fmt(p_ties["p"], 4) if p_ties["p"] >= 0.0001
                                           else f"{p_ties['p']:.2e}")
    repl["PLACEHOLDER_TIES_R2"]        = fmt(ols_ties["r_squared"], 3)

    # --- TA Bonferroni ---
    bonf_ta = stats_ta.get("bonferroni_pairwise", [])
    sig_n = [str(b["condition"]) for b in bonf_ta if b["significant"]]
    repl["PLACEHOLDER_TA_BONF_SIG"] = (
        "Conditions n = " + ", ".join(sig_n) if sig_n else "no conditions"
    )

    # --- Compliance breakpoints ---
    def thresh_str(val):
        return str(val) if val is not None else ">7"

    repl["PLACEHOLDER_TA_ART9_CAU"]   = thresh_str(thresh_ta["art9_caution_at_n_cartridges"])
    repl["PLACEHOLDER_TA_ART9_NC"]    = thresh_str(thresh_ta["art9_noncompliant_at_n_cartridges"])
    repl["PLACEHOLDER_TA_ART14_CAU"]  = thresh_str(thresh_ta["art14_caution_at_n_cartridges"])
    repl["PLACEHOLDER_TA_ART14_NC"]   = thresh_str(thresh_ta["art14_noncompliant_at_n_cartridges"])
    repl["PLACEHOLDER_TA_ART15_CAU"]  = thresh_str(thresh_ta["art15_caution_at_n_cartridges"])
    repl["PLACEHOLDER_TA_ART15_NC"]   = thresh_str(thresh_ta["art15_noncompliant_at_n_cartridges"])

    repl["PLACEHOLDER_TIES_ART9_CAU"]  = thresh_str(thresh_ties["art9_caution_at_n_cartridges"])
    repl["PLACEHOLDER_TIES_ART9_NC"]   = thresh_str(thresh_ties["art9_noncompliant_at_n_cartridges"])
    repl["PLACEHOLDER_TIES_ART14_CAU"] = thresh_str(thresh_ties["art14_caution_at_n_cartridges"])
    repl["PLACEHOLDER_TIES_ART14_NC"]  = thresh_str(thresh_ties["art14_noncompliant_at_n_cartridges"])
    repl["PLACEHOLDER_TIES_ART15_CAU"] = thresh_str(thresh_ties["art15_caution_at_n_cartridges"])
    repl["PLACEHOLDER_TIES_ART15_NC"]  = thresh_str(thresh_ties["art15_noncompliant_at_n_cartridges"])

    # --- TIES delta range and burden profile ---
    ties_deltas = [c["delta"] for c in conditions_ties]
    repl["PLACEHOLDER_TIES_DELTA_RANGE"] = (
        f"δ = {fmt(min(ties_deltas))}–{fmt(max(ties_deltas))}"
    )
    repl["PLACEHOLDER_TIES_ART9_N"] = thresh_str(thresh_ties["art9_caution_at_n_cartridges"])
    repl["PLACEHOLDER_TIES_BURDEN_PROFILE"] = burden_profile_summary(conditions_ties)

    return repl


def apply_replacements(text, replacements):
    for placeholder, value in sorted(replacements.items(), key=lambda x: -len(x[0])):
        text = text.replace(placeholder, str(value))
    remaining = re.findall(r"PLACEHOLDER_\w+", text)
    if remaining:
        print(f"WARNING: {len(remaining)} unfilled placeholders: {set(remaining)}")
    return text


def main():
    results_dir = "results"
    required_files = [
        "conditions_ta.json", "conditions_ties.json",
        "statistics_ta.json", "statistics_ties.json",
        "compliance_ta.json", "compliance_ties.json",
        "pilot_thresholds.json",
    ]
    for f in required_files:
        path = os.path.join(results_dir, f)
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run experiment.py first.")
            sys.exit(1)

    paper_path = "PAPER.md"
    if not os.path.exists(paper_path):
        print(f"ERROR: {paper_path} not found.")
        sys.exit(1)

    replacements = build_replacements(results_dir)

    with open(paper_path) as f:
        paper_text = f.read()

    filled = apply_replacements(paper_text, replacements)

    with open(paper_path, "w") as f:
        f.write(filled)

    print(f"Filled {len(replacements)} placeholders in {paper_path}.")
    print(f"\nKey results:")
    print(f"  Pilot δ Zone1/2 boundary: {replacements['PLACEHOLDER_PILOT_Z1']}")
    print(f"  Pilot δ Zone2/3 boundary: {replacements['PLACEHOLDER_PILOT_Z2']}")
    print(f"  TA  Pearson r(δ,drop): {replacements['PLACEHOLDER_TA_PEARSON_R']}  "
          f"p={replacements['PLACEHOLDER_TA_PEARSON_P']}")
    print(f"  TIES Pearson r(δ,drop): {replacements['PLACEHOLDER_TIES_PEARSON_R']}  "
          f"p={replacements['PLACEHOLDER_TIES_PEARSON_P']}")
    print(f"  TA  Art.9 caution at n={replacements['PLACEHOLDER_TA_ART9_CAU']}")
    print(f"  TIES Art.9 caution at n={replacements['PLACEHOLDER_TIES_ART9_CAU']}")
    print(f"  TIES B(n) profile: {replacements['PLACEHOLDER_TIES_BURDEN_PROFILE']}")


if __name__ == "__main__":
    main()

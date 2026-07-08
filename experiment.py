"""
EU AI Act Governance Experiment for Modular RAP Systems
========================================================
Trains 7 domain-specific LoRA cartridges, progressively fuses them via
(a) Task Arithmetic and (b) TIES-Merging, then maps empirical drift/burden/
accuracy metrics to EU AI Act compliance zones.

Pilot/validation split: conditions n=1–4 are used to derive δ thresholds;
conditions n=5–7 and all TIES-Merging conditions serve as independent
validation of those thresholds.

Usage:
    python experiment.py [--skip-training] [--device cpu]

Outputs (results/):
    conditions_ta.json    — Task Arithmetic per-condition metrics
    conditions_ties.json  — TIES-Merging per-condition metrics
    statistics_ta.json    — TA statistical tests
    statistics_ties.json  — TIES statistical tests
    compliance_ta.json    — TA compliance zones + empirical thresholds
    compliance_ties.json  — TIES compliance zones (same thresholds applied)
    pilot_thresholds.json — Thresholds derived from TA pilot (n=1–4 only)
    fig1_drift_curve.{pdf,png}
    fig2_accuracy_degradation.{pdf,png}
    fig3_oversight_esys.{pdf,png}
    fig4_compliance_heatmap.{pdf,png}
    fig5_delta_accuracy_correlation.{pdf,png}
    fig6_method_comparison.{pdf,png}
"""

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.config import ExperimentConfig, CARTRIDGE_REGISTRY
from src.data_loader import load_all_cartridges
from src.cartridge import (
    train_cartridge,
    save_cartridge,
    load_cartridge,
    fuse_cartridges,
    fuse_cartridges_ties,
    run_inference,
)
from src.metrics import build_condition_metrics, run_statistical_tests, compute_performance
from src.eu_compliance import (
    classify_all_zones,
    find_empirical_thresholds,
    generate_compliance_table,
    derive_pilot_thresholds,
)
from src.visualize import generate_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    log.info("Saved %s", path)


def run_fusion_conditions(
    trained_models,
    all_cartridge_data,
    primary_data,
    tokenizer,
    baseline_hidden,
    baseline_labels,
    baseline_accuracy,
    cfg,
    device,
    fusion_method: str = "ta",
) -> list:
    """
    Run all n=1..N fusion conditions using the specified method.
    fusion_method: "ta" (Task Arithmetic) or "ties" (TIES-Merging)
    """
    n_cartridges = len(trained_models)
    conditions = []

    for n in range(1, n_cartridges + 1):
        log.info("--- [%s] Condition %d: fusing cartridges 1-%d ---", fusion_method.upper(), n, n)
        models_to_fuse = trained_models[:n]
        cartridge_ids = [all_cartridge_data[i]["spec"]["id"] for i in range(n)]

        if n == 1:
            fused_model = trained_models[0]
        elif fusion_method == "ties":
            fused_model = fuse_cartridges_ties(
                cartridge_models=models_to_fuse,
                primary_model=trained_models[0],
                cfg=cfg,
                device=device,
                trim_ratio=0.20,
            )
        else:  # "ta"
            fused_model = fuse_cartridges(
                cartridge_models=models_to_fuse,
                primary_model=trained_models[0],
                cfg=cfg,
                device=device,
            )

        logits, hidden, labels = run_inference(
            fused_model, primary_data["eval"], tokenizer, cfg, device
        )

        cond_metrics = build_condition_metrics(
            condition_id=n,
            cartridge_ids=cartridge_ids,
            baseline_hidden=baseline_hidden,
            fused_hidden=hidden,
            logits=logits,
            true_labels=labels,
            baseline_accuracy=baseline_accuracy,
            cfg=cfg,
        )
        cond_metrics["fusion_method"] = fusion_method
        conditions.append(cond_metrics)
        log.info(
            "  δ=%.4f  acc=%.4f  drop=%.4f  burden=%.4f  E_sys=%.4f",
            cond_metrics["delta"],
            cond_metrics["accuracy"],
            cond_metrics["accuracy_drop"],
            cond_metrics["oversight_burden"],
            cond_metrics["e_sys"],
        )

    return conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="Load pre-trained adapters instead of re-training")
    parser.add_argument("--device", default=None,
                        help="Force device: cpu | cuda | mps")
    parser.add_argument("--max-cartridges", type=int, default=7,
                        help="Run first N cartridges only (useful for quick tests)")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.adapters_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(args.device) if args.device else pick_device()
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    log.info("=== Loading datasets ===")
    all_cartridge_data = load_all_cartridges(cfg)
    if not all_cartridge_data:
        log.error("No datasets loaded.")
        sys.exit(1)
    all_cartridge_data = all_cartridge_data[: args.max_cartridges]
    n_cartridges = len(all_cartridge_data)
    log.info("Loaded %d cartridge datasets", n_cartridges)

    # ------------------------------------------------------------------
    # 2. Train or load cartridges
    # ------------------------------------------------------------------
    log.info("=== Training / loading cartridges ===")
    trained_models = []
    for i, cdata in enumerate(all_cartridge_data):
        cartridge_id = cdata["spec"]["id"]
        adapter_path = os.path.join(cfg.adapters_dir, f"cartridge_{cartridge_id}")
        if args.skip_training and os.path.exists(adapter_path):
            log.info("Loading pre-trained cartridge %d", cartridge_id)
            model = load_cartridge(cartridge_id, cfg, device)
        else:
            log.info("Training cartridge %d — %s", cartridge_id, cdata["spec"]["name"])
            model, _ = train_cartridge(
                cartridge_id=cartridge_id,
                train_dataset=cdata["train"],
                cfg=cfg,
                device=device,
            )
            if cfg.save_adapters:
                save_cartridge(model, cartridge_id, cfg)
        trained_models.append(model)

    # ------------------------------------------------------------------
    # 3. Tokeniser and primary task data
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    primary_data = all_cartridge_data[0]

    # ------------------------------------------------------------------
    # 4. Baseline inference (Condition 1 — single cartridge)
    # ------------------------------------------------------------------
    log.info("=== Running baseline inference (Condition 1) ===")
    baseline_logits, baseline_hidden, baseline_labels = run_inference(
        trained_models[0], primary_data["eval"], tokenizer, cfg, device
    )
    baseline_perf = compute_performance(baseline_logits, baseline_labels)
    baseline_accuracy = baseline_perf["accuracy"]
    log.info("Baseline accuracy: %.4f  F1-macro: %.4f",
             baseline_accuracy, baseline_perf["f1_macro"])

    # ------------------------------------------------------------------
    # 5. Task Arithmetic conditions (n=1..N)
    # ------------------------------------------------------------------
    log.info("=== Running Task Arithmetic fusion conditions ===")
    conditions_ta = run_fusion_conditions(
        trained_models, all_cartridge_data, primary_data,
        tokenizer, baseline_hidden, baseline_labels, baseline_accuracy,
        cfg, device, fusion_method="ta",
    )

    # ------------------------------------------------------------------
    # 6. Pilot/validation split on Task Arithmetic results
    #    Pilot:      conditions n=1–4
    #    Validation: conditions n=5–7
    # ------------------------------------------------------------------
    PILOT_N = 4
    pilot_conditions = [c for c in conditions_ta if c["n_cartridges"] <= PILOT_N]
    validation_conditions = [c for c in conditions_ta if c["n_cartridges"] > PILOT_N]

    log.info("=== Deriving pilot thresholds from TA conditions n=1–%d ===", PILOT_N)
    pilot_thresholds = derive_pilot_thresholds(pilot_conditions)
    log.info("Pilot thresholds: δ Zone1≤%.3f  Zone2≤%.3f",
             pilot_thresholds["zone1_delta_max"],
             pilot_thresholds["zone2_delta_max"])

    # Build a cfg-like object with pilot-derived thresholds for classification
    class PilotCfg:
        pass
    pilot_cfg = PilotCfg()
    for k, v in pilot_thresholds.items():
        if k.startswith("zone"):
            setattr(pilot_cfg, k, v)
    pilot_cfg.confidence_threshold = cfg.confidence_threshold

    # ------------------------------------------------------------------
    # 7. TIES-Merging conditions (n=1..N)
    # ------------------------------------------------------------------
    log.info("=== Running TIES-Merging fusion conditions ===")
    conditions_ties = run_fusion_conditions(
        trained_models, all_cartridge_data, primary_data,
        tokenizer, baseline_hidden, baseline_labels, baseline_accuracy,
        cfg, device, fusion_method="ties",
    )

    # ------------------------------------------------------------------
    # 8. Statistical analysis
    # ------------------------------------------------------------------
    log.info("=== Statistical analysis ===")
    stats_ta   = run_statistical_tests(conditions_ta)
    stats_ties = run_statistical_tests(conditions_ties)
    log.info("[TA]   Pearson r(δ,acc_drop)=%.4f  p=%.6f",
             stats_ta["pearson_delta_accdrop"]["r"],
             stats_ta["pearson_delta_accdrop"]["p"])
    log.info("[TIES] Pearson r(δ,acc_drop)=%.4f  p=%.6f",
             stats_ties["pearson_delta_accdrop"]["r"],
             stats_ties["pearson_delta_accdrop"]["p"])

    # ------------------------------------------------------------------
    # 9. EU AI Act compliance classification
    #    - TA classified with pilot-derived thresholds
    #    - TIES classified with the same thresholds (independent validation)
    # ------------------------------------------------------------------
    log.info("=== EU AI Act compliance classification ===")
    zones_ta   = classify_all_zones(conditions_ta,   pilot_cfg)
    zones_ties = classify_all_zones(conditions_ties, pilot_cfg)
    thresholds_ta   = find_empirical_thresholds(conditions_ta,   pilot_cfg)
    thresholds_ties = find_empirical_thresholds(conditions_ties, pilot_cfg)

    log.info("Task Arithmetic compliance table:")
    log.info("\n" + generate_compliance_table(zones_ta))
    log.info("TIES-Merging compliance table:")
    log.info("\n" + generate_compliance_table(zones_ties))

    # ------------------------------------------------------------------
    # 10. Save results
    # ------------------------------------------------------------------
    save_json(conditions_ta,   f"{cfg.results_dir}/conditions_ta.json")
    save_json(conditions_ties, f"{cfg.results_dir}/conditions_ties.json")
    save_json(stats_ta,        f"{cfg.results_dir}/statistics_ta.json")
    save_json(stats_ties,      f"{cfg.results_dir}/statistics_ties.json")
    save_json(pilot_thresholds,f"{cfg.results_dir}/pilot_thresholds.json")
    save_json(
        {"zones": zones_ta, "empirical_thresholds": thresholds_ta},
        f"{cfg.results_dir}/compliance_ta.json",
    )
    save_json(
        {"zones": zones_ties, "empirical_thresholds": thresholds_ties,
         "note": "TIES-Merging conditions classified with TA-pilot-derived thresholds."},
        f"{cfg.results_dir}/compliance_ties.json",
    )

    # ------------------------------------------------------------------
    # 11. Figures
    # ------------------------------------------------------------------
    log.info("=== Generating figures ===")
    generate_all_figures(
        conditions_ta, zones_ta, stats_ta, cfg,
        conditions_ties=conditions_ties,
    )

    # ------------------------------------------------------------------
    # 12. Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nPilot thresholds (derived from TA n=1–{PILOT_N}):")
    print(f"  δ Zone1 ≤ {pilot_thresholds['zone1_delta_max']:.3f}  "
          f"Zone2 ≤ {pilot_thresholds['zone2_delta_max']:.3f}")

    for method, conditions, zones, stats in [
        ("Task Arithmetic", conditions_ta, zones_ta, stats_ta),
        ("TIES-Merging",    conditions_ties, zones_ties, stats_ties),
    ]:
        print(f"\n{'─'*70}")
        print(f"{method}")
        print(f"{'─'*70}")
        print(f"  {'N':>2}  {'δ':>8}  {'Accuracy':>10}  {'Drop':>8}  "
              f"{'Burden':>8}  {'E_sys':>8}  {'Zone':>14}")
        print(f"  {'-'*65}")
        for c, z in zip(conditions, zones):
            print(
                f"  {c['n_cartridges']:>2}  "
                f"{c['delta']:>8.4f}  "
                f"{c['accuracy']:>10.4f}  "
                f"{c['accuracy_drop']:>8.4f}  "
                f"{c['oversight_burden']:>8.4f}  "
                f"{c['e_sys']:>8.4f}  "
                f"{z['overall_label']:>14}"
            )
        print(f"\n  Pearson r(δ, acc_drop) = "
              f"{stats['pearson_delta_accdrop']['r']:.4f}  "
              f"p = {stats['pearson_delta_accdrop']['p']:.6f}")
        print(f"  OLS R² = {stats['ols_delta_accdrop']['r_squared']:.4f}")
        thresholds = thresholds_ta if method == "Task Arithmetic" else thresholds_ties
        print(f"  Art. 9 caution at n = {thresholds['art9_caution_at_n_cartridges']}")
        print(f"  Art. 14 caution at n = {thresholds['art14_caution_at_n_cartridges']}")
        print(f"  Art. 15 caution at n = {thresholds['art15_caution_at_n_cartridges']}")

    print(f"\n{'='*70}")
    print(f"All outputs in: {os.path.abspath(cfg.results_dir)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

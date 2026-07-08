"""Publication-quality figures for the paper."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Dict

from .config import ExperimentConfig

ZONE_COLORS = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}
ZONE_LABELS = {1: "Zone 1: Compliant", 2: "Zone 2: Caution", 3: "Zone 3: Non-Compliant"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


def _save(fig, name: str, results_dir: str):
    path = os.path.join(results_dir, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    path_png = os.path.join(results_dir, f"{name}.png")
    fig.savefig(path_png, bbox_inches="tight")
    plt.close(fig)


def plot_drift_curve(conditions: List[Dict], zones: List[Dict], cfg: ExperimentConfig):
    """Figure 1: Manifold Drift (δ) vs number of cartridges with EU AI Act thresholds."""
    ns = [c["n_cartridges"] for c in conditions]
    deltas = [c["delta"] for c in conditions]
    zone_colors = [ZONE_COLORS[z["overall_zone"]] for z in zones]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Shade compliance zones
    ax.axhspan(0, cfg.zone1_delta_max, alpha=0.08, color=ZONE_COLORS[1])
    ax.axhspan(cfg.zone1_delta_max, cfg.zone2_delta_max, alpha=0.08, color=ZONE_COLORS[2])
    ax.axhspan(cfg.zone2_delta_max, max(deltas) * 1.3 + 0.05, alpha=0.08, color=ZONE_COLORS[3])

    ax.plot(ns, deltas, "k-o", linewidth=1.8, markersize=7, zorder=5, label="δ (Manifold Drift)")
    for n, d, col in zip(ns, deltas, zone_colors):
        ax.plot(n, d, "o", color=col, markersize=10, zorder=6)

    ax.axhline(cfg.zone1_delta_max, color=ZONE_COLORS[2], linestyle="--", linewidth=1.2,
               label=f"Art. 9 Caution threshold (δ={cfg.zone1_delta_max})")
    ax.axhline(cfg.zone2_delta_max, color=ZONE_COLORS[3], linestyle="--", linewidth=1.2,
               label=f"Art. 9 Non-Compliant threshold (δ={cfg.zone2_delta_max})")

    ax.set_xlabel("Number of Fused Cartridges (n)")
    ax.set_ylabel("Manifold Drift (δ)")
    ax.set_title("Figure 1 — Activation Space Drift vs Cartridge Count\n"
                 "with EU AI Act Article 9 Risk Management Thresholds")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xticks(ns)
    ax.set_xlim(0.5, max(ns) + 0.5)
    fig.tight_layout()
    _save(fig, "fig1_drift_curve", cfg.results_dir)


def plot_accuracy_degradation(conditions: List[Dict], zones: List[Dict], cfg: ExperimentConfig):
    """Figure 2: Accuracy drop vs n cartridges with Art. 15 thresholds."""
    ns = [c["n_cartridges"] for c in conditions]
    acc = [c["accuracy"] for c in conditions]
    acc_drop = [c["accuracy_drop"] for c in conditions]
    zone_colors = [ZONE_COLORS[z["articles"]["art_15"]["zone"]] for z in zones]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: absolute accuracy
    ax = axes[0]
    ax.plot(ns, acc, "b-o", linewidth=1.8, markersize=7)
    for n, a, col in zip(ns, acc, zone_colors):
        ax.plot(n, a, "o", color=col, markersize=10, zorder=6)
    ax.set_xlabel("Number of Fused Cartridges (n)")
    ax.set_ylabel("Accuracy on Primary Task")
    ax.set_title("Task Accuracy vs Cartridge Count")
    ax.set_xticks(ns)
    ax.set_ylim(0, 1.05)

    # Right: accuracy drop vs proposed thresholds
    ax = axes[1]
    ax.axhspan(0, cfg.zone1_acc_drop_max, alpha=0.08, color=ZONE_COLORS[1])
    ax.axhspan(cfg.zone1_acc_drop_max, cfg.zone2_acc_drop_max, alpha=0.08, color=ZONE_COLORS[2])
    ax.axhspan(cfg.zone2_acc_drop_max, max(acc_drop) * 1.3 + 0.02, alpha=0.08, color=ZONE_COLORS[3])

    ax.plot(ns, acc_drop, "r-s", linewidth=1.8, markersize=7, label="Accuracy Drop")
    ax.axhline(cfg.zone1_acc_drop_max, color=ZONE_COLORS[2], linestyle="--", linewidth=1.2,
               label=f"Art. 15 Caution ({cfg.zone1_acc_drop_max:.0%})")
    ax.axhline(cfg.zone2_acc_drop_max, color=ZONE_COLORS[3], linestyle="--", linewidth=1.2,
               label=f"Art. 15 Non-Compliant ({cfg.zone2_acc_drop_max:.0%})")
    ax.set_xlabel("Number of Fused Cartridges (n)")
    ax.set_ylabel("Accuracy Drop vs Single-Cartridge Baseline")
    ax.set_title("Art. 15 Accuracy Degradation with EU AI Act Thresholds")
    ax.legend(fontsize=9)
    ax.set_xticks(ns)

    patches = [mpatches.Patch(color=ZONE_COLORS[i], alpha=0.5, label=ZONE_LABELS[i]) for i in [1, 2, 3]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Figure 2 — Accuracy Degradation (EU AI Act Article 15)", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_accuracy_degradation", cfg.results_dir)


def plot_oversight_and_esys(conditions: List[Dict], zones: List[Dict], cfg: ExperimentConfig):
    """Figure 3: Oversight burden (Art. 14) and E_sys on one plot."""
    ns = [c["n_cartridges"] for c in conditions]
    burden = [c["oversight_burden"] for c in conditions]
    esys = [c["e_sys"] for c in conditions]

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax2 = ax1.twinx()

    # Shade Art. 14 zones on left axis
    ax1.axhspan(0, cfg.zone1_burden_max, alpha=0.07, color=ZONE_COLORS[1])
    ax1.axhspan(cfg.zone1_burden_max, cfg.zone2_burden_max, alpha=0.07, color=ZONE_COLORS[2])
    ax1.axhspan(cfg.zone2_burden_max, 1.0, alpha=0.07, color=ZONE_COLORS[3])

    l1, = ax1.plot(ns, burden, "r-^", linewidth=1.8, markersize=8, label="Oversight Burden B(n)")
    ax1.axhline(cfg.zone1_burden_max, color=ZONE_COLORS[2], linestyle="--", linewidth=1.0,
                label=f"Art. 14 Caution ({cfg.zone1_burden_max:.0%})")
    ax1.axhline(cfg.zone2_burden_max, color=ZONE_COLORS[3], linestyle="--", linewidth=1.0,
                label=f"Art. 14 Non-Compliant ({cfg.zone2_burden_max:.0%})")
    ax1.set_xlabel("Number of Fused Cartridges (n)")
    ax1.set_ylabel("Oversight Burden B(n)\n(fraction uncertain predictions)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_xticks(ns)
    ax1.set_ylim(0, 1.05)

    l2, = ax2.plot(ns, esys, "b-o", linewidth=1.8, markersize=8, label="E_sys")
    ax2.set_ylabel("System Efficacy E_sys", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Peak E_sys annotation — place text below-right of peak to avoid title collision
    if esys:
        peak_idx = int(np.argmax(esys))
        ax2.annotate(
            f"E_sys peak\nn={ns[peak_idx]}",
            xy=(ns[peak_idx], esys[peak_idx]),
            xytext=(ns[peak_idx] + 1.2, esys[peak_idx] * 0.80),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="blue"),
            color="blue",
        )

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center", bbox_to_anchor=(0.72, 0.27), fontsize=9)

    ax1.set_title("Figure 3 — Human Oversight Burden and System Efficacy\n"
                  "EU AI Act Article 14 Compliance Boundary")
    fig.tight_layout()
    _save(fig, "fig3_oversight_esys", cfg.results_dir)


def plot_compliance_heatmap(conditions: List[Dict], zones: List[Dict], cfg: ExperimentConfig):
    """Figure 4: 2D compliance zone heatmap (δ × burden)."""
    deltas = np.array([c["delta"] for c in conditions])
    burdens = np.array([c["oversight_burden"] for c in conditions])
    ns = [c["n_cartridges"] for c in conditions]
    zone_nums = [z["overall_zone"] for z in zones]

    fig, ax = plt.subplots(figsize=(5, 4))

    # Background grid
    d_range = np.linspace(0, max(deltas) * 1.2, 200)
    b_range = np.linspace(0, max(burdens) * 1.2, 200)
    D, B = np.meshgrid(d_range, b_range)

    def zone_val(d, b):
        z_art9 = 1 if d <= cfg.zone1_delta_max else (2 if d <= cfg.zone2_delta_max else 3)
        z_art14 = 1 if b <= cfg.zone1_burden_max else (2 if b <= cfg.zone2_burden_max else 3)
        return max(z_art9, z_art14)

    Z = np.vectorize(zone_val)(D, B)
    cmap = plt.cm.colors.ListedColormap(["#d5f5e3", "#fdebd0", "#fadbd8"])
    ax.contourf(D, B, Z, levels=[0.5, 1.5, 2.5, 3.5], colors=["#d5f5e3", "#fdebd0", "#fadbd8"],
                alpha=0.6)

    scatter = ax.scatter(deltas, burdens, c=zone_nums, cmap=plt.cm.colors.ListedColormap(
        [ZONE_COLORS[1], ZONE_COLORS[2], ZONE_COLORS[3]]),
        s=120, zorder=5, edgecolors="black", linewidths=0.8,
        vmin=1, vmax=3)

    # Staggered offsets so labels don't collide — n=2–7 all sit at B=1.0
    # with very similar δ, so alternate left/right and step down in y.
    label_offsets = [
        (6,   4),   # n=1: isolated, right/above
        (6,  18),   # n=2: right, high above
        (-46,  8),  # n=3: left, lower above (vertical stagger clears n=2)
        (6,  -14),  # n=4: right, below
        (-46, -14), # n=5: left, below
        (6,  -26),  # n=6: right, further below
        (-46, -26), # n=7: left, further below
    ]
    for (d, b, n), (ox, oy) in zip(zip(deltas, burdens, ns), label_offsets):
        ax.annotate(f"n={n}", (d, b), textcoords="offset points", xytext=(ox, oy), fontsize=8)

    ax.axvline(cfg.zone1_delta_max, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(cfg.zone2_delta_max, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(cfg.zone1_burden_max, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(cfg.zone2_burden_max, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)

    # Add left margin so the n=1 dot at δ=0 isn't clipped by the axis edge
    ax.set_xlim(-0.005, max(deltas) * 1.2)

    ax.set_xlabel("Manifold Drift δ  (Art. 9 metric)")
    ax.set_ylabel("Oversight Burden B(n)  (Art. 14 metric)")
    ax.set_title("Figure 4 — EU AI Act Compliance Zone Map\n"
                 "Art. 9 × Art. 14 Cross-Metric Classification")

    patches = [mpatches.Patch(color=ZONE_COLORS[i], alpha=0.7, label=ZONE_LABELS[i]) for i in [1, 2, 3]]
    ax.legend(handles=patches, fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig4_compliance_heatmap", cfg.results_dir)


def plot_correlation(conditions: List[Dict], stats_results: Dict, cfg: ExperimentConfig):
    """Figure 5: Scatter δ vs accuracy_drop with OLS fit and R² annotation."""
    deltas = np.array([c["delta"] for c in conditions])
    acc_drops = np.array([c["accuracy_drop"] for c in conditions])

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    pearson_r = stats_results.get("pearson_delta_accdrop", {}).get("r", np.nan)
    pearson_p = stats_results.get("pearson_delta_accdrop", {}).get("p", np.nan)

    ax.scatter(deltas, acc_drops, s=100, zorder=5, edgecolors="black",
               color="#3498db", linewidths=0.8,
               label=f"Conditions (r={pearson_r:.3f}, p={pearson_p:.4f})")

    ols = stats_results.get("ols_delta_accdrop", {})
    slope = ols.get("slope", np.nan)
    intercept = ols.get("intercept", np.nan)
    r2 = ols.get("r_squared", np.nan)
    p_ols = ols.get("p", np.nan)

    if not np.isnan(slope):
        x_fit = np.linspace(deltas.min(), deltas.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "r--", linewidth=1.5,
                label=f"OLS fit (R²={r2:.2f})")

    ax.set_xlabel("Manifold Drift δ")
    ax.set_ylabel("Accuracy Drop")
    ax.set_title("Figure 5 — Activation Drift vs Accuracy Drop\n"
                 "(RQ2: δ as Art. 9 risk predictor)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig5_delta_accuracy_correlation", cfg.results_dir)


def plot_fusion_method_comparison(
    conditions_ta: List[Dict],
    conditions_ties: List[Dict],
    cfg: ExperimentConfig,
):
    """
    Figure 6 — B(n) and δ under Task Arithmetic vs TIES-Merging.

    This is the paper's core comparative figure: it shows that δ increases
    consistently under both methods (δ is fusion-method-agnostic), while
    B(n) depends heavily on the fusion method — demonstrating that the Art. 14
    burden cliff observed under Task Arithmetic is a method-specific artifact,
    not an inherent property of modular architectures.
    """
    ns = [c["n_cartridges"] for c in conditions_ta]

    burden_ta   = [c["oversight_burden"] for c in conditions_ta]
    burden_ties = [c["oversight_burden"] for c in conditions_ties]
    delta_ta    = [c["delta"] for c in conditions_ta]
    delta_ties  = [c["delta"] for c in conditions_ties]
    acc_ta      = [c["accuracy"] for c in conditions_ta]
    acc_ties    = [c["accuracy"] for c in conditions_ties]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # --- Left: δ comparison ---
    ax = axes[0]
    ax.plot(ns, delta_ta,   "b-o", linewidth=1.8, markersize=7, label="Task Arithmetic")
    ax.plot(ns, delta_ties, "gs", linewidth=1.8, markersize=7, linestyle="--",
            label="TIES-Merging")
    ax.axhline(cfg.zone1_delta_max, color=ZONE_COLORS[2], linestyle=":", linewidth=1.2,
               label=f"Zone 1/2 boundary (δ={cfg.zone1_delta_max})")
    ax.axhline(cfg.zone2_delta_max, color=ZONE_COLORS[3], linestyle=":", linewidth=1.2,
               label=f"Zone 2/3 boundary (δ={cfg.zone2_delta_max})")
    ax.set_xlabel("n Cartridges")
    ax.set_ylabel("Manifold Drift δ  (Art. 9)")
    ax.set_title("Activation Drift\n(method-agnostic)")
    ax.legend(fontsize=8)
    ax.set_xticks(ns)

    # --- Middle: B(n) comparison ---
    ax = axes[1]
    ax.plot(ns, burden_ta,   "r-^", linewidth=1.8, markersize=7, label="Task Arithmetic")
    ax.plot(ns, burden_ties, "mD", linewidth=1.8, markersize=7, linestyle="--",
            label="TIES-Merging")
    ax.axhline(cfg.zone1_burden_max, color=ZONE_COLORS[2], linestyle=":", linewidth=1.2,
               label=f"Art. 14 Caution (B={cfg.zone1_burden_max})")
    ax.axhline(cfg.zone2_burden_max, color=ZONE_COLORS[3], linestyle=":", linewidth=1.2,
               label=f"Art. 14 NC (B={cfg.zone2_burden_max})")
    ax.set_xlabel("n Cartridges")
    ax.set_ylabel("Oversight Burden B(n)  (Art. 14)")
    ax.set_title("Oversight Burden\n(collapses to 1.0 under both methods)")
    ax.legend(fontsize=8)
    ax.set_xticks(ns)
    ax.set_ylim(-0.05, 1.1)

    # --- Right: Accuracy comparison ---
    ax = axes[2]
    ax.plot(ns, acc_ta,   "b-o", linewidth=1.8, markersize=7, label="Task Arithmetic")
    ax.plot(ns, acc_ties, "gs", linewidth=1.8, markersize=7, linestyle="--",
            label="TIES-Merging")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=1.0, label="Chance (50%)")
    ax.set_xlabel("n Cartridges")
    ax.set_ylabel("Accuracy on Primary Task  (Art. 15)")
    ax.set_title("Task Accuracy\n(method-dependent)")
    ax.legend(fontsize=8)
    ax.set_xticks(ns)
    ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Figure 6 — Task Arithmetic vs TIES-Merging: δ Detects Non-Compliance at n=2\n"
        "While TIES-Merging Preserves Accuracy Beyond TA's Collapse Point",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "fig6_method_comparison", cfg.results_dir)


def generate_all_figures(conditions, zones, stats_results, cfg,
                         conditions_ties=None):
    os.makedirs(cfg.results_dir, exist_ok=True)
    plot_drift_curve(conditions, zones, cfg)
    plot_accuracy_degradation(conditions, zones, cfg)
    plot_oversight_and_esys(conditions, zones, cfg)
    plot_compliance_heatmap(conditions, zones, cfg)
    plot_correlation(conditions, stats_results, cfg)
    if conditions_ties is not None:
        plot_fusion_method_comparison(conditions, conditions_ties, cfg)
    print(f"[visualize] All figures saved to {cfg.results_dir}/")

# EU AI Act Governance Experiment for Modular RAP/CAST Systems

Empirical derivation of EU AI Act (Regulation EU 2024/1689) compliance thresholds for Retrieval-Augmented Parameterisation (RAP) systems. Accompanies the paper:

> Silaparasetty, V. (2026). *Governing Modular AI Under the EU AI Act: Empirical Compliance Thresholds for Retrieval-Augmented Parameterisation.* Preprint.

---

## What this experiment does

Trains 7 domain-specific LoRA adapters ("cartridges") on publicly available Annex III high-risk datasets, then progressively fuses them via Task Arithmetic, measuring:

- **δ (Manifold Drift)** — activation space distance as cartridges accumulate
- **B(n) (Oversight Burden)** — fraction of low-confidence predictions requiring human review
- **E_sys (System Efficacy)** — accuracy gain per unit oversight cost
- **EU AI Act compliance zone** — Compliant / Caution / Non-Compliant per Art. 9, 14, 15

Produces 5 publication-quality figures and 3 JSON result files.

---

## Requirements

- Python 3.12
- Internet connection (datasets download from HuggingFace on first run; cached thereafter)
- ~6 GB disk space (model weights + dataset cache)
- ~1–2 GB RAM

No GPU required. The default model is `roberta-base` (125M parameters), which runs on CPU.

---

## Setup

```bash
git clone https://github.com/VinitaSilaparasetty/rap-eu-governance
cd rap-eu-governance
pip install -r requirements.txt
```

> **Note for Intel Mac users**: The `requirements.txt` pins `numpy==1.26.4` to maintain compatibility with PyTorch 2.2.2 (the maximum version available for macOS x86_64). Do not upgrade numpy without also upgrading PyTorch.

---

## Running the experiment

```bash
python experiment.py
```

This trains all 7 cartridges and runs all 7 fusion conditions. Expected runtime:

| Hardware | Estimated time |
|---|---|
| Intel Mac (CPU) | 15–30 min |
| Apple Silicon (CPU) | 8–15 min |
| NVIDIA GPU | 2–5 min |

### Quick test (3 cartridges only)

```bash
python experiment.py --max-cartridges 3
```

### Skip retraining (load saved adapters)

```bash
python experiment.py --skip-training
```

Adapters are saved to `results/adapters/` on the first run.

### Force a specific device

```bash
python experiment.py --device cpu
python experiment.py --device mps   # Apple Silicon
python experiment.py --device cuda  # NVIDIA GPU
```

---

## Outputs

All outputs are written to `results/`:

```
results/
├── conditions.json          # per-condition metrics (δ, accuracy, B(n), E_sys, zone)
├── statistics.json          # Pearson r, OLS R², Bonferroni tests
├── compliance.json          # EU AI Act zone classifications + empirical thresholds
├── adapters/                # saved LoRA adapter weights
│   ├── cartridge_1/
│   ├── cartridge_2/
│   └── ...
├── fig1_drift_curve.{pdf,png}
├── fig2_accuracy_degradation.{pdf,png}
├── fig3_oversight_esys.{pdf,png}
├── fig4_compliance_heatmap.{pdf,png}
└── fig5_delta_accuracy_correlation.{pdf,png}
```

---

## Exact reproducibility

The experiment is fully deterministic given the same seed and package versions.

**Seed**: `42` (set in `src/config.py` — controls dataset sampling, model initialisation, training order)

**Package versions** (from `requirements.txt`):

```
torch==2.2.2
transformers==4.44.2
peft==0.10.0
datasets==2.18.0
accelerate==0.28.0
numpy==1.26.4
scipy==1.11.4
scikit-learn==1.3.2
```

**To exactly reproduce the paper's results**:

1. Install the pinned requirements: `pip install -r requirements.txt`
2. Run with no flags: `python experiment.py`
3. Results in `results/conditions.json` will match Table 3 in the paper.

If you get different results, the most likely cause is a different package version. Check with:

```bash
pip show torch transformers peft numpy scipy | grep Version
```

---

## Datasets

All datasets are downloaded automatically from HuggingFace on first run. They are publicly available with no authentication required.

| Cartridge | HuggingFace ID | Config | EU AI Act Annex |
|---|---|---|---|
| 1 | `nguha/legalbench` | `corporate_lobbying` | III-5b |
| 2 | `nguha/legalbench` | `unfair_tos` | III-6 |
| 3 | `nguha/legalbench` | `overruling` | III-6 |
| 4 | `nguha/legalbench` | `hearsay` | III-6 |
| 5 | `nguha/legalbench` | `telemarketing_sales_rule` | III-5b |
| 6 | `takala/financial_phrasebank` | `sentences_allagree` | III-2 |
| 7 | `coastalcph/lex_glue` | `eurlex` | Annex III (all) |

---

## Modifying the experiment

**Change the base model** — edit `src/config.py`:
```python
base_model: str = "meta-llama/Llama-3.2-3B"  # requires GPU
```

**Change compliance thresholds** — edit `ExperimentConfig` in `src/config.py`:
```python
zone1_delta_max: float = 0.15   # Art. 9 caution
zone2_delta_max: float = 0.30   # Art. 9 non-compliant
zone1_burden_max: float = 0.20  # Art. 14 caution
zone2_burden_max: float = 0.40  # Art. 14 non-compliant
```

**Add a cartridge** — append an entry to `CARTRIDGE_REGISTRY` in `src/config.py`.

---

## Project structure

```
rap-eu-governance/
├── experiment.py          # main orchestration script
├── requirements.txt       # pinned dependencies
├── PAPER.md               # full research paper
└── src/
    ├── config.py          # ExperimentConfig + CARTRIDGE_REGISTRY
    ├── data_loader.py     # dataset loading and binarisation
    ├── cartridge.py       # LoRA training and Task Arithmetic fusion
    ├── metrics.py         # δ, B(n), E_sys, statistical tests
    ├── eu_compliance.py   # EU AI Act compliance zone classification
    └── visualize.py       # 5 publication figures
```

---

## Citation

```bibtex
@misc{silaparasetty2026governing,
  title   = {Governing Modular AI Under the EU AI Act: Empirical Compliance
             Thresholds for Retrieval-Augmented Parameterisation},
  author  = {Silaparasetty, Vinita},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/VinitaSilaparasetty/rap-eu-governance}
}
```

---

## License

AGPL-3.0. See `LICENSE`.

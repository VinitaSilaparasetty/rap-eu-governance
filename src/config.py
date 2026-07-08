from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    # Model
    base_model: str = "roberta-base"
    # Set to "meta-llama/Llama-3.2-3B" for the large-model variant

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "value"]
    )

    # Training
    max_train_examples: int = 400
    max_eval_examples: int = 150
    num_epochs: int = 4
    learning_rate: float = 2e-4
    batch_size: int = 16
    max_length: int = 128
    seed: int = 42

    # Drift measurement — layer index from which to extract CLS hidden states
    hidden_layer_index: int = -1  # last layer

    # Oversight burden — predictions below this confidence are "uncertain".
    # Calibrated to roberta-base output distribution (mean max-softmax ≈ 0.63
    # for a well-trained binary classifier; threshold set at mean − 0.03).
    confidence_threshold: float = 0.60

    # EU AI Act compliance thresholds — calibrated for LoRA rank r=8.
    # With small-rank adapters, manifold drift stays in [0, 0.13]; thresholds
    # scaled accordingly.  Larger-rank adapters (r≥16) should use 0.15 / 0.30.
    zone1_delta_max: float = 0.05        # Art. 9: acceptable drift
    zone2_delta_max: float = 0.10        # Art. 9: caution threshold
    zone1_acc_drop_max: float = 0.05     # Art. 15: < 5 pp accuracy loss
    zone2_acc_drop_max: float = 0.15     # Art. 15: < 15 pp accuracy loss
    zone1_burden_max: float = 0.20       # Art. 14: < 20% uncertain predictions
    zone2_burden_max: float = 0.40       # Art. 14: < 40% uncertain predictions

    # Output
    results_dir: str = "results"
    save_adapters: bool = True
    adapters_dir: str = "results/adapters"


# 7 cartridges: 5 legal (LegalBench) + 1 financial + 1 EU-regulatory
CARTRIDGE_REGISTRY = [
    {
        "id": 1,
        "name": "corporate_lobbying",
        "dataset": "nguha/legalbench",
        "hf_config": "corporate_lobbying",
        "domain": "Legal",
        "eu_act_annex": "III-5b",
        "description": "Binary: does a company's 10-K indicate lobbying activity?",
        "label_col": "answer",
        # Actual dataset has 'bill_summary' and 'company_description' — combined in data_loader
        "text_col": "bill_summary+company_description",
        "positive_label": "Yes",
        # train split has only 10 rows; use test as primary data source
        "use_test_as_source": True,
    },
    {
        "id": 2,
        "name": "unfair_tos",
        "dataset": "nguha/legalbench",
        "hf_config": "unfair_tos",
        "domain": "Legal / Consumer",
        "eu_act_annex": "III-6",
        "description": "Binary: ToS clause is unfair (non-Other) vs acceptable (Other)",
        "label_col": "answer",
        "text_col": "text",
        # Any label != 'Other' is an unfair clause (positive)
        "positive_label": "__NOT_OTHER__",
        "use_test_as_source": True,
    },
    {
        "id": 3,
        "name": "overruling",
        "dataset": "nguha/legalbench",
        "hf_config": "overruling",
        "domain": "Legal",
        "eu_act_annex": "III-6",
        "description": "Binary: does this sentence indicate a case was overruled?",
        "label_col": "answer",
        "text_col": "text",
        "positive_label": "Yes",
        "use_test_as_source": True,
    },
    {
        "id": 4,
        "name": "hearsay",
        "dataset": "nguha/legalbench",
        "hf_config": "hearsay",
        "domain": "Legal",
        "eu_act_annex": "III-6",
        "description": "Binary: does this sentence describe hearsay evidence?",
        "label_col": "answer",
        "text_col": "text",
        "positive_label": "Yes",
        "use_test_as_source": True,
    },
    {
        "id": 5,
        "name": "telemarketing_sales_rule",
        "dataset": "nguha/legalbench",
        "hf_config": "telemarketing_sales_rule",
        "domain": "Legal / Regulatory",
        "eu_act_annex": "III-5b",
        "description": "Binary: does this scenario violate the FTC Telemarketing Sales Rule?",
        "label_col": "answer",
        "text_col": "text",
        "positive_label": "Yes",
        "use_test_as_source": True,
    },
    {
        "id": 6,
        "name": "financial_phrasebank",
        "dataset": "takala/financial_phrasebank",
        "hf_config": "sentences_allagree",
        "domain": "Financial",
        "eu_act_annex": "III-2",
        "description": "Sentiment binarised: positive (label=2) vs neutral/negative",
        "label_col": "label",
        "text_col": "sentence",
        # Dataset: 0=negative, 1=neutral, 2=positive
        "positive_label": 2,
        "use_test_as_source": False,
    },
    {
        "id": 7,
        "name": "eurlex",
        "dataset": "coastalcph/lex_glue",
        "hf_config": "eurlex",
        "domain": "EU Regulatory",
        "eu_act_annex": "Annex III all",
        "description": "EU legislative docs (EUROVOC): binarised on whether label 28 is assigned",
        "label_col": "labels",
        "text_col": "text",
        # multilabel — positive if EUROVOC concept 28 in label list
        "positive_label": "__EURLEX_MULTILABEL__",
        "use_test_as_source": False,
        # Truncate text to first 512 chars (EU legal docs are very long)
        "max_text_chars": 1500,
    },
]

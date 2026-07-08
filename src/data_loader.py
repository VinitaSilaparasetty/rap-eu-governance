import random
import logging
from typing import Tuple, List, Dict, Any

import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets

from .config import ExperimentConfig, CARTRIDGE_REGISTRY

log = logging.getLogger(__name__)

# EUROVOC concept ID used as positive class for EURLEX binary task.
# Concept 28 = "economics" — one of the more balanced classes.
EURLEX_POSITIVE_LABEL = 28


def _get_text(ex: Dict, text_col: str, spec: Dict) -> str:
    """Extract and normalise the text field, handling combined-column specs."""
    max_chars = spec.get("max_text_chars", None)

    if "+" in text_col:
        # Concatenate multiple fields
        parts = [str(ex.get(c, "")).strip() for c in text_col.split("+")]
        text = " [SEP] ".join(p for p in parts if p)
    else:
        text = str(ex.get(text_col, "")).strip()

    if max_chars:
        text = text[:max_chars]
    return text


def _binarise(raw_label: Any, spec: Dict) -> int:
    """Convert a raw dataset label to binary 0/1."""
    positive = spec["positive_label"]

    if positive == "__NOT_OTHER__":
        return 1 if str(raw_label).strip() != "Other" else 0

    if positive == "__EURLEX_MULTILABEL__":
        if isinstance(raw_label, list):
            return 1 if EURLEX_POSITIVE_LABEL in raw_label else 0
        return 0

    if isinstance(positive, int):
        return 1 if raw_label == positive else 0

    return 1 if str(raw_label).strip() == str(positive) else 0


def _process_rows(rows: List[Dict], spec: Dict) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    text_col = spec["text_col"]
    label_col = spec["label_col"]

    for ex in rows:
        text = _get_text(ex, text_col, spec)
        if not text:
            continue
        raw_label = ex.get(label_col)
        if raw_label is None:
            continue
        texts.append(text)
        labels.append(_binarise(raw_label, spec))

    return texts, labels


def _balanced_sample(
    texts: List[str], labels: List[int], n: int, seed: int
) -> Tuple[List[str], List[int]]:
    """Return up to n/2 positive and n/2 negative examples, shuffled."""
    rng = random.Random(seed)
    pos = [(t, l) for t, l in zip(texts, labels) if l == 1]
    neg = [(t, l) for t, l in zip(texts, labels) if l == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    combined = pos[:half] + neg[:half]
    if not combined:
        return [], []
    rng.shuffle(combined)
    t, la = zip(*combined)
    return list(t), list(la)


def load_cartridge_data(
    spec: Dict, cfg: ExperimentConfig
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load and preprocess a single cartridge dataset.
    Returns (train_texts, train_labels, eval_texts, eval_labels).
    """
    dataset_id = spec["dataset"]
    hf_config = spec.get("hf_config")
    use_test_as_source = spec.get("use_test_as_source", False)

    log.info("Loading %s / %s", dataset_id, hf_config)

    try:
        ds = (
            load_dataset(dataset_id, hf_config, trust_remote_code=True)
            if hf_config
            else load_dataset(dataset_id, trust_remote_code=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load {dataset_id}/{hf_config}: {e}") from e

    available = list(ds.keys())

    if use_test_as_source:
        # LegalBench has ~5-10 train examples; use test as the main data pool
        test_split = "test" if "test" in available else available[0]
        all_rows = list(ds[test_split])
        all_texts, all_labels = _process_rows(all_rows, spec)

        # 80/20 split from the pooled data
        rng = random.Random(cfg.seed)
        indices = list(range(len(all_texts)))
        rng.shuffle(indices)
        split_idx = max(1, int(len(indices) * 0.8))
        train_idx, eval_idx = indices[:split_idx], indices[split_idx:]
        train_texts = [all_texts[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        eval_texts = [all_texts[i] for i in eval_idx]
        eval_labels = [all_labels[i] for i in eval_idx]

    else:
        train_split = "train" if "train" in available else available[0]
        test_split = "test" if "test" in available else (
            "validation" if "validation" in available else train_split
        )
        train_texts, train_labels = _process_rows(list(ds[train_split]), spec)
        eval_texts, eval_labels = _process_rows(list(ds[test_split]), spec)

        if test_split == train_split:
            # Single-split dataset: manual 80/20
            rng = random.Random(cfg.seed)
            idx = list(range(len(train_texts)))
            rng.shuffle(idx)
            cut = max(1, int(len(idx) * 0.8))
            eval_texts  = [train_texts[i] for i in idx[cut:]]
            eval_labels = [train_labels[i] for i in idx[cut:]]
            train_texts = [train_texts[i] for i in idx[:cut]]
            train_labels = [train_labels[i] for i in idx[:cut]]

    # Balance and cap
    train_texts, train_labels = _balanced_sample(
        train_texts, train_labels, cfg.max_train_examples, cfg.seed
    )
    eval_texts, eval_labels = _balanced_sample(
        eval_texts, eval_labels, cfg.max_eval_examples, cfg.seed + 1
    )

    if not train_texts:
        raise RuntimeError(
            f"No training examples for {spec['name']} — check label mapping. "
            f"positive_label={spec['positive_label']!r}"
        )

    pos_rate = sum(train_labels) / max(len(train_labels), 1)
    log.info(
        "%s — train: %d, eval: %d, pos_rate=%.2f",
        spec["name"], len(train_texts), len(eval_texts), pos_rate,
    )

    return train_texts, train_labels, eval_texts, eval_labels


def make_hf_dataset(texts: List[str], labels: List[int]) -> Dataset:
    return Dataset.from_dict({"text": texts, "label": labels})


def load_all_cartridges(cfg: ExperimentConfig) -> List[Dict[str, Any]]:
    """Return list of dicts with train/eval data for each cartridge."""
    result = []
    for spec in CARTRIDGE_REGISTRY:
        try:
            tr_t, tr_l, ev_t, ev_l = load_cartridge_data(spec, cfg)
            result.append({
                "spec": spec,
                "train": make_hf_dataset(tr_t, tr_l),
                "eval": make_hf_dataset(ev_t, ev_l),
            })
        except Exception as e:
            log.error("Skipping cartridge %s: %s", spec["name"], e)
    return result

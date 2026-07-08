"""
Cartridge training and fusion via Task Arithmetic (Ilharco et al., 2022).

Each LoRA adapter is a 'cartridge'.  Fusion averages the adapter weight
deltas — equivalent to model merging in the low-rank subspace.
"""

import os
import copy
import logging
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from datasets import Dataset
from tqdm import tqdm

from .config import ExperimentConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification head + base encoder
# ---------------------------------------------------------------------------

class CartridgeClassifier(nn.Module):
    """Base encoder (LoRA-adapted) + task-specific linear head."""

    def __init__(self, base_model: AutoModel, hidden_size: int, num_labels: int = 2):
        super().__init__()
        self.encoder = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # CLS token representation
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_hidden))

        result = {"logits": logits, "hidden": cls_hidden}
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            result["loss"] = loss
        return result


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def _tokenise(texts: List[str], tokenizer, max_length: int):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _make_loader(dataset: Dataset, tokenizer, cfg: ExperimentConfig, shuffle: bool):
    texts = dataset["text"]
    labels = torch.tensor(dataset["label"], dtype=torch.long)
    enc = _tokenise(texts, tokenizer, cfg.max_length)
    torch_dataset = torch.utils.data.TensorDataset(
        enc["input_ids"], enc["attention_mask"], labels
    )
    return DataLoader(torch_dataset, batch_size=cfg.batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Train a single cartridge
# ---------------------------------------------------------------------------

def train_cartridge(
    cartridge_id: int,
    train_dataset: Dataset,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[CartridgeClassifier, AutoTokenizer]:
    """Fine-tune a LoRA cartridge on one task dataset."""

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    base_encoder = AutoModel.from_pretrained(cfg.base_model)

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    lora_encoder = get_peft_model(base_encoder, lora_cfg)

    model = CartridgeClassifier(
        lora_encoder, base_encoder.config.hidden_size
    ).to(device)

    loader = _make_loader(train_dataset, tokenizer, cfg, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(loader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Cartridge {cartridge_id} epoch {epoch+1}", leave=False):
            input_ids, attn_mask, labels = [b.to(device) for b in batch]
            out = model(input_ids, attn_mask, labels)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += out["loss"].item()
        log.info("Cartridge %d epoch %d loss=%.4f", cartridge_id, epoch + 1, total_loss / len(loader))

    return model, tokenizer


def save_cartridge(model: CartridgeClassifier, cartridge_id: int, cfg: ExperimentConfig):
    """Persist adapter weights and classifier head."""
    out_dir = os.path.join(cfg.adapters_dir, f"cartridge_{cartridge_id}")
    os.makedirs(out_dir, exist_ok=True)
    model.encoder.save_pretrained(out_dir)
    torch.save(model.classifier.state_dict(), os.path.join(out_dir, "head.pt"))
    log.info("Saved cartridge %d to %s", cartridge_id, out_dir)


def load_cartridge(
    cartridge_id: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> CartridgeClassifier:
    """Reload a previously saved cartridge."""
    adapter_dir = os.path.join(cfg.adapters_dir, f"cartridge_{cartridge_id}")
    base_encoder = AutoModel.from_pretrained(cfg.base_model)
    lora_encoder = PeftModel.from_pretrained(base_encoder, adapter_dir)
    model = CartridgeClassifier(lora_encoder, base_encoder.config.hidden_size).to(device)
    head_path = os.path.join(adapter_dir, "head.pt")
    model.classifier.load_state_dict(torch.load(head_path, map_location=device))
    return model


# ---------------------------------------------------------------------------
# Cartridge fusion — shared utilities
# ---------------------------------------------------------------------------

def _extract_lora_deltas(model: CartridgeClassifier) -> Dict[str, torch.Tensor]:
    """Return a copy of LoRA adapter weight tensors (the task vector)."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.encoder.named_parameters()
        if "lora_" in name
    }


def _apply_deltas(
    primary_model: CartridgeClassifier,
    fused_deltas: Dict[str, torch.Tensor],
    device: torch.device,
) -> CartridgeClassifier:
    """Write fused_deltas into a deep copy of primary_model and return it."""
    fused_model = copy.deepcopy(primary_model).to(device)
    current_state = dict(fused_model.encoder.named_parameters())
    with torch.no_grad():
        for name, delta in fused_deltas.items():
            if name in current_state:
                current_state[name].copy_(delta.to(device))
    return fused_model


# ---------------------------------------------------------------------------
# Fusion Method 1 — Task Arithmetic (Ilharco et al., 2022)
# ---------------------------------------------------------------------------

def fuse_cartridges(
    cartridge_models: List[CartridgeClassifier],
    primary_model: CartridgeClassifier,
    cfg: ExperimentConfig,
    device: torch.device,
    scaling: float = 1.0,
) -> CartridgeClassifier:
    """
    Equal-weight Task Arithmetic:  τ_fused = (1/n) * Σ τᵢ
    The primary cartridge's classifier head is kept for evaluation.
    """
    n = len(cartridge_models)
    if n == 1:
        return cartridge_models[0]

    all_deltas = [_extract_lora_deltas(m) for m in cartridge_models]

    fused_deltas: Dict[str, torch.Tensor] = {}
    for key in all_deltas[0]:
        stacked = torch.stack([d[key].float() for d in all_deltas], dim=0)
        fused_deltas[key] = (scaling / n) * stacked.sum(0)

    return _apply_deltas(primary_model, fused_deltas, device)


# ---------------------------------------------------------------------------
# Fusion Method 2 — TIES-Merging (Yadav et al., 2023)
# ---------------------------------------------------------------------------

def fuse_cartridges_ties(
    cartridge_models: List[CartridgeClassifier],
    primary_model: CartridgeClassifier,
    cfg: ExperimentConfig,
    device: torch.device,
    trim_ratio: float = 0.20,
) -> CartridgeClassifier:
    """
    TIES-Merging in the LoRA subspace (Yadav et al., NeurIPS 2023).

    Three steps applied per LoRA parameter tensor:
      1. Trim  — zero out the bottom (1 - trim_ratio) of parameters by magnitude
                 within each cartridge's task vector.
      2. Elect — determine the dominant sign per parameter position via
                 majority vote across trimmed task vectors.
      3. Merge — average only the parameters that agree with the dominant sign,
                 ignoring trimmed-out (zero) entries.

    Motivation: equal-weight averaging causes sign conflicts between adapters
    trained on divergent tasks, collapsing softmax confidence to ~0.50.
    TIES resolves conflicts before averaging, preserving calibration.
    """
    n = len(cartridge_models)
    if n == 1:
        return cartridge_models[0]

    all_deltas = [_extract_lora_deltas(m) for m in cartridge_models]

    fused_deltas: Dict[str, torch.Tensor] = {}
    for key in all_deltas[0]:
        # Stack: shape [n, *param_shape]
        task_vecs = torch.stack([d[key].float() for d in all_deltas], dim=0)

        # Step 1: Trim — keep top trim_ratio of parameters by magnitude, per model
        flat = task_vecs.view(n, -1)            # [n, d]
        abs_flat = flat.abs()
        k = max(1, int(abs_flat.shape[1] * trim_ratio))
        # kth-largest value per model (threshold per row)
        kth_vals = abs_flat.topk(k, dim=1).values[:, -1]   # [n]
        thresh = kth_vals.view(n, *([1] * (task_vecs.dim() - 1)))
        trimmed = task_vecs * (task_vecs.abs() >= thresh).float()

        # Step 2: Elect — majority-vote dominant sign per parameter position
        sign_vote = trimmed.sign().sum(dim=0)   # [...], + means majority positive
        dominant = sign_vote.sign()             # -1, 0, or +1

        # Step 3: Merge — average only parameters agreeing with the dominant sign
        # A parameter agrees if its sign matches dominant, OR dominant is 0 (tie)
        dom_exp = dominant.unsqueeze(0)         # [1, *param_shape]
        agree = ((trimmed.sign() == dom_exp) | (dom_exp == 0)) & (trimmed != 0)
        filtered = trimmed * agree.float()
        counts = agree.float().sum(dim=0).clamp(min=1.0)
        fused_deltas[key] = filtered.sum(dim=0) / counts

    return _apply_deltas(primary_model, fused_deltas, device)


# ---------------------------------------------------------------------------
# Inference — returns logits AND hidden states
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: CartridgeClassifier,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        logits      — (N, num_labels) float32
        hidden      — (N, hidden_size) float32  [CLS vectors]
        true_labels — (N,) int
    """
    model.eval()
    loader = _make_loader(dataset, tokenizer, cfg, shuffle=False)

    all_logits, all_hidden, all_labels = [], [], []
    for batch in loader:
        input_ids, attn_mask, labels = [b.to(device) for b in batch]
        out = model(input_ids, attn_mask)
        all_logits.append(out["logits"].cpu().float().numpy())
        all_hidden.append(out["hidden"].cpu().float().numpy())
        all_labels.append(labels.cpu().numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_hidden, axis=0),
        np.concatenate(all_labels, axis=0),
    )

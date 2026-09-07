"""
src/sandisk_yield/training/trainer_b.py
======================================
Model B Deep Inspection PyTorch training loop with early stopping and loss weighting.
"""

from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sandisk_yield.models.model_b import ModelB, WaferBlockDataset
from sandisk_yield.training.losses import FocalLoss
from sandisk_yield.training.evaluation import compute_die_yield_metrics
from sandisk_yield.seed import get_device
from sandisk_yield.logging_utils import setup_logger

logger = setup_logger("sandisk_yield.trainer_b")


def train_model_b(
    train_dataset: WaferBlockDataset,
    val_dataset: Optional[WaferBlockDataset] = None,
    tabular_dim: int = 500,
    block_length: int = 2000,
    embedding_dim: int = 8,
    hidden_dim: int = 32,
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 4,
    device: Optional[torch.device] = None,
    dropout: float = 0.4,
    use_attention: bool = True
) -> Tuple[ModelB, Dict[str, list]]:
    """
    Trains Model B network end-to-end with Focal Loss and early stopping.
    """
    if device is None:
        device = get_device("auto")
        
    model = ModelB(
        tabular_dim=tabular_dim,
        block_length=block_length,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        block_feat_dim=train_dataset.X_blk_feats.shape[1],
        dropout=dropout,
        use_attention=use_attention
    ).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None
    
    criterion = FocalLoss(alpha=0.35, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    
    history = {"train_loss": [], "val_pr_auc": [], "val_f1": []}
    best_val_score = -1.0
    best_weights = None
    patience_counter = 0
    
    logger.info(f"Training Model B on {device} (Epochs: {epochs}, Batch size: {batch_size})...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            x_tab = batch["tab"].to(device)
            x_blk = batch["raw_block"].to(device)
            x_blk_feat = batch["blk_feats"].to(device)
            y = batch["y"].to(device)
            
            optimizer.zero_grad()
            logits, _, _ = model(x_tab, x_blk, x_blk_feat)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        avg_train_loss = total_loss / max(1, n_batches)
        history["train_loss"].append(avg_train_loss)
        
        # Validation pass
        if val_loader is not None:
            model.eval()
            val_probs = []
            val_trues = []
            with torch.no_grad():
                for batch in val_loader:
                    x_tab = batch["tab"].to(device)
                    x_blk = batch["raw_block"].to(device)
                    x_blk_feat = batch["blk_feats"].to(device)
                    logits, _, _ = model(x_tab, x_blk, x_blk_feat)
                    val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                    val_trues.extend(batch["y"].numpy())
                    
            m = compute_die_yield_metrics(np.array(val_trues), np.array(val_probs))
            val_pr_auc = m["pr_auc"]
            val_f1 = m["fail_f1"]
            history["val_pr_auc"].append(val_pr_auc)
            history["val_f1"].append(val_f1)
            scheduler.step(val_pr_auc)
            
            logger.info(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Val Fail F1: {val_f1:.4f}")
            
            if val_pr_auc > best_val_score:
                best_val_score = val_pr_auc
                # Bug: a shallow dict copy shares tensors with later epochs.
                best_weights = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        else:
            logger.info(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, history

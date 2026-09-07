"""
src/sandisk_yield/models/model_b.py
===================================
Model B: End-to-end Deep Inspection Model fusing tabular features & raw sub-die block readings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sandisk_yield.models.block_encoder import BlockEncoder
from sandisk_yield.models.fusion import EvidenceFusionNet
from sandisk_yield.features.blocks import parse_block_readings_row, compute_block_features_df
from sandisk_yield.seed import get_device


class WaferBlockDataset(Dataset):
    def __init__(
        self,
        X_tab: np.ndarray,
        block_strings: List[Union[str, np.ndarray]],
        X_blk_feats: np.ndarray,
        y: Optional[np.ndarray] = None,
        block_length: int = 2000
    ):
        assert len(X_tab) == len(block_strings) == len(X_blk_feats), "Model B row counts differ"
        assert y is None or len(y) == len(X_tab), "Model B targets differ in length"
        assert np.isfinite(X_tab).all() and np.isfinite(X_blk_feats).all(), "Nonfinite Model B inputs"
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.X_blk_feats = torch.tensor(X_blk_feats, dtype=torch.float32)
        self.block_length = block_length
        self.block_strings = block_strings
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_tab)

    def __getitem__(self, idx):
        raw_blk = parse_block_readings_row(self.block_strings[idx], expected_len=self.block_length)
        assert np.isfinite(raw_blk).all(), "Nonfinite raw block readings"
        # Reshape to (1, block_length)
        blk_tensor = torch.tensor(raw_blk, dtype=torch.float32).unsqueeze(0)
        
        item = {
            "tab": self.X_tab[idx],
            "raw_block": blk_tensor,
            "blk_feats": self.X_blk_feats[idx]
        }
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


class ModelB(nn.Module):
    """
    Combined Deep Inspection Architecture:
      Raw Block -> BlockEncoder -> Block Embedding
      Tabular + Block Embedding + Block Anomaly Features -> EvidenceFusionNet -> p_B
    """

    def __init__(
        self,
        tabular_dim: int,
        block_length: int = 2000,
        embedding_dim: int = 8,
        block_feat_dim: int = 23,
        hidden_dim: int = 32,
        dropout: float = 0.4,
        use_attention: bool = True
    ):
        super().__init__()
        self.tabular_dim = tabular_dim
        self.block_length = block_length
        self.embedding_dim = embedding_dim
        self.block_feat_dim = block_feat_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_attention = use_attention
        
        self.encoder = BlockEncoder(block_length=block_length, embedding_dim=embedding_dim,
                                    use_attention=use_attention, dropout=dropout)
        self.fusion = EvidenceFusionNet(
            tabular_dim=tabular_dim,
            block_embed_dim=embedding_dim,
            block_feat_dim=block_feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(
        self,
        x_tab: torch.Tensor,
        x_raw_block: torch.Tensor,
        x_blk_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode block
        blk_emb, attn_weights = self.encoder(x_raw_block)
        # Fuse evidence
        logits, modal_weights = self.fusion(x_tab, blk_emb, x_blk_feats)
        return logits, modal_weights, attn_weights

    def predict_proba(
        self,
        X_tab: np.ndarray,
        block_strings: List[Union[str, np.ndarray]],
        X_blk_feats: np.ndarray,
        batch_size: int = 64,
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        if device is None:
            device = get_device("auto")
            
        self.to(device)
        self.eval()
        dataset = WaferBlockDataset(X_tab, block_strings, X_blk_feats, block_length=self.block_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        probs = []
        with torch.no_grad():
            for batch in loader:
                x_tab = batch["tab"].to(device)
                x_blk = batch["raw_block"].to(device)
                x_blk_feat = batch["blk_feats"].to(device)
                logits, _, _ = self(x_tab, x_blk, x_blk_feat)
                batch_probs = torch.sigmoid(logits).cpu().numpy()
                probs.extend(batch_probs)
                
        p1 = np.array(probs, dtype=np.float32)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def save(self, path: Union[str, Path]):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "tabular_dim": self.tabular_dim,
            "block_length": self.block_length,
            "embedding_dim": self.embedding_dim,
            "block_feat_dim": self.block_feat_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "use_attention": self.use_attention,
        }, p)

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[torch.device] = None) -> "ModelB":
        if device is None:
            device = get_device("auto")
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            tabular_dim=checkpoint["tabular_dim"],
            block_length=checkpoint["block_length"],
            embedding_dim=checkpoint["embedding_dim"],
            block_feat_dim=checkpoint.get("block_feat_dim", 23),
            hidden_dim=checkpoint.get("hidden_dim", 128),
            dropout=checkpoint.get("dropout", 0.2),
            use_attention=checkpoint.get("use_attention", True)
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model

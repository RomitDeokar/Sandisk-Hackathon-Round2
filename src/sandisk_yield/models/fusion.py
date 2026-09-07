"""
src/sandisk_yield/models/fusion.py
==================================
Multi-resolution evidence fusion network with learned gating.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceFusionNet(nn.Module):
    """
    Fuses Model A evidence (tabular representation/score) with
    the deep block embedding and explicit block anomaly features.
    Computes dynamic modal gating weights:
        w_tabular, w_block_embed, w_block_anomaly
    """

    def __init__(
        self,
        tabular_dim: int,
        block_embed_dim: int = 8,
        block_feat_dim: int = 23,
        hidden_dim: int = 32,
        dropout: float = 0.4
    ):
        super().__init__()
        
        # Sub-branch projections
        self.tab_proj = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.block_emb_proj = nn.Sequential(
            nn.Linear(block_embed_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.block_feat_proj = nn.Sequential(
            nn.Linear(block_feat_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Modal Gating Network: computes 3 softmax contribution weights
        self.gating_net = nn.Sequential(
            nn.Linear(64 + 64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Combined classification head
        self.classifier = nn.Sequential(
            # BatchNorm failed on a final training batch containing one die.
            nn.Dropout(dropout),
            nn.Linear(64 + 64 + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        x_tab: torch.Tensor,
        x_blk_emb: torch.Tensor,
        x_blk_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_tab = self.tab_proj(x_tab)
        h_emb = self.block_emb_proj(x_blk_emb)
        h_feat = self.block_feat_proj(x_blk_feat)
        
        # Compute modal contribution weights
        raw_concat = torch.cat([h_tab, h_emb, h_feat], dim=1)
        modal_weights = torch.softmax(self.gating_net(raw_concat), dim=-1)  # (batch, 3)
        
        # Modulate representations
        w_tab = modal_weights[:, 0:1]
        w_emb = modal_weights[:, 1:2]
        w_feat = modal_weights[:, 2:3]
        
        fused = torch.cat([h_tab * (1.0 + w_tab), h_emb * (1.0 + w_emb), h_feat * (1.0 + w_feat)], dim=1)
        logits = self.classifier(fused)
        
        return logits.squeeze(-1), modal_weights

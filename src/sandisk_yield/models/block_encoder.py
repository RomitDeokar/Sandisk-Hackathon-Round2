"""
src/sandisk_yield/models/block_encoder.py
========================================
PyTorch Conv1D neural block encoder for sub-die raw sequence analysis.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockEncoder(nn.Module):
    """
    Compact 1D Convolutional Neural Network with self-attention/gating
    for processing [batch, 1, block_length] sub-die electrical waveforms.
    Outputs:
      - embedding: (batch, embedding_dim)
      - attention_weights: (batch, seq_len_pooled) for explainability
    """

    def __init__(self, block_length: int = 2000, embedding_dim: int = 8,
                 use_attention: bool = True, dropout: float = 0.4):
        super().__init__()
        self.block_length = block_length
        self.embedding_dim = embedding_dim
        
        # Ten wafers provide few independent examples: one small convolution.
        self.use_attention = use_attention
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, stride=2, padding=7)
        self.attention = nn.Linear(8, 1)
        
        # Adaptive pooling & projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(8 * 2, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, 1, block_length)
        h = F.relu(self.conv1(x))
        
        # Compute attention scores over sequence
        if self.use_attention:
            attn_raw = self.attention(h.transpose(1, 2)).transpose(1, 2)
            attn_weights = torch.softmax(attn_raw, dim=-1)
        else:
            attn_weights = torch.full_like(h[:, :1], 1.0 / h.shape[-1])
        
        # Weighted pooling + global max pooling for localized sharp spikes
        weighted_h = torch.sum(h * attn_weights, dim=-1, keepdim=True)  # (batch, 8, 1)
        max_h = self.global_max_pool(h)  # (batch, 8, 1)
        
        pooled = torch.cat([weighted_h, max_h], dim=1).squeeze(-1)  # (batch, 16)
        embedding = self.fc(pooled)  # (batch, embedding_dim)
        
        return embedding, attn_weights.squeeze(1)

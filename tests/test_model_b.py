"""
tests/test_model_b.py
======================
Unit tests for Model B forward pass, tensor dimensions, and attention extraction.
"""

import numpy as np
import pytest
import torch

from sandisk_yield.models.block_encoder import BlockEncoder
from sandisk_yield.models.fusion import EvidenceFusionNet
from sandisk_yield.models.model_b import ModelB


def test_block_encoder_forward():
    batch_size = 4
    block_len = 2000
    embed_dim = 64
    
    encoder = BlockEncoder(block_length=block_len, embedding_dim=embed_dim)
    x = torch.randn(batch_size, 1, block_len)
    
    emb, attn = encoder(x)
    assert emb.shape == (batch_size, embed_dim)
    assert attn.shape[0] == batch_size
    # Saliency weights must sum to approximately 1
    assert torch.allclose(attn.sum(dim=-1), torch.ones(batch_size), atol=1e-4)


def test_evidence_fusion_net():
    batch_size = 4
    tab_dim = 100
    fusion = EvidenceFusionNet(tabular_dim=tab_dim, block_embed_dim=64, block_feat_dim=23)
    
    x_tab = torch.randn(batch_size, tab_dim)
    x_emb = torch.randn(batch_size, 64)
    x_feat = torch.randn(batch_size, 23)
    
    logits, modal_weights = fusion(x_tab, x_emb, x_feat)
    assert logits.shape == (batch_size,)
    assert modal_weights.shape == (batch_size, 3)
    # Modal weights must sum to 1
    assert torch.allclose(modal_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-4)


def test_model_b_end_to_end():
    batch_size = 3
    tab_dim = 50
    block_len = 500
    
    model_b = ModelB(tabular_dim=tab_dim, block_length=block_len, embedding_dim=32)
    
    x_tab = np.random.randn(batch_size, tab_dim).astype(np.float32)
    x_blk_feats = np.random.randn(batch_size, 23).astype(np.float32)
    block_strings = [" ".join(["100.0"] * block_len)] * batch_size
    
    probs = model_b.predict_proba(x_tab, block_strings, x_blk_feats, batch_size=2)
    assert probs.shape == (batch_size, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

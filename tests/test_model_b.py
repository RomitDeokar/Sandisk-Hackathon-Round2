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


@pytest.mark.parametrize("value", [None, "", "1 2", "1 2 3 4 5", "1 2 bad 4",
                                  "1 2 nan 4", [1, 2, np.inf, 4], [[1, 2], [3, 4]]])
def test_block_parser_rejects_corrupt_measurements(value):
    from sandisk_yield.features.blocks import parse_block_readings_row
    with pytest.raises(ValueError):
        parse_block_readings_row(value, expected_len=4)


def test_compact_statistics_are_finite_and_label_independent():
    import pandas as pd
    from sandisk_yield.features.blocks import compress_block_readings_stats
    df = pd.DataFrame({"block_readings": [np.ones(32), np.arange(32)], "label": [0, 1]}, index=[8, 2])
    features = compress_block_readings_stats(df, 32)
    assert features.shape == (2, 27)
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[8, "blk_skew"] == 0
    assert features.loc[8, "blk_kurtosis"] == 0
    assert features.loc[2, "blk_variance"] == pytest.approx(np.var(np.arange(32)))
    pd.testing.assert_frame_equal(features, compress_block_readings_stats(df.assign(label=[1, 0]), 32))


@pytest.mark.parametrize("mode", ["stats", "global", "zonal"])
def test_tree_resolver_serialization_and_schema(mode, tmp_path):
    import pandas as pd
    import joblib
    from sandisk_yield.training.cross_validation import BlockFeatureModel
    rng = np.random.default_rng(12)
    X = pd.DataFrame({"feature_1": rng.normal(size=80)}, index=np.arange(80) * 2)
    raw = pd.DataFrame({"block_readings": list(rng.normal(size=(80, 32)))}, index=X.index)
    y = pd.Series(np.arange(80) % 5 == 0, index=X.index).astype(int)
    model = BlockFeatureModel(mode, block_length=32,
        model_params={"n_estimators": 8, "n_jobs": 1, "verbosity": -1}).fit(X, raw, y)
    expected = model.predict_proba(X, raw.block_readings.tolist())
    path = tmp_path / "resolver.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(X, raw.block_readings.tolist()), expected)
    assert np.isfinite(expected).all()
    with pytest.raises(AssertionError):
        model.predict_proba(X.rename(columns={"feature_1": "wrong"}), raw.block_readings.tolist())

"""Shared wafer folds with training-only preprocessing for both model families."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sandisk_yield.data.alignment import assert_feature_alignment, assert_row_alignment, log_feature_uniqueness
from sandisk_yield.data.splitter import make_grouped_folds
from sandisk_yield.features.pipeline import FeaturePipeline
from sandisk_yield.features.blocks import (compute_block_features_df,
    compress_block_readings_global, compress_block_readings_zonal)
from sandisk_yield.models.model_a import ModelA
from sandisk_yield.models.model_b import WaferBlockDataset
from sandisk_yield.training.trainer_b import train_model_b
from sandisk_yield.training.evaluation import summarize_oof
from sandisk_yield.schema import create_new_failure_target
from sandisk_yield.seed import seed_everything
from sandisk_yield.logging_utils import setup_logger

logger = setup_logger("sandisk_yield.cv")


class BlockFeatureModel:
    """Persisted adapter preserving the cascade's Model B prediction interface.

    CNN uses the existing end-to-end neural fusion; global/zonal use a small
    LightGBM over tabular + summary features. Thus comparisons include the
    downstream learner change, not only a controlled encoder ablation.
    """
    def __init__(self, mode="cnn", block_length=2000, model_params=None, training_params=None,
                 model_type="lightgbm"):
        if mode not in {"cnn", "global", "zonal"}:
            raise ValueError("Unknown block mode")
        self.mode, self.block_length = mode, block_length
        self.model_params, self.training_params = model_params, training_params or {}
        self.model_type = model_type

    def block_features(self, raw):
        func = {"cnn": compute_block_features_df, "global": compress_block_readings_global,
                "zonal": compress_block_readings_zonal}[self.mode]
        return func(raw, expected_len=self.block_length)

    def fit(self, X, raw, y):
        assert_row_alignment(X, raw, y)
        self.feature_names_ = list(X.columns)
        blocks = self.block_features(raw)
        self.block_columns_ = list(blocks.columns)
        log_feature_uniqueness(blocks, f"Model B {self.mode} train")
        if self.mode == "cnn":
            # Neural inputs need comparable scales; use only fit-fold statistics.
            self.tab_mean_ = X.mean().to_numpy()
            self.tab_scale_ = X.std(ddof=0).clip(lower=1e-6).to_numpy()
            self.blk_mean_ = blocks.mean().to_numpy()
            self.blk_scale_ = blocks.std(ddof=0).clip(lower=1e-6).to_numpy()
            dataset = WaferBlockDataset(
                (X.to_numpy() - self.tab_mean_) / self.tab_scale_,
                raw["block_readings"].tolist(),
                (blocks.to_numpy() - self.blk_mean_) / self.blk_scale_,
                y.to_numpy(), self.block_length)
            # Fixed epochs: outer validation labels must not select checkpoints.
            self.model, _ = train_model_b(dataset, block_length=self.block_length,
                tabular_dim=X.shape[1], **self.training_params)
            logger.info("CNN embedding=%s dropout=%s weight_decay=%s attention=%s",
                        self.model.embedding_dim, self.model.dropout,
                        self.training_params.get("weight_decay", 1e-4), self.model.use_attention)
        else:
            self.model = ModelA(model_type=self.model_type, params=self.model_params)
            self.model.fit(pd.concat([X, blocks], axis=1), y)
        return self

    def predict_proba(self, X_tab, block_strings, X_blk_feats=None, batch_size=64, device=None):
        if isinstance(X_tab, pd.DataFrame):
            assert_feature_alignment(self.feature_names_, X_tab.columns)
        X = pd.DataFrame(np.asarray(X_tab), columns=self.feature_names_)
        raw = pd.DataFrame({"block_readings": block_strings})
        assert len(X) == len(raw), "Block/tabular row count mismatch"
        blocks = self.block_features(raw)
        assert_feature_alignment(self.block_columns_, blocks.columns)
        # Recompute mode-specific summaries; callers historically pass 23 anomaly
        # columns, which are not the schema for global/zonal modes.
        if self.mode == "cnn":
            return self.model.predict_proba(
                (X.to_numpy() - self.tab_mean_) / self.tab_scale_, block_strings,
                (blocks.to_numpy() - self.blk_mean_) / self.blk_scale_, batch_size, device)
        return self.model.predict_proba(pd.concat([X, blocks], axis=1))


def train_models_cv(df, modes=("cnn",), feature_params=None, model_params=None,
                    training_params=None, block_length=2000, n_splits=5, seed=42,
                    model_type="lightgbm"):
    y_all, eligible = create_new_failure_target(df)
    assert_row_alignment(df, y_all, eligible)
    assert df["wafer_id"].notna().all(), "Missing wafer IDs"
    assert y_all.loc[eligible].isin([0, 1]).all(), "Missing/nonbinary eligible targets"
    raw_eligible = df.loc[eligible]
    y = y_all.loc[eligible]
    folds = make_grouped_folds(raw_eligible, n_splits=n_splits, seed=seed)
    names = ["Model A"] + [f"Model B ({mode})" for mode in modes]
    oof = {name: np.full(len(y), np.nan) for name in names}
    seen = np.zeros(len(y), dtype=int)
    for fold, (tr, va) in enumerate(folds, 1):
        train_raw, val_raw = raw_eligible.iloc[tr], raw_eligible.iloc[va]
        assert set(train_raw.wafer_id).isdisjoint(set(val_raw.wafer_id))
        if y.iloc[tr].nunique() != 2:
            raise ValueError(f"Fold {fold} training partition lacks one class")
        if y.iloc[va].nunique() != 2:
            logger.warning("Fold %s validation has one class; PR-AUC is degenerate", fold)
        pipeline = FeaturePipeline(**(feature_params or {}))
        # Bug fix: supervised feature selection and imputation belong inside CV.
        pipeline.fit(train_raw, y.iloc[tr])
        # Include old failures when computing wafer neighborhoods, then select
        # exactly the same eligible row index as labels and OOF assignments.
        full_tr = df.loc[df.wafer_id.isin(train_raw.wafer_id)]
        full_va = df.loc[df.wafer_id.isin(val_raw.wafer_id)]
        X_tr = pipeline.transform(full_tr).loc[train_raw.index]
        X_va = pipeline.transform(full_va).loc[val_raw.index]
        assert_feature_alignment(X_tr.columns, X_va.columns)
        assert_row_alignment(X_tr, train_raw, y.iloc[tr])
        assert_row_alignment(X_va, val_raw, y.iloc[va])
        log_feature_uniqueness(X_tr, f"Fold {fold} train")
        log_feature_uniqueness(X_va, f"Fold {fold} validation")
        for name in names:
            seed_everything(seed + fold)
            if name == "Model A":
                model = ModelA(model_type=model_type, params=model_params).fit(X_tr, y.iloc[tr])
                p_tr = model.predict_proba(X_tr)[:, 1]
                p_va = model.predict_proba(X_va)[:, 1]
            else:
                mode = name[len("Model B ("):-1]
                model = BlockFeatureModel(mode, block_length, model_params, training_params, model_type).fit(X_tr, train_raw, y.iloc[tr])
                p_tr = model.predict_proba(X_tr, train_raw.block_readings.tolist())[:, 1]
                p_va = model.predict_proba(X_va, val_raw.block_readings.tolist())[:, 1]
            oof[name][va] = p_va
            logger.info("%s fold %s: train PR-AUC=%.6f validation PR-AUC=%.6f; train fails=%s val fails=%s",
                        name, fold, average_precision_score(y.iloc[tr], p_tr),
                        average_precision_score(y.iloc[va], p_va), y.iloc[tr].sum(), y.iloc[va].sum())
        seen[va] += 1
    assert (seen == 1).all(), "Each eligible row must be held out exactly once"
    assert all(np.isfinite(p).all() for p in oof.values()), "Incomplete/nonfinite OOF predictions"
    summaries = {name: summarize_oof(y.to_numpy(), p, folds) for name, p in oof.items()}
    logger.warning("Thresholds use pooled OOF labels: tuned F1/recall/precision/accuracy are descriptive, not nested-CV estimates; std uses ddof=0.")
    # Refit deployable preprocessing and models only after OOF evaluation.
    pipeline = FeaturePipeline(**(feature_params or {})).fit(raw_eligible, y)
    X = pipeline.transform(df).loc[raw_eligible.index]
    models = {"Model A": ModelA(model_type=model_type, params=model_params).fit(X, y)}
    for mode in modes:
        seed_everything(seed)
        models[f"Model B ({mode})"] = BlockFeatureModel(mode, block_length, model_params, training_params, model_type).fit(X, raw_eligible, y)
    return pipeline, models, oof, summaries, folds

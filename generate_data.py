"""
==============================================================
Synthetic Die-Level Feature Generator for Hackathon
==============================================================
Generates realistic die-level parametric features on top of
WM-811K wafer maps for multi-resolution die yield prediction.

Important:
    Measurements are generated from pre-target latent process
    conditions. Final failures are sampled only AFTER die-level
    and block-level measurements are generated. This prevents
    target leakage.

Usage:
    python generate_data.py
    python generate_data.py --num_wafers 500
    python generate_data.py --csv

Prerequisites:
    Download WM-811K dataset (LSWMD.pkl) from Kaggle:
    https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
    Place it at: data/LSWMD.pkl
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Auto-generate feature definitions from compact config
    config["features"] = generate_feature_definitions(config)
    return config


def generate_feature_definitions(config):
    """
    Auto-generate feature definitions deterministically from config.

    Each feature contains:
        name
        base_mean
        base_std
        fail_shift

    fail_shift is used as the feature's sensitivity to latent
    process risk. It is NOT applied using the final target label.
    """
    n = config["num_features"]
    fg = config["feature_generation"]

    feat_rng = np.random.default_rng(config["seed"] + 1000)

    mean_range = fg["base_mean_range"]
    cv_range = fg["cv_range"]
    shift_frac_range = fg["fail_shift_fraction_range"]
    log_scale = fg.get("log_scale_mean", True)

    features = []

    for i in range(n):

        # Sample base mean
        if log_scale:
            log_low = np.log10(max(mean_range[0], 1e-6))
            log_high = np.log10(mean_range[1])
            base_mean = 10 ** feat_rng.uniform(log_low, log_high)
        else:
            base_mean = feat_rng.uniform(
                mean_range[0],
                mean_range[1]
            )

        # Randomly make some features negative
        if feat_rng.random() < 0.2:
            base_mean = -base_mean

        # Sample coefficient of variation
        cv = feat_rng.uniform(
            cv_range[0],
            cv_range[1]
        )

        base_std = abs(base_mean) * cv

        # Feature sensitivity to latent process risk
        shift_frac = feat_rng.uniform(
            shift_frac_range[0],
            shift_frac_range[1]
        )

        sign = feat_rng.choice([-1, 1])

        fail_shift = (
            sign *
            shift_frac *
            base_std
        )

        features.append({
            "name": f"feature_{i + 1}",
            "base_mean": float(base_mean),
            "base_std": float(base_std),
            "fail_shift": float(fail_shift),
        })

    return features


def load_wm811k(pkl_path):
    """
    Load WM-811K dataset.

    Returns:
        labeled wafers
        none-type wafers

    Each wafer map:
        0 = no die
        1 = pass
        2 = fail
    """

    if not os.path.exists(pkl_path):
        print(
            f"ERROR: WM-811K file not found at '{pkl_path}'"
        )
        print(
            "Please download LSWMD.pkl from:"
        )
        print(
            "  https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map"
        )
        print(
            f"And place it at: {pkl_path}"
        )
        sys.exit(1)

    print(
        f"Loading WM-811K from {pkl_path}..."
    )

    try:
        df = pd.read_pickle(pkl_path)

    except (ModuleNotFoundError, UnicodeDecodeError):

        # Compatibility shim for old WM-811K pickle
        import types

        import pandas.core.indexes.base
        import pandas.core.indexes.range

        indexes_pkg = types.ModuleType(
            "pandas.indexes"
        )

        indexes_pkg.__path__ = []

        sys.modules["pandas.indexes"] = indexes_pkg
        sys.modules[
            "pandas.indexes.base"
        ] = pandas.core.indexes.base

        sys.modules[
            "pandas.indexes.range"
        ] = pandas.core.indexes.range

        try:
            import pandas.core.indexes.numeric

            sys.modules[
                "pandas.indexes.numeric"
            ] = pandas.core.indexes.numeric

        except ModuleNotFoundError:

            sys.modules[
                "pandas.indexes.numeric"
            ] = pandas.core.indexes.base

            sys.modules[
                "pandas.core.indexes.numeric"
            ] = pandas.core.indexes.base

        with open(pkl_path, "rb") as f:
            df = pickle.load(
                f,
                encoding="latin1"
            )

    # Labeled failure wafers
    labeled = df[
        df["failureType"].apply(
            lambda x:
                isinstance(x, np.ndarray)
                and len(x) > 0
                and x[0][0] != "none"
                and x[0][0] != ""
        )
    ].copy()

    # None-type wafers
    none_wafers = df[
        df["failureType"].apply(
            lambda x:
                isinstance(x, np.ndarray)
                and len(x) > 0
                and x[0][0] == "none"
        )
    ].copy()

    print(
        f"  Found {len(labeled)} labeled-failure wafers, "
        f"{len(none_wafers)} none-type wafers"
    )

    return labeled, none_wafers


def select_wafers(
    labeled_df,
    none_df,
    num_wafers,
    target_fail_rate,
    rng
):
    """
    Select a mix of failure-pattern and none-type wafers.

    target_fail_rate controls the approximate proportion of
    failure-pattern wafers.
    """

    avg_internal_fail_rate = 0.12

    fraction_failure_wafers = min(
        target_fail_rate /
        avg_internal_fail_rate,
        0.8
    )

    fraction_failure_wafers = max(
        fraction_failure_wafers,
        0.05
    )

    num_failure_wafers = min(
        int(num_wafers * fraction_failure_wafers),
        len(labeled_df)
    )

    num_none_wafers = min(
        num_wafers - num_failure_wafers,
        len(none_df)
    )

    failure_idx = rng.choice(
        len(labeled_df),
        size=num_failure_wafers,
        replace=False
    )

    none_idx = rng.choice(
        len(none_df),
        size=num_none_wafers,
        replace=False
    )

    wafer_maps = []
    wafer_ids = []

    for i, idx in enumerate(failure_idx):

        wm = labeled_df.iloc[idx]["waferMap"]

        wafer_maps.append(wm)

        wafer_ids.append(
            f"W_F_{i:04d}"
        )

    for i, idx in enumerate(none_idx):

        wm = none_df.iloc[idx]["waferMap"]

        wafer_maps.append(wm)

        wafer_ids.append(
            f"W_N_{i:04d}"
        )

    # Shuffle wafers
    order = rng.permutation(
        len(wafer_maps)
    )

    wafer_maps = [
        wafer_maps[i]
        for i in order
    ]

    wafer_ids = [
        wafer_ids[i]
        for i in order
    ]

    return wafer_maps, wafer_ids


def compute_radial_map(wafer_map):
    """Compute normalized radial distance from wafer center."""

    rows, cols = wafer_map.shape

    cy = rows / 2.0
    cx = cols / 2.0

    y_coords, x_coords = np.mgrid[
        0:rows,
        0:cols
    ]

    radial = np.sqrt(
        (y_coords - cy) ** 2 +
        (x_coords - cx) ** 2
    )

    max_r = radial.max()

    if max_r > 0:
        radial = radial / max_r

    return radial


def compute_linear_gradient(wafer_map, rng):
    """
    Compute a random linear process gradient across the wafer.
    """

    rows, cols = wafer_map.shape

    angle = rng.uniform(
        0,
        2 * np.pi
    )

    y_coords, x_coords = np.mgrid[
        0:rows,
        0:cols
    ]

    yn = (
        y_coords - rows / 2
    ) / (rows / 2)

    xn = (
        x_coords - cols / 2
    ) / (cols / 2)

    gradient = (
        np.cos(angle) * xn +
        np.sin(angle) * yn
    )

    return gradient


def compute_neighborhood_fail_density(
    wafer_map,
    window_size
):
    """
    Compute local density of ORIGINAL WM-811K failures.

    This is safe because it uses old_label information only,
    not newly generated target labels.
    """

    fail_mask = (
        wafer_map == 2
    ).astype(float)

    valid_mask = (
        wafer_map > 0
    ).astype(float)

    fail_sum = uniform_filter(
        fail_mask,
        size=window_size,
        mode="constant",
        cval=0.0
    )

    valid_sum = uniform_filter(
        valid_mask,
        size=window_size,
        mode="constant",
        cval=0.0
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):

        density = np.where(
            valid_sum > 0,
            fail_sum / valid_sum,
            0.0
        )

    return density


def compute_zone_features(
    wafer_map,
    features_grid,
    zone_rows,
    zone_cols
):
    """
    Compute zone-level aggregated features.

    Kept for reference/backward compatibility.
    """

    rows, cols = wafer_map.shape
    n_features = features_grid.shape[2]

    zone_mean_grid = np.zeros_like(
        features_grid
    )

    zone_var_grid = np.zeros_like(
        features_grid
    )

    zone_yield_grid = np.zeros(
        (rows, cols),
        dtype=float
    )

    row_edges = np.linspace(
        0,
        rows,
        zone_rows + 1,
        dtype=int
    )

    col_edges = np.linspace(
        0,
        cols,
        zone_cols + 1,
        dtype=int
    )

    for zr in range(zone_rows):

        for zc in range(zone_cols):

            r_start = row_edges[zr]
            r_end = row_edges[zr + 1]

            c_start = col_edges[zc]
            c_end = col_edges[zc + 1]

            zone_mask = (
                wafer_map[
                    r_start:r_end,
                    c_start:c_end
                ] > 0
            )

            zone_features = (
                features_grid[
                    r_start:r_end,
                    c_start:c_end
                ]
            )

            if zone_mask.sum() > 0:

                valid_features = (
                    zone_features[zone_mask]
                )

                z_mean = (
                    valid_features.mean(
                        axis=0
                    )
                )

                z_var = (
                    valid_features.var(
                        axis=0
                    )
                )

                zone_pass = (
                    wafer_map[
                        r_start:r_end,
                        c_start:c_end
                    ] == 1
                ).sum()

                zone_total = (
                    zone_mask.sum()
                )

                z_yield = (
                    zone_pass /
                    zone_total
                )

            else:

                z_mean = np.zeros(
                    n_features
                )

                z_var = np.zeros(
                    n_features
                )

                z_yield = 1.0

            zone_mean_grid[
                r_start:r_end,
                c_start:c_end
            ] = z_mean

            zone_var_grid[
                r_start:r_end,
                c_start:c_end
            ] = z_var

            zone_yield_grid[
                r_start:r_end,
                c_start:c_end
            ] = z_yield

    return (
        zone_mean_grid,
        zone_var_grid,
        zone_yield_grid
    )


def generate_latent_process_risk(
    wafer_map,
    config,
    rng
):
    """
    Generate a PRE-TARGET latent process-risk field.

    This represents hidden manufacturing/process variation that
    exists before the final test outcome is realized.

    It depends only on:
        - wafer geometry
        - original WM-811K failures
        - spatial process variation
        - random wafer-level process variation

    It does NOT depend on newly generated failures.

    The same latent process risk is later used to generate:
        1. die-level measurements
        2. block-level measurements

    Only after those measurements are generated do we sample
    the final new-failure outcome.
    """

    rows, cols = wafer_map.shape

    radial_strength = config[
        "radial_gradient_strength"
    ]

    linear_strength = config[
        "linear_gradient_strength"
    ]

    window_size = config[
        "neighborhood_window"
    ]

    # Pre-target spatial fields
    radial_map = compute_radial_map(
        wafer_map
    )

    linear_map = compute_linear_gradient(
        wafer_map,
        rng
    )

    old_fail_density = (
        compute_neighborhood_fail_density(
            wafer_map,
            window_size
        )
    )

    # Independent wafer-level process condition
    wafer_process = rng.normal(
        0.0,
        0.35
    )

    # Normalize old failure density approximately
    density_component = (
        old_fail_density * 2.0
    )

    # Center radial component
    radial_component = (
        radial_map - 0.5
    )

    # Normalize linear gradient
    linear_component = (
        linear_map / 2.0
    )

    # Combine into latent risk
    latent_risk = (
        wafer_process
        + density_component
        + radial_strength * radial_component
        + linear_strength * linear_component
    )

    # Add smooth random process variation
    random_field = rng.normal(
        0.0,
        0.15,
        size=(rows, cols)
    )

    random_field = uniform_filter(
        random_field,
        size=3,
        mode="nearest"
    )

    latent_risk += random_field

    # Normalize to approximately [-1, 1]
    valid_mask = wafer_map > 0

    valid_values = latent_risk[
        valid_mask
    ]

    if len(valid_values) > 1:

        mean = valid_values.mean()
        std = valid_values.std()

        if std > 1e-8:

            latent_risk = (
                latent_risk - mean
            ) / std

    # Clip extreme process conditions
    latent_risk = np.clip(
        latent_risk,
        -3.0,
        3.0
    )

    # Invalid/non-die locations carry zero risk
    latent_risk[
        ~valid_mask
    ] = 0.0

    return latent_risk


def generate_die_features(
    wafer_map,
    config,
    rng,
    latent_risk
):
    """
    Generate synthetic die-level parametric features.

    IMPORTANT:
        No final target label is used to generate measurements.

    Features depend only on:
        - base manufacturing variation
        - wafer geometry
        - original WM-811K failure neighborhood
        - latent pre-target process risk

    The latent process signal is deliberately strong enough to make
    Model A predictive, while remaining stochastic and non-deterministic.
    """

    feature_defs = config["features"]

    n_features = len(feature_defs)

    rows, cols = wafer_map.shape

    window_size = config["neighborhood_window"]

    radial_strength = config["radial_gradient_strength"]

    linear_strength = config["linear_gradient_strength"]

    neigh_influence = config["neighborhood_influence"]

    radial_map = compute_radial_map(
        wafer_map
    )

    linear_map = compute_linear_gradient(
        wafer_map,
        rng
    )

    old_fail_density = (
        compute_neighborhood_fail_density(
            wafer_map,
            window_size
        )
    )

    old_label_map = (
        wafer_map == 2
    ).astype(int)

    features_grid = np.zeros(
        (
            rows,
            cols,
            n_features
        ),
        dtype=np.float64
    )

    valid_mask = wafer_map > 0

    # --------------------------------------------------------------
    # Shared latent process signal
    # --------------------------------------------------------------

    # Compress latent risk to a stable range while retaining
    # meaningful differences between low- and high-risk dies.
    risk_scale = np.tanh(
        latent_risk / 1.25
    )

    # Small spatial component from the original wafer geometry.
    spatial_risk = (
        0.35 * radial_map
        + 0.20 * linear_map
        + 0.45 * old_fail_density
    )

    spatial_risk = np.clip(
        spatial_risk,
        -1.0,
        1.0
    )

    # Combined pre-target process state.
    process_state = (
        0.80 * risk_scale
        + 0.20 * spatial_risk
    )

    process_state = np.clip(
        process_state,
        -1.5,
        1.5
    )

    # --------------------------------------------------------------
    # Generate each parametric measurement
    # --------------------------------------------------------------

    for f_idx, fdef in enumerate(
        feature_defs
    ):

        base_mean = fdef[
            "base_mean"
        ]

        base_std = fdef[
            "base_std"
        ]

        fail_shift = fdef[
            "fail_shift"
        ]

        # Base measurement noise.
        base = rng.normal(
            base_mean,
            base_std,
            size=(rows, cols)
        )

        # ----------------------------------------------------------
        # Wafer-level spatial process variation
        # ----------------------------------------------------------

        radial_coeff = (
            rng.uniform(-1, 1)
            * radial_strength
            * base_std
        )

        linear_coeff = (
            rng.uniform(-1, 1)
            * linear_strength
            * base_std
        )

        base += (
            radial_map
            * radial_coeff
        )

        base += (
            linear_map
            * linear_coeff
        )

        # ----------------------------------------------------------
        # Original WM-811K failure neighborhood
        # ----------------------------------------------------------

        base += (
            old_fail_density
            * neigh_influence
            * fail_shift
        )

        # ----------------------------------------------------------
        # LATENT PROCESS SIGNAL
        # ----------------------------------------------------------

        # The same underlying manufacturing condition that later
        # influences the failure probability also affects the
        # measurements.
        #
        # Crucially, this uses latent_risk rather than final labels.

        latent_effect = (
            process_state
            * fail_shift
            * 1.8
        )

        base += latent_effect

        # ----------------------------------------------------------
        # Small independent measurement noise
        # ----------------------------------------------------------

        process_noise = rng.normal(
            0.0,
            base_std * 0.02,
            size=(rows, cols)
        )

        base += process_noise

        # Keep invalid wafer positions neutral.
        base[
            ~valid_mask
        ] = base_mean

        features_grid[
            :,
            :,
            f_idx
        ] = base

    return (
        features_grid,
        old_label_map
    )

def generate_new_failure_labels(
    wafer_map,
    latent_risk,
    config,
    rng
):
    """
    Generate NEW failures after measurements exist.

    IMPORTANT:
        Final labels are generated only after die-level and
        block-level measurements have been created.

    The failure probability depends on the same PRE-TARGET
    latent process risk that generated the measurements.

    No final target is used to generate any measurement.
    """

    rows, cols = wafer_map.shape

    old_label_map = (
        wafer_map == 2
    ).astype(int)

    passing_mask = (
        wafer_map == 1
    )

    new_fail_rate = config.get(
        "new_fail_rate",
        0.02
    )

    # --------------------------------------------------------------
    # PRE-TARGET RISK
    # --------------------------------------------------------------

    risk = np.asarray(
        latent_risk,
        dtype=float
    )

    # Standardize risk over valid dies.
    valid_risk = risk[passing_mask]

    if len(valid_risk) > 1:
        risk_mean = valid_risk.mean()
        risk_std = valid_risk.std()

        if risk_std > 1e-8:
            risk_z = (
                risk - risk_mean
            ) / risk_std
        else:
            risk_z = risk - risk_mean
    else:
        risk_z = risk

    # --------------------------------------------------------------
    # Convert latent risk into failure probability.
    #
    # Higher latent risk -> substantially higher failure probability.
    # Still stochastic: the target is NOT deterministic.
    # --------------------------------------------------------------

    risk_score = np.clip(
        risk_z,
        -3.0,
        3.0
    )

    # Logistic mapping.
    risk_probability = (
        1.0 /
        (
            1.0 +
            np.exp(
                -1.35 * risk_score
            )
        )
    )

    # Normalize so the average probability is approximately
    # the configured new-failure rate.
    mean_risk_probability = (
        risk_probability[passing_mask].mean()
        if np.any(passing_mask)
        else 0.5
    )

    if mean_risk_probability > 1e-8:
        risk_multiplier = (
            risk_probability /
            mean_risk_probability
        )
    else:
        risk_multiplier = np.ones_like(
            risk_probability
        )

    # --------------------------------------------------------------
    # Spatial process contribution.
    # --------------------------------------------------------------

    old_fail_density = (
        compute_neighborhood_fail_density(
            wafer_map,
            config["neighborhood_window"]
        )
    )

    radial_map = compute_radial_map(
        wafer_map
    )

    spatial_multiplier = (
        1.0
        + 0.35 * old_fail_density
        + 0.20 * radial_map
    )

    # --------------------------------------------------------------
    # Final pre-target probability.
    #
    # The expected overall rate remains close to new_fail_rate,
    # while high-risk dies receive substantially larger probability.
    # --------------------------------------------------------------

    new_fail_prob = (
        new_fail_rate
        * risk_multiplier
        * spatial_multiplier
    )

    # Keep probabilities valid and prevent pathological extremes.
    new_fail_prob = np.clip(
        new_fail_prob,
        0.0,
        0.35
    )

    new_fail_prob[
        ~passing_mask
    ] = 0.0

    # --------------------------------------------------------------
    # FINAL STOCHASTIC TARGET GENERATION
    # --------------------------------------------------------------

    random_draw = rng.random(
        (rows, cols)
    )

    new_fail_mask = (
        random_draw <
        new_fail_prob
    ) & passing_mask

    # Final target:
    # original failures remain failures.
    label_map = (
        old_label_map |
        new_fail_mask.astype(int)
    )

    return (
        old_label_map,
        new_fail_mask.astype(int),
        label_map.astype(int)
    )


def wafer_to_dataframe(
    wafer_id,
    wafer_map,
    features_grid,
    old_label_map,
    label_map,
    feature_names
):
    """
    Convert wafer data into dataframe rows.
    """

    valid_mask = (
        wafer_map > 0
    )

    die_rows, die_cols = np.where(
        valid_mask
    )

    n_dies = len(
        die_rows
    )

    records = {
        "wafer_id": [
            wafer_id
        ] * n_dies,

        "die_row": die_rows,

        "die_col": die_cols,
    }

    # Die-level features
    for f_idx, fname in enumerate(
        feature_names
    ):

        records[fname] = (
            features_grid[
                die_rows,
                die_cols,
                f_idx
            ]
        )

    # Original/pre-test label
    records["old_label"] = (
        old_label_map[
            die_rows,
            die_cols
        ]
    )

    # Final/post-test label
    records["label"] = (
        label_map[
            die_rows,
            die_cols
        ]
    )

    return pd.DataFrame(
        records
    )


def generate_block_readings(
    n_dies,
    latent_risk_array,
    config,
    rng
):
    """
    Generate block-level readings.

    Block measurements depend on the PRE-TARGET latent process
    condition rather than the final target label.

    This preserves the intended idea that high-resolution block
    measurements contain additional information about an underlying
    manufacturing condition.

    The final failure label is NOT used here.
    """

    br_config = config.get(
        "block_readings",
        {}
    )

    k = config.get(
        "num_block_readings",
        2000
    )

    base_mean = br_config.get(
        "base_mean",
        100.0
    )

    base_std = br_config.get(
        "base_std",
        15.0
    )

    fail_shift_frac = br_config.get(
        "fail_shift_fraction",
        0.3
    )

    anomalous_frac = br_config.get(
        "anomalous_block_fraction",
        0.05
    )

    corr_kernel = br_config.get(
        "block_correlation_kernel",
        5
    )

    fail_shift = (
        fail_shift_frac *
        base_std
    )

    readings_list = []

    for i in range(n_dies):

        risk = float(
            latent_risk_array[i]
        )

        # Base block measurements
        readings = rng.normal(
            base_mean,
            base_std,
            size=k
        )

        # Smooth neighboring block readings
        if corr_kernel > 1:

            from scipy.ndimage import (
                uniform_filter1d
            )

            smooth = (
                uniform_filter1d(
                    readings,
                    size=corr_kernel,
                    mode="nearest"
                )
            )

            readings = (
                0.6 * readings
                + 0.4 * smooth
            )

        # Smooth transformation of latent process risk
        risk_signal = np.tanh(
            risk / 1.5
        )

        # Small global process shift shared across blocks
        readings += (
            risk_signal
            * fail_shift
            * 0.75
        )

        # Higher-risk process conditions have a higher probability
        # of localized block anomalies, but this remains stochastic.
        risk_probability = (
            1.0 /
            (
                1.0 +
                np.exp(
                    -risk
                )
            )
        )

        # Keep anomaly probability bounded
        anomaly_probability = (
            anomalous_frac
            * (
                0.25
                + 1.5 *
                risk_probability
            )
        )

        anomaly_probability = np.clip(
            anomaly_probability,
            0.0,
            min(
                0.25,
                anomalous_frac * 3.0
            )
        )

        expected_anomalous = (
            k *
            anomaly_probability
        )

        n_anomalous = int(
            rng.poisson(
                expected_anomalous
            )
        )

        n_anomalous = min(
            n_anomalous,
            max(
                1,
                int(k * 0.20)
            )
        )

        if n_anomalous > 0:

            # Select a cluster center
            seed_pos = rng.integers(
                0,
                k
            )

            anomalous_positions = set()

            # Cluster around the seed
            max_attempts = (
                n_anomalous * 20
            )

            attempts = 0

            while (
                len(anomalous_positions)
                < n_anomalous
                and attempts < max_attempts
            ):

                offset = int(
                    rng.normal(
                        0,
                        max(
                            1,
                            k * 0.05
                        )
                    )
                )

                pos = (
                    seed_pos +
                    offset
                ) % k

                anomalous_positions.add(
                    pos
                )

                attempts += 1

            # Risk-dependent anomaly amplitude
            amplitude = (
                fail_shift
                * (
                    0.7
                    + 1.2 *
                    max(
                        risk_signal,
                        0.0
                    )
                )
            )

            for pos in anomalous_positions:

                readings[pos] += (
                    rng.normal(
                        amplitude,
                        max(
                            amplitude * 0.3,
                            1e-6
                        )
                    )
                )

        # Convert to compact string
        readings_str = " ".join(
            f"{v:.2f}"
            for v in readings
        )

        readings_list.append(
            readings_str
        )

    return readings_list


def generate_dataset(
    wafer_maps,
    wafer_ids,
    config,
    rng
):
    """
    Generate the complete dataset.

    Causal order:

        WM-811K geometry + old_label
                    ↓
          latent process risk
                ↙       ↘
        die features   block readings
                ↘       ↙
          final new-failure sampling
                    ↓
                final label
    """

    feature_names = [
        f["name"]
        for f in config["features"]
    ]

    all_dfs = []

    for i, (wmap, wid) in enumerate(
        zip(
            wafer_maps,
            wafer_ids
        )
    ):

        if (i + 1) % 100 == 0:

            print(
                f"  Processing wafer "
                f"{i + 1}/{len(wafer_maps)}..."
            )

        # --------------------------------------------------
        # STEP 1: Generate pre-target latent process risk
        # --------------------------------------------------

        latent_risk = (
            generate_latent_process_risk(
                wmap,
                config,
                rng
            )
        )

        # --------------------------------------------------
        # STEP 2: Generate die-level measurements
        # --------------------------------------------------

        features_grid, old_label_map = (
            generate_die_features(
                wmap,
                config,
                rng,
                latent_risk
            )
        )

        # --------------------------------------------------
        # STEP 3: Generate block-level measurements
        # --------------------------------------------------

        valid_mask = (
            wmap > 0
        )

        die_rows, die_cols = np.where(
            valid_mask
        )

        latent_risk_array = (
            latent_risk[
                die_rows,
                die_cols
            ]
        )

        block_readings = (
            generate_block_readings(
                len(die_rows),
                latent_risk_array,
                config,
                rng
            )
        )

        # --------------------------------------------------
        # STEP 4: ONLY NOW generate final target labels
        # --------------------------------------------------

        (
            old_label_map,
            new_fail_mask,
            label_map
        ) = generate_new_failure_labels(
            wmap,
            latent_risk,
            config,
            rng
        )

        # --------------------------------------------------
        # STEP 5: Build dataframe
        # --------------------------------------------------

        df = wafer_to_dataframe(
            wid,
            wmap,
            features_grid,
            old_label_map,
            label_map,
            feature_names
        )

        df["block_readings"] = (
            block_readings
        )

        all_dfs.append(
            df
        )

    return pd.concat(
        all_dfs,
        ignore_index=True
    )


def print_summary(
    df,
    split_name
):
    """Print dataset summary statistics."""

    n_wafers = (
        df["wafer_id"].nunique()
    )

    n_dies = len(df)

    n_fail = int(
        df["label"].sum()
    )

    fail_rate = (
        n_fail /
        n_dies *
        100
    )

    print(
        f"\n{'=' * 50}"
    )

    print(
        f"  {split_name} Dataset Summary"
    )

    print(
        f"{'=' * 50}"
    )

    print(
        f"  Wafers:     {n_wafers}"
    )

    print(
        f"  Total dies: {n_dies:,}"
    )

    print(
        f"  Failed:     {n_fail:,} "
        f"({fail_rate:.2f}%)"
    )

    print(
        f"  Passed:     "
        f"{n_dies - n_fail:,} "
        f"({100 - fail_rate:.2f}%)"
    )

    # Feature overlap
    feature_cols = [
        c for c in df.columns
        if not c.startswith("zone_")
        and c not in (
            "wafer_id",
            "die_row",
            "die_col",
            "label",
            "old_label",
            "block_readings",
            "neighborhood_fail_density"
        )
    ]

    if len(feature_cols) > 0:

        print(
            "\n  Feature Separability "
            "(Cohen's d):"
        )

        pass_df = df[
            df["label"] == 0
        ]

        fail_df = df[
            df["label"] == 1
        ]

        for col in feature_cols[:5]:

            p_mean = (
                pass_df[col].mean()
            )

            p_std = (
                pass_df[col].std()
            )

            f_mean = (
                fail_df[col].mean()
            )

            f_std = (
                fail_df[col].std()
            )

            pooled_std = np.sqrt(
                (
                    p_std ** 2
                    + f_std ** 2
                ) / 2
            )

            d = (
                abs(
                    f_mean -
                    p_mean
                ) /
                pooled_std
                if pooled_std > 0
                else 0
            )

            print(
                f"    {col:20s}: "
                f"d = {d:.4f} "
                f"(low = high overlap)"
            )

        print(
            f"    ... "
            f"({len(feature_cols)} "
            f"features total)"
        )

    # Important task-specific statistic
    eligible = (
        df["old_label"] == 0
    )

    eligible_new_fails = (
        (df["old_label"] == 0)
        &
        (df["label"] == 1)
    )

    if eligible.sum() > 0:

        eligible_rate = (
            eligible_new_fails.sum()
            /
            eligible.sum()
            * 100
        )

        print(
            f"\n  New-failure rate among "
            f"originally passing dies: "
            f"{eligible_rate:.2f}%"
        )

    print(
        f"{'=' * 50}\n"
    )


def save_outputs(
    train_df,
    test_df,
    wafer_maps_dict,
    config,
    export_csv=False
):
    """
    Save datasets.

    Validation set is the test set with the final target label
    removed.
    """

    output_dir = Path(
        config["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    formats = config.get(
        "output_formats",
        [
            "parquet",
            "pkl"
        ]
    )

    # Validation version of test
    val_df = test_df.drop(
        columns=["label"]
    )

    if "parquet" in formats:

        train_df.to_parquet(
            output_dir /
            "train.parquet",
            index=False
        )

        test_df.to_parquet(
            output_dir /
            "test.parquet",
            index=False
        )

        val_df.to_parquet(
            output_dir /
            "validation.parquet",
            index=False
        )

        print(
            f"  Saved: "
            f"{output_dir}/train.parquet, "
            f"test.parquet, "
            f"validation.parquet"
        )

    if "pkl" in formats:

        train_df.to_pickle(
            output_dir /
            "train.pkl"
        )

        test_df.to_pickle(
            output_dir /
            "test.pkl"
        )

        val_df.to_pickle(
            output_dir /
            "validation.pkl"
        )

        with open(
            output_dir /
            "wafer_maps.pkl",
            "wb"
        ) as f:

            pickle.dump(
                wafer_maps_dict,
                f
            )

        print(
            f"  Saved: "
            f"{output_dir}/train.pkl, "
            f"test.pkl, "
            f"validation.pkl, "
            f"wafer_maps.pkl"
        )

    if (
        export_csv
        or
        "csv" in formats
    ):

        train_df.to_csv(
            output_dir /
            "train.csv",
            index=False
        )

        test_df.to_csv(
            output_dir /
            "test.csv",
            index=False
        )

        val_df.to_csv(
            output_dir /
            "validation.csv",
            index=False
        )

        print(
            f"  Saved: "
            f"{output_dir}/train.csv, "
            f"test.csv, "
            f"validation.csv"
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic die-level "
            "features for hackathon"
        )
    )

    parser.add_argument(
        "--num_wafers",
        type=int,
        default=None,
        help=(
            "Total number of wafers "
            "(overrides config train+test)"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file"
    )

    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export CSV format"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(
        args.config
    )

    # Determine wafer counts
    if args.num_wafers:

        n_train = int(
            args.num_wafers * 0.8
        )

        n_test = (
            args.num_wafers -
            n_train
        )

    else:

        n_train = config[
            "num_wafers_train"
        ]

        n_test = config[
            "num_wafers_test"
        ]

    total_needed = (
        n_train +
        n_test
    )

    # Initialize deterministic RNG
    rng = np.random.default_rng(
        config["seed"]
    )

    # Load WM-811K
    labeled_df, none_df = (
        load_wm811k(
            config["wm811k_path"]
        )
    )

    # Select wafers
    print(
        f"\nSelecting "
        f"{n_train} train + "
        f"{n_test} test wafers..."
    )

    all_maps, all_ids = (
        select_wafers(
            labeled_df,
            none_df,
            total_needed,
            config["target_fail_rate"],
            rng
        )
    )

    train_maps = all_maps[
        :n_train
    ]

    train_ids = all_ids[
        :n_train
    ]

    test_maps = all_maps[
        n_train:
    ]

    test_ids = all_ids[
        n_train:
    ]

    # Store wafer maps
    wafer_maps_dict = {
        str(wid): wmap
        for wid, wmap
        in zip(
            all_ids,
            all_maps
        )
    }

    # Generate train
    print(
        f"\nGenerating train features "
        f"({n_train} wafers)..."
    )

    train_df = generate_dataset(
        train_maps,
        train_ids,
        config,
        rng
    )

    # Generate test
    print(
        f"\nGenerating test features "
        f"({n_test} wafers)..."
    )

    test_df = generate_dataset(
        test_maps,
        test_ids,
        config,
        rng
    )

    # Summaries
    print_summary(
        train_df,
        "TRAIN"
    )

    print_summary(
        test_df,
        "TEST"
    )

    # Save
    print(
        "Saving outputs..."
    )

    save_outputs(
        train_df,
        test_df,
        wafer_maps_dict,
        config,
        export_csv=args.csv
    )

    print(
        "\nDone! Dataset generation complete."
    )


if __name__ == "__main__":
    main()

"""
Run Complete Pipeline

Author : Raphaël VASSART
===============================

This script is the single entry point to:
    1) load a pre-built OPI dataset (.pkl from dataset_builder.py),
    2) print a reward-composition diagnostic (safety check before training),
    3) fit the OfflinePolicyIteration model,
    4) run the full validation suite (Bellman, actions, policy improvement, stability, grouped K-fold CV),
    5) persist EVERYTHING needed to redraw plots later without having to retrain anything.

Output layout
---------------
    results/<tag>__<timestamp>/
        ├── config.json              # the exact OPIConfig used
        ├── run_info.json            # timestamp, paths, dataset summary
        ├── reward_diagnostic.txt    # per-feature / per-action imbalance
        ├── model.npz                # S, cost_shift, training history
        ├── evaluation_report.json   # all metrics, machine-readable
        ├── evaluation_report.txt    # same content, human-readable
        └── arrays/
              ├── bellman_residuals.npy
              ├── sensitivities.npy
              ├── q_policy.npy
              └── q_dataset.npy

Example
--------
    python run.py \
        --pickle data/all_to_all_opi_dataset.pkl \
        --csv    data/all_to_all_transitions.csv \
        --score-csv data/ExpS1_Full_scores.csv \
        --tag    baseline_seed0 \
        --lambda-reg 1e-3 --gamma 0.9 --n-iter 100 \
        --cv-folds 5
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pickle
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from elsa_configOPI import OPIConfig
from OPI_core import Dataset, OfflinePolicyIteration
from OPI_validation import run_full_evaluation, print_report, save_report
from reward import RewardComputer

logger = logging.getLogger("run_training")


# ============================================================================
# |                                                                          |
# |  1) Reward-composition diagnostic                                        |
# |                                                                          |
# ============================================================================

_FEATURE_HUMAN_NAMES = {
    "f1_drop_deg_signed": "f1 – plantarflexion drop (deg)",
    "f1_hs_to_mpa_time": "f1B – HS→MPA time (s)",
    "f1_max_abs_speed_hs_to_mpa": "f1C – max PF speed (deg/s)",
    "f2_mpa_to_pushoff_time": "f2 – MPA→push-off time (s)",
    "f3_max_abs_speed_pushoff": "f3 – push-off max speed (deg/s)",
    "f4_pushoff_duration": "f4 – push-off duration (s)",
    "f5_peak_torque_pushoff_norm": "f5 – norm. peak PF torque (Nm/kg)",
    "f6_generated_mechanical_energy_norm": "f6 – norm. push-off energy (J/kg)",
}
_ACTION_NAMES = ["Δd_dors [Nm·s/rad]", "Δθ_trig [rad]", "Δτ_lim [Nm]"]

# ============================================================================
# |                                                                          |
# |  1b) Reward recomputation from source CSV (clean, ablation-safe)         |
# |                                                                          |
# ============================================================================
#
# WHY THIS EXISTS
# ---------------
# dataset_builder.py freezes the shaped reward r_k into the transitions CSV at build time (line 1098 "reward": float(reward)). 
# When an ablation changes ANY of the four reward-shaping hyperparameters (cfg.R_a, cfg.feature_weights, cfg.lambda_, cfg.alpha) the frozen rewards are stale and must be recomputed, 
# otherwise OPI.fit() learns from the BASELINE reward regardless of cfg, silently invalidating the ablation (first diagnosed on run 795493: R_a ×5 produced bit-identical trajectories to the baseline 795205).
# ============================================================================

def _resolve_reward_columns(csv_df: pd.DataFrame, cfg: OPIConfig):
    """
    Locate the correct feature / action / human-score column layout inside the transitions CSV, honouring the builder's reward_from setting ('next' → x_k1 / condition_id_k1; 'current' → x_k / condition_id_k). 
    Centralised so that _recompute_rewards and any future sibling helper stay in sync.
    """
    if "reward_from" in csv_df.columns:
        labels = csv_df["reward_from"].astype(str).unique()
        if len(labels) > 1:
            raise ValueError(
                f"Mixed reward_from labels in CSV: {labels}. "
                "Recomputation assumes a homogeneous dataset."
            )
        reward_block = str(labels[0])
    else:
        reward_block = "next" # dataset_builder's default
    prefix = "x_k1__" if reward_block == "next" else "x_k__"

    feat_cols = [c for c in csv_df.columns if c.startswith(prefix)]
    if len(feat_cols) < cfg.n_features:
        raise ValueError(
            f"CSV has only {len(feat_cols)} '{prefix}*' feature columns "
            f"but cfg.n_features={cfg.n_features}."
        )
    feat_cols = feat_cols[: cfg.n_features]

    act_cols = ["a_k__d_plant", "a_k__theta_trig", "a_k__tau_lim"]
    if not all(c in csv_df.columns for c in act_cols):
        # Permissive fallback: accept any a_k__* (handles legacy builds that wrote 'a_k__theta' instead of 'a_k__theta_trig').
        act_cols = [c for c in csv_df.columns if c.startswith("a_k__")][: cfg.n_actions]
    if len(act_cols) != cfg.n_actions:
        raise ValueError(
            f"Expected {cfg.n_actions} 'a_k__*' columns, found {len(act_cols)}."
        )

    return feat_cols, act_cols, reward_block


def _recompute_rewards(dataset: Dataset, csv_df: Optional[pd.DataFrame], cfg: OPIConfig, *, score_csv_path: Optional[Path] = None, logger: Optional[logging.Logger] = None, consistency_tol: float = 1e-5):
    """
    Recompute the rewards of all transitions in the dataset according to the current cfg, using the source CSV as ground truth for the human-feedback terms.
    Used during the ablation study to ensure that any change in reward-shaping hyperparameters is correctly reflected in the rewards used for training, without having to rebuild the dataset from scratch.
    Returns
    -------
    int
        Number of transitions whose 'r' field actually changed
        (|Δr| > 1e-12). Zero on the fast path.
    """
    # -- 1) Fast-path -------------------------------------------------
    # Build-time defaults == defaults of a freshly-constructed OPIConfig.
    # dataset_builder does not override R_a or feature_weights, and its CLI defaults are lambda_human=0.5, alpha=0.5 — matching OPIConfig's own defaults.
    ref_cfg = OPIConfig(n_features=cfg.n_features, n_actions=cfg.n_actions)
    ra_same = np.allclose(ref_cfg.R_a, cfg.R_a)
    w_same = np.allclose(ref_cfg.feature_weights, cfg.feature_weights)
    lam_same = np.isclose(ref_cfg.lambda_, cfg.lambda_)
    alpha_same = np.isclose(ref_cfg.alpha, cfg.alpha)
    coding_same = (ref_cfg.assist_coding == cfg.assist_coding) and np.isclose(ref_cfg.assist_optimum, cfg.assist_optimum)
    cost_same = ref_cfg.action_cost == cfg.action_cost

    if ra_same and w_same and lam_same and alpha_same and coding_same and cost_same:
        return 0

    # -- 2) Inputs are mandatory past this point --------------------------
    if csv_df is None:
        raise ValueError(
            "Reward recomputation required but no transitions CSV was "
            "provided (pass --csv to run.py)."
        )
    if score_csv_path is None:
        diffs = [n for n, same in zip(
            ("R_a", "feature_weights", "lambda_", "alpha", "assist_coding", "action_cost"),
            (ra_same, w_same, lam_same, alpha_same, coding_same, cost_same)) if not same]
        raise ValueError(
            f"Reward recomputation required (cfg differs on: {diffs}) but "
            "no source scores CSV was provided. Pass --score-csv to run.py "
            "pointing to the subject's scores CSV (e.g. ExpS1_scores.csv) — "
            "the same file consumed by dataset_builder at build time. "
            "Running without this would silently invalidate the ablation."
        )

    # -- 3) Resolve CSV layout --------------------------------------------
    feat_cols, act_cols, reward_block = _resolve_reward_columns(csv_df, cfg)
    for col in ("comfort_used", "assistance_used"):
        if col not in csv_df.columns:
            raise ValueError(f"Transitions CSV is missing column '{col}'.")

    N = len(dataset)
    if len(csv_df) != N:
        raise ValueError(
            f"CSV has {len(csv_df)} rows but dataset has {N} transitions."
        )

    # -- 4) Load human-reward stats from the SOURCE (ground truth) -------
    # Both use the per-subject z-score over all blocks of the scores CSV, with their own assistance coding.
    rc_build = RewardComputer.fit_from_scores_csv(ref_cfg, score_csv_path)
    rc_new = RewardComputer.fit_from_scores_csv(cfg, score_csv_path)

    # -- 5) Extract per-transition arrays --------------------------------
    X = csv_df[feat_cols].to_numpy(dtype=float) # standardised x̃
    A_mat = csv_df[act_cols].to_numpy(dtype=float)
    comfort = csv_df["comfort_used"].to_numpy(dtype=float)
    assist  = csv_df["assistance_used"].to_numpy(dtype=float)

    # -- 6) Consistency check - catch score-CSV drift or wrong file ------
    # Rebuild the reward with the BUILD-time defaults and compare to the frozen stored rewards. 
    # If the source CSV is the right one and has not been edited since build time, this matches to ~1e-15.
    r_build = rc_build.compute_batch(X, A_mat, comfort, assist)
    r_stored = np.array([t.r for t in dataset._data], dtype=float)
    max_drift = float(np.max(np.abs(r_build - r_stored)))
    if max_drift > consistency_tol:
        raise ValueError(
            f"Source-CSV consistency check FAILED (max |Δ| = {max_drift:.3e} "
            f"> tol {consistency_tol:.0e}). The score CSV at "
            f"{score_csv_path!r} does not reproduce the rewards stored in the "
            "dataset when evaluated with build-time defaults. Either the "
            "scores CSV has been modified since the dataset was built, or "
            "a wrong --score-csv path was passed. Refusing to overwrite "
            "rewards with an inconsistent source."
        )
    if logger is not None:
        logger.info(
            "Source-CSV consistency check PASSED (max |r_build - r_stored| "
            "= %.3e < tol %.0e).", max_drift, consistency_tol
        )

    # -- 7) Compute r under the CURRENT cfg and overwrite ----------------
    r_new = rc_new.compute_batch(X, A_mat, comfort, assist)

    dr = r_new - r_stored
    for i, t in enumerate(dataset._data):
        t.r = float(r_new[i])
    n_corrected = int(np.sum(np.abs(dr) > 1e-12))

    if logger is not None:
        logger.info(
            "Rewards recomputed from source (R_a_same=%s, w_same=%s, "
            "lam_same=%s, alpha_same=%s, coding=%s, action_cost=%s, score_csv=%s): n=%d, "
            "mean|Δr|=%.3e, max|Δr|=%.3e, ⟨r_old⟩=%.3f, ⟨r_new⟩=%.3f.",
            ra_same, w_same, lam_same, alpha_same, cfg.assist_coding, cfg.action_cost, str(score_csv_path),
            n_corrected,
            float(np.mean(np.abs(dr))), float(np.max(np.abs(dr))),
            float(np.mean(r_stored)), float(np.mean(r_new)),
        )
    return n_corrected


# Backward-compatibility alias for any external tool that imported the historical symbol.
_recompute_prosthesis_rewards = _recompute_rewards


def reward_diagnostic(dataset: Dataset, cfg: OPIConfig, csv_df: Optional[pd.DataFrame] = None):
    """
    Decompose the mean reward into its feature-term and action-term contributions, and return a human-readable string plus a dict of numbers.

    This helps catch situations where one feature or one action dimension dominates the reward because of a scale mismatch.
    """
    x_feat, a, x1_feat, r, p, p1 = dataset.arrays(cfg)

    # Retrieve per-feature σ: either from the CSV (column sigma__<feature>) or fallback
    sigmas = None
    feat_cols = None
    if csv_df is not None:
        feat_cols = [c.replace("raw_k__", "")
                     for c in csv_df.columns if c.startswith("raw_k__")]
        if len(feat_cols) == cfg.n_features:
            sigmas = np.array([csv_df[f"sigma__{c}"].iloc[0] for c in feat_cols])
    if sigmas is None:
        sigmas = np.full(cfg.n_features, np.nan)
        feat_cols = [f"feat_{i}" for i in range(cfg.n_features)]
        logger.warning("No CSV provided to reward_diagnostic: σ_ref column "
                       "will be reported as NaN.")

    # Feature-term contribution: mean of wᵢ · x̃ᵢ²
    weights = np.asarray(cfg.feature_weights, dtype=float)
    x_tilde = x_feat # already standardised by the dataset builder
    per_feat = weights * np.mean(x_tilde ** 2, axis=0)
    feat_total = float(per_feat.sum())

    # Action-term contribution: mean of aᵀ R_a a, decomposed by dimension
    Ra = np.asarray(cfg.R_a, dtype=float)
    off_diag_mass = float(np.sum(np.abs(Ra - np.diag(np.diag(Ra)))))
    diag_mass = float(np.sum(np.abs(np.diag(Ra))))
    if off_diag_mass > 1e-8 * max(diag_mass, 1e-12):
        logger.warning("R_a has non-negligible off-diagonal entries; the "
                       "per-action-dimension percentages may not sum to 100 %%.")
    per_act = np.array([Ra[i, i] * np.mean(a[:, i] ** 2) for i in range(cfg.n_actions)])
    act_total = float(np.mean(np.einsum("ni,ij,nj->n", a, Ra, a)))

    # ------------- Build human-readable text (generated by Claude AI) -------------
    lines = []
    sep = "═" * 78
    sub = "─" * 78
    lines.append(sep)
    lines.append(" REWARD COMPOSITION DIAGNOSTIC")
    lines.append(sep)
    lines.append(f" Observed mean reward over the dataset : {r.mean():.3f}")
    lines.append(f"   ├── feature term   <Σᵢ wᵢ x̃ᵢ²> ≈ {feat_total:.3f}   (R̂_p_feat = -{feat_total:.2f})")
    lines.append(f"   └── action  term   <aᵀ R_a a>    ≈ {act_total:.3f}   (R̂_p_act  = -{act_total:.2f})")
    lines.append(f"   Ratio action/feature : {act_total/max(feat_total, 1e-12):.2f}×")
    lines.append("")

    lines.append(sub)
    lines.append(" Per-feature contribution to the feature term")
    lines.append(sub)
    lines.append(f"   {'feature':<38}  {'σ_ref':>10}  {'<|x̃|>':>10}  {'<x̃²>':>10}  {'% feat':>8}")
    for i, col in enumerate(feat_cols):
        human = _FEATURE_HUMAN_NAMES.get(col, col)
        xt = x_tilde[:, i]
        pct = 100.0 * per_feat[i] / max(feat_total, 1e-12)
        lines.append(f"   {human:<38}  {sigmas[i]:>10.4f}  {np.mean(np.abs(xt)):>10.3f}  "
                     f"{np.mean(xt**2):>10.3f}  {pct:>7.1f}%")

    lines.append("")
    lines.append(sub)
    lines.append(" Per-action-dimension contribution to the action term")
    lines.append(sub)
    lines.append(f"   {'action':<28}  {'|a|_max':>10}  {'<a²>':>12}  {'% act':>8}")
    for i, name in enumerate(_ACTION_NAMES):
        ai_sq = np.mean(a[:, i] ** 2)
        pct = 100.0 * (Ra[i, i] * ai_sq) / max(act_total, 1e-12)
        lines.append(f"   {name:<28}  {np.max(np.abs(a[:, i])):>10.4f}  {ai_sq:>12.3f}  {pct:>7.1f}%")

    lines.append("")
    lines.append(" → Imbalance warning threshold: any feature or action >70 % of its term.")
    # Soft warnings
    for i, col in enumerate(feat_cols):
        pct = 100.0 * per_feat[i] / max(feat_total, 1e-12)
        if pct > 70.0:
            lines.append(f"   ⚠ Feature '{col}' explains {pct:.0f}% of the feature term.")
    for i, name in enumerate(_ACTION_NAMES):
        pct = 100.0 * (Ra[i, i] * np.mean(a[:, i] ** 2)) / max(act_total, 1e-12)
        if pct > 70.0:
            lines.append(f"   ⚠ Action dim '{name}' explains {pct:.0f}% of the action term.")
    lines.append(sep)

    text = "\n".join(lines)
    payload = {
        "reward_mean_observed": float(r.mean()),
        "feature_term_mean":    feat_total,
        "action_term_mean":     act_total,
        "sigmas_used":          sigmas.tolist(),
        "feature_cols":         list(feat_cols),
        "per_feature_contrib":  per_feat.tolist(),
        "per_action_contrib":   per_act.tolist(),
    }
    return text, payload


# ============================================================================
# |                                                                          |
# |  2) Pair-ID extraction for the grouped K-fold CV                         |
# |                                                                          |
# ============================================================================

def extract_pair_ids(dataset: Dataset, csv_df: Optional[pd.DataFrame]) -> List[str]:
    """
    Produce a list of pair_id strings — one per transition — used by the grouped K-fold CV to keep pseudo-transitions from the same real block-pair on the same side of the split.

    Priority order:
      1) If a CSV is provided AND its length matches the dataset, read the
         'pair_id' column directly.
      2) Otherwise, infer a surrogate pair_id from the (p, p1) signatures
         of every transition (same real pair <-> same (p, p1) pair).
    """
    if csv_df is not None and "pair_id" in csv_df.columns and len(csv_df) == len(dataset):
        logger.info("Using pair_ids from CSV column 'pair_id'.")
        return csv_df["pair_id"].astype(str).tolist()

    logger.info("No CSV provided - inferring pair_ids from (p, p1) signatures.")
    surrogate = []
    seen = {}
    for t in dataset:
        key = (tuple(np.round(t.p, 6)), tuple(np.round(t.p1, 6) if t.p1 is not None else t.p + t.a))
        if key not in seen:
            seen[key] = f"pair_{len(seen):03d}"
        surrogate.append(seen[key])
    logger.info("Inferred %d unique pair_ids.", len(set(surrogate)))
    return surrogate


def extract_pair_blocks(csv_df: pd.DataFrame) -> dict:
    """Map each pair_id to its two block indices (k, k+1), used to purge the CV folds of shared blocks."""
    first = csv_df.groupby("pair_id", sort=False)[["condition_id_k", "condition_id_k1"]].first()
    return {str(pid): (int(row.condition_id_k), int(row.condition_id_k1)) for pid, row in first.iterrows()}


def make_fold_reward_fn(csv_df: pd.DataFrame, cfg: OPIConfig, pair_ids: List[str]):
    """
    Return fold_rewards(train_idx), the rewards of all transitions with the z-score of the ratings computed on the training pairs of the fold only.
    Each pair counts once in these statistics (its 1600 pseudo-transitions share the same ratings).
    """
    feat_cols, act_cols, _ = _resolve_reward_columns(csv_df, cfg)
    X = csv_df[feat_cols].to_numpy(dtype=float)
    A_mat = csv_df[act_cols].to_numpy(dtype=float)
    comfort = csv_df["comfort_used"].to_numpy(dtype=float)
    assist = csv_df["assistance_used"].to_numpy(dtype=float)
    pair_ids = np.asarray(pair_ids)
    rc = RewardComputer(cfg)

    def fold_rewards(train_idx: np.ndarray) -> np.ndarray:
        _, first = np.unique(pair_ids[train_idx], return_index=True)
        one_row_per_pair = np.asarray(train_idx)[np.sort(first)]
        rc.fit_human_stats(comfort[one_row_per_pair], assist[one_row_per_pair])
        return rc.compute_batch(X, A_mat, comfort, assist)

    return fold_rewards


# ============================================================================
# |                                                                          |
# |  3) Config serialisation                                                 |
# |                                                                          |
# ============================================================================

def cfg_to_dict(cfg: OPIConfig) -> dict:
    """Convert OPIConfig to a JSON-serialisable dict (including nested dataclasses)."""
    out = {}
    for field in dataclasses.fields(cfg):
        val = getattr(cfg, field.name)
        if dataclasses.is_dataclass(val):
            out[field.name] = asdict(val)
        elif isinstance(val, np.ndarray):
            out[field.name] = val.tolist()
        else:
            out[field.name] = val
    # Derived properties that matter for reproducibility
    out["_derived"] = {
        "n_state_total": cfg.n_state_total,
        "n_joint":       cfg.n_joint,
    }
    return out


# ============================================================================
# |                                                                          |
# |  3b) Feature-weights presets                                             |
# |                                                                          |
# ============================================================================

def build_feature_weights(preset: str, dataset: Dataset, n_features: int, cfg_probe: OPIConfig):
    """
    Build the per-feature weight vector w used by the prosthesis reward

            r_p = - wᵀ (x̃²) - aᵀ R_a a

    from a named preset. Returning ``None`` lets ``OPIConfig.__post_init__``
    fall back to its built-in default ``np.ones(n_features)``.

    Presets
    -------
    'uniform'
        Returns ``None`` → equivalent to wᵢ = 1 for all features. This is
        my baseline choice and reflects the a priori INDIFFERENCE principle
        when no biomechanical prior is available: every standardised feature
        x̃ᵢ = (fᵢ - fᵢʳᵉᶠ)/σᵢʳᵉᶠ is already on a comparable scale, so giving
        them equal weight is the default.

    'inv_var'
        Weights are set proportionally to the INVERSE empirical variance of
        each standardised feature over the dataset:

                wᵢ ∝ 1 / ⟨x̃ᵢ²⟩_dataset ,        Σᵢ wᵢ = n_features.

        The proportionality constant is fixed so that Σᵢ wᵢ equals the
        number of features (same "total mass" as the uniform baseline),
        which keeps the magnitude of the feature term comparable across
        presets and makes the reward-composition diagnostic directly
        comparable to the baseline.

        Rationale — why this ablation matters for ELSA:
        In baseline run 795205, the per-feature contribution to the feature
        term was highly imbalanced: f6 (push-off energy) explained 38.9 %
        of the feature term on its own, while f3 only explained 10.0 %.
        Setting wᵢ ∝ 1/⟨x̃ᵢ²⟩ forces every feature to contribute ~equally on average, 
        which is the MAXIMUM-ENTROPY choice (Jaynes 1957) when there is no explicit
        reason to prefer one feature over another. Comparing this preset
        to the uniform baseline isolates the effect of the σʳᵉᶠ-induced
        imbalance on the learned policy.
    """
    if preset == "uniform":
        return None

    if preset == "inv_var":
        # x_feat in Dataset transitions is ALREADY the standardised x̃
        # (see run.reward_diagnostic comment at line 263). Reading it via
        # dataset.arrays() is therefore a direct read, not a recomputation.
        x_feat, *_ = dataset.arrays(cfg_probe)
        if x_feat.shape[1] != n_features:
            raise ValueError(
                f"Dataset has {x_feat.shape[1]} features but n_features="
                f"{n_features}; cannot build 'inv_var' weights."
            )
        mean_sq = np.mean(x_feat ** 2, axis=0)           # <x̃ᵢ²>, shape (F,)
        # Guard against a degenerate all-zero feature (should never happen
        # on the real dataset but protects the denominator numerically).
        eps = 1e-8
        w_raw = 1.0 / np.maximum(mean_sq, eps)
        # Normalise to the same total mass as the uniform baseline
        # (Σ w = n_features). This ensures that changing the preset does
        # NOT change the overall magnitude of the feature term — only its
        # distribution across features.
        w = w_raw * (float(n_features) / float(w_raw.sum()))
        return w

    raise ValueError(
        f"Unknown feature-weights preset: {preset!r}. "
        f"Supported presets: 'uniform', 'inv_var'."
    )



# ============================================================================
# |                                                                          |
# |  4) Main pipeline                                                        |
# |                                                                          |
# ============================================================================

def run(args: argparse.Namespace) -> Path:
    """Full training + evaluation pipeline; returns the output directory path."""
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s")

    # -- Output directory -------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{args.tag}__{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir.resolve())

    # -- Load dataset -----------------------------------------------------
    with open(args.pickle, "rb") as f:
        dataset: Dataset = pickle.load(f)
    logger.info("Loaded pickle dataset: N = %d transitions.", len(dataset))

    csv_df = pd.read_csv(args.csv) if args.csv else None
    if csv_df is not None:
        logger.info("Loaded CSV with %d rows, %d columns.", len(csv_df), len(csv_df.columns))

    # -- Build OPIConfig from CLI -----------------------------------------
    ra_scale = np.array([
        args.ra_scale_d_plant,
        args.ra_scale_theta_trig,
        args.ra_scale_tau_lim,
    ], dtype=float)
    r_a_per_dim_scale = None if np.allclose(ra_scale, 1.0) else ra_scale
    
    # Feature-weights preset: resolved BEFORE OPIConfig construction so that it can be injected as an explicit override. 
    # We use a lightweight probe config for the dataset.arrays() call inside the preset builder (it reads only x_feat and does not depend on feature_weights itself).
    cfg_probe = OPIConfig(n_features=args.n_features, n_actions=3)
    feature_weights = build_feature_weights(
        preset=args.feature_weights_preset,
        dataset=dataset,
        n_features=args.n_features,
        cfg_probe=cfg_probe,
    )
    if feature_weights is not None:
        logger.info(
            "Feature-weights preset '%s' → w = %s  (sum = %.3f).",
            args.feature_weights_preset,
            np.array2string(feature_weights, precision=4, separator=", "),
            float(np.sum(feature_weights)),
        )
    else:
        logger.info(
            "Feature-weights preset '%s' → falling back to uniform weights "
            "(wᵢ = 1 for all features).",
            args.feature_weights_preset,
        )

    
    cfg_kwargs = dict(
        n_features=args.n_features,
        n_actions=3,
        include_params_in_state=args.include_params_in_state,
        gamma=args.gamma,
        n_iterations=args.n_iter,
        tol=args.tol,
        lambda_=args.lambda_human,
        alpha=args.alpha,
        lambda_reg=args.lambda_reg,
        SDP_eps=args.sdp_eps,
        cost_shift_margin=args.cost_shift_margin,
        use_bias=not args.no_bias,
        R_a_per_dim_scale=r_a_per_dim_scale,
        feature_weights=feature_weights,
        assist_coding=args.assist_coding,
        action_cost=args.action_cost,
    )
    cfg = OPIConfig(**cfg_kwargs)
    # Verify runtime consistency with the dataset
    x_feat_probe, _, _, _, _, _ = dataset.arrays(cfg)
    if x_feat_probe.shape[1] != cfg.n_features:
        raise ValueError(
            f"Dataset has {x_feat_probe.shape[1]} features but cfg.n_features={cfg.n_features}. "
            "Either adjust --n-features or rebuild the dataset."
        )

    # -- Reward diagnostic ------------------------------------------------
    diag_text, diag_payload = reward_diagnostic(dataset, cfg, csv_df)
    print(diag_text)
    (out_dir / "reward_diagnostic.txt").write_text(diag_text, encoding="utf-8")
    
    # -- Reward recomputation from current cfg ----------------------------
    # The dataset pickle/CSV stores rewards that were FROZEN at dataset-build time. 
    # This means that any change to cfg.R_a or cfg.feature_weights in THIS training run only affects the reward-diagnostic display above but NOT the rewards used by OPI.fit().

    # Root cause: the issue was exposed by ablation run 795493 (R_a ×5 on τ_lim), whose fit trajectory was BIT-IDENTICAL to the baseline run 795205 despite the diagnostic showing a correctly scaled action term. 
    # The baseline used OPIConfig()'s default R_a = diag(1/half²) to build the CSV; subsequent cfg overrides are therefore ignored at training time unless we rewrite t.r explicitly here.

    # Fix: apply a per-transition correction
    #     r_new = r_old + (1 - λ) · (r_prosthesis_new - r_prosthesis_old)
    # where
    #     r_prosthesis(cfg) = - w(cfg)ᵀ · x̃²  -  aᵀ · R_a(cfg) · a
    # is computed from the CSV's standardised features x̃ (column x_{k,k1}__*) and the stored action a_k. 
    # The human-reward component is preserved intact because it does not depend on cfg.R_a or cfg.feature_weights.
    # This is algebraically EXACT, not an approximation, as long as the only cfg changes relative to build time are R_a and feature_weights.
    n_corrected = _recompute_rewards(
        dataset, csv_df, cfg,
        score_csv_path=Path(args.score_csv) if args.score_csv else None,
        logger=logger,
    )
    if n_corrected > 0:
        logger.info(
            "Rewards fully recomputed for %d / %d transitions "
            "(current cfg differs from dataset-build defaults on one or "
            "more of {R_a, feature_weights, lambda_, alpha}).",
            n_corrected, len(dataset),
        )
    else:
        logger.info(
            "Rewards used as-stored: current cfg matches dataset-build "
            "defaults on R_a, feature_weights, lambda_, and alpha; no "
            "recomputation needed."
        )


    # -- Persist config and run metadata UP FRONT (so it's saved even if fit crashes) -
    (out_dir / "config.json").write_text(
        json.dumps(cfg_to_dict(cfg), indent=2), encoding="utf-8")
    run_info = {
        "timestamp": timestamp,
        "tag": args.tag,
        "pickle_path": str(Path(args.pickle).resolve()),
        "csv_path": str(Path(args.csv).resolve()) if args.csv else None,
        "n_transitions": len(dataset),
        "cli_args": {k: v for k, v in vars(args).items()},
        "reward_composition": {k: v for k, v in diag_payload.items() if k != "feature_cols"}, # keep JSON small
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")

    # -- Fit --------------------------------------------------------------
    logger.info("Starting OPI.fit() …")
    t0 = time.perf_counter()
    model = OfflinePolicyIteration(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        model.fit(dataset)
    fit_time = time.perf_counter() - t0
    logger.info("OPI.fit() done in %.1f s  (cost_shift = %.3f, iterations = %d).",
                fit_time, model.cost_shift_, len(model.history["frobenius"]))

    # -- Save S and training history --------------------------------------
    np.savez(
        out_dir / "model.npz",
        S=model.S,
        cost_shift=np.array([model.cost_shift_]),
        frobenius=np.array(model.history["frobenius"]),
        min_eig_Saa=np.array(model.history["min_eig_Saa"]),
        sdp_status=np.array(model.history["sdp_status"], dtype=object),
        fit_time_seconds=np.array([fit_time]),
        feature_norm_lo=model.feature_norm._lo,
        feature_norm_hi=model.feature_norm._hi,
    )
    logger.info("Model saved to %s", out_dir / "model.npz")

    # -- Full evaluation --------------------------------------------------
    if args.skip_eval:
        logger.info("Skipping evaluation as requested (--skip-eval).")
        return out_dir

    pair_ids = extract_pair_ids(dataset, csv_df)
    if (args.cv_purge != "none" or args.cv_rating_stats == "fold") and csv_df is None:
        raise ValueError("--cv-purge and --cv-rating-stats fold need the transitions CSV (pass --csv).")
    pair_blocks = extract_pair_blocks(csv_df) if args.cv_purge != "none" else None
    fold_reward_fn = make_fold_reward_fn(csv_df, cfg, pair_ids) if args.cv_rating_stats == "fold" else None
    logger.info("Starting full evaluation (this can be slow — %d predict() calls per check).",
                len(dataset))
    t0 = time.perf_counter()
    report = run_full_evaluation(
        model, dataset,
        pair_ids=pair_ids,
        noise_fraction=args.stability_noise,
        n_perturbations=args.stability_perturbations,
        cv_k_folds=args.cv_folds,
        stability_seed=args.seed,
        cv_pair_blocks=pair_blocks,
        cv_purge=args.cv_purge,
        cv_fold_reward_fn=fold_reward_fn,
    )
    eval_time = time.perf_counter() - t0
    logger.info("Evaluation done in %.1f s.", eval_time)

    # -- Save report (txt + json + arrays) ---------------------------------
    paths = save_report(report, out_dir)
    logger.info("Report saved: %s", paths)

    # -- Pretty-print to console -------------------------------------------
    print_report(report)

    return out_dir


# ============================================================================
# |                                                                          |
# |  5) CLI (generated by Claude AI)                                         |
# |                                                                          |
# ============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # I/O
    p.add_argument("--pickle", required=True,
                   help="Path to the pre-built .pkl OPI Dataset.")
    p.add_argument("--csv", default=None,
                   help="Optional companion CSV (same rows as the pickle). Used for "
                        "pair_ids in CV and for authoritative sigmas in the diagnostic.")
    p.add_argument("--score-csv", default=None,
                   help="Path to the per-subject scores CSV consumed by "
                        "dataset_builder at build time (e.g. ExpS1_scores.csv). "
                        "Required for ANY ablation that differs from build-time "
                        "defaults — it is the ground truth used to recompute the "
                        "human-reward z-score statistics (μ_c, σ_c, μ_a, σ_a). "
                        "Baseline re-runs do not need it (fast path skips "
                        "recomputation). See reward.RewardComputer.fit_from_scores_csv.")
    p.add_argument("--output-dir", default="results",
                   help="Root directory where a per-run subfolder will be created.")
    p.add_argument("--tag", default="run",
                   help="Short label used to identify this experiment in the folder name.")

    # Dimensions
    p.add_argument("--n-features", type=int, default=6,
                   help="Number of biomechanical features. Must match the pickle.")
    p.add_argument("--include-params-in-state", action="store_true",
                   help="Append current absolute parameters p_k to the state vector.")

    # RL hyper-parameters
    p.add_argument("--gamma",       type=float, default=0.9)
    p.add_argument("--n-iter",      type=int,   default=100)
    p.add_argument("--tol",         type=float, default=1e-4)
    p.add_argument("--lambda-reg",  type=float, default=1e-3,
                   help="Tikhonov weight on ‖S‖²_F.")

    # Per-action-dimension R_a rescaling.
    # Three independent multipliers, defaulting to 1.0 (no rescaling).
    # Motivation: when the greedy QP saturates one specific face of the
    # admissible polytope on many samples (diagnosed via the action-space
    # saturation metrics of OPI_validation), tightening R_a selectively
    # on that dimension discourages the parasitic optimum without
    # perturbing the healthy dimensions.
    p.add_argument("--ra-scale-d-plant", type=float, default=1.0,
                   help="Multiplier applied to R_a diagonal entry for Δd_dors.")
    p.add_argument("--ra-scale-theta-trig", type=float, default=1.0,
                   help="Multiplier applied to R_a diagonal entry for Δθ_trig.")
    p.add_argument("--ra-scale-tau-lim", type=float, default=1.0,
                   help="Multiplier applied to R_a diagonal entry for Δτ_lim. "
                        "Run 795205 diagnostic recommends ≥5 to address the "
                        "~47 %% lower-bound saturation observed on τ_lim.")

    # Reward shaping
    p.add_argument("--lambda-human", type=float, default=0.5,
                   help="Balance between prosthesis and human reward.")
    p.add_argument("--alpha",        type=float, default=0.5,
                   help="Balance between comfort and perceived assistance.")
    p.add_argument("--assist-coding", choices=["centered", "raw"], default="centered",
                   help="'centered' z-scores -|A - 5| (bipolar assistance scale, used in the paper); 'raw' is the thesis coding.")
    p.add_argument("--action-cost", choices=["mixed", "separate"], default="mixed",
                   help="'mixed': aᵀR_a a weighted by (1 - λ); 'separate': kept at full weight whatever λ.")

    # Feature-weights preset (ablation axis on the per-feature weight w_i
    # of the prosthesis reward term).
    p.add_argument(
        "--feature-weights-preset",
        choices=["uniform", "inv_var"],
        default="uniform",
        help=(
            "Preset for the per-feature weight vector w used by the "
            "prosthesis reward r_p = - wᵀ (x̃²) - aᵀ R_a a.\n"
            "  'uniform' (baseline): wᵢ = 1 for all features.\n"
            "  'inv_var': wᵢ ∝ 1/⟨x̃ᵢ²⟩, renormalised so that Σᵢ wᵢ = "
            "n_features. This rebalances the feature term so that every "
            "feature contributes ~equally on average, neutralising the "
            "σʳᵉᶠ-induced dominance observed in the baseline diagnostic "
            "(f6 explained 38.9 %% of the feature term on run 795205)."
        ),
    )

    # Solver
    p.add_argument("--sdp-eps",           type=float, default=1e-6)
    p.add_argument("--cost-shift-margin", type=float, default=1e-3)
    p.add_argument("--no-bias", action="store_true",
                   help="Disable the bias term in μ. By default use_bias=True.")

    # Evaluation
    p.add_argument("--cv-folds",                type=int,   default=5)
    p.add_argument("--cv-purge", choices=["validation", "training", "none"], default="validation",
                   help="Side from which the pairs sharing a raw block with the other side of the split are removed.")
    p.add_argument("--cv-rating-stats", choices=["fold", "global"], default="fold",
                   help="'fold': z-score of the ratings computed on the training pairs of each fold; 'global': on all blocks.")
    p.add_argument("--stability-noise",         type=float, default=0.02,
                   help="Noise magnitude (fraction of per-feature std) for stability test.")
    p.add_argument("--stability-perturbations", type=int,   default=5)
    p.add_argument("--seed",                    type=int,   default=0)
    p.add_argument("--skip-eval", action="store_true",
                   help="Fit only, do not run the validation suite.")

    # Logging
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    out = run(args)
    print(f"\n[done] Results saved to: {out.resolve()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
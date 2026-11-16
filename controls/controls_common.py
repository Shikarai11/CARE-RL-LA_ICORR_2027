"""
Shared helpers for the offline controls (permutation test, lambda_h sweep, policy comparison)
=============================================================================================

- `load_inputs`: reads the all-to-all pickle, its companion CSV and the subject's score CSV.
- `make_cfg`: builds the OPIConfig as run.py does from its CLI options.
- `rewards_for`: rewards of every row for a given config, with the ratings of each block pair
  optionally replaced (permutation test). Same computation as run.py (RewardComputer.compute_batch).
- `fold_reward_fn`: same rewards, with the z-score of the ratings computed on the training pairs
  of a CV fold, as run.py does with --cv-rating-stats fold.
- `build_dataset`: Dataset of all rows with a given reward vector.
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code folder (OPI_core, OPI_validation, reward, elsa_configOPI, run)

from elsa_configOPI import OPIConfig            # noqa: E402
from OPI_core import Dataset, Transition         # noqa: E402
from reward import RewardComputer                # noqa: E402
from run import _resolve_reward_columns, extract_pair_blocks  # noqa: E402


@dataclass
class Inputs:
    transitions: List[Transition]   # original rows, in dataset order
    pair_ids: np.ndarray            # (N,) pair_id of every row
    pair_order: List[str]           # unique pair_ids in dataset (temporal) order
    pair_of_row: np.ndarray         # (N,) index of the pair of every row in pair_order
    pair_blocks: Dict[str, Tuple[int, int]]
    X: np.ndarray                   # (N, F) standardised features used by the reward
    A: np.ndarray                   # (N, 3) physical actions used by the reward
    comfort_pair: np.ndarray        # (P,) raw comfort score of each pair (block k+1)
    assist_pair: np.ndarray         # (P,) raw assistance score of each pair (block k+1)
    score_csv: str


def make_cfg(*, lambda_human: float = 0.5, alpha: float = 0.5, gamma: float = 0.9, lambda_reg: float = 1e-3,
             ra_scale=(1.0, 1.0, 1.0), n_iter: int = 100, assist_coding: str = "centered",
             action_cost: str = "mixed") -> OPIConfig:
    """Same construction as run.py with its default options (the paper baseline)."""
    ra_scale = np.asarray(ra_scale, dtype=float)
    return OPIConfig(
        n_features=6, n_actions=3, include_params_in_state=False,
        gamma=gamma, n_iterations=n_iter, tol=1e-4,
        lambda_=lambda_human, alpha=alpha, lambda_reg=lambda_reg,
        SDP_eps=1e-6, cost_shift_margin=1e-3, use_bias=True,
        R_a_per_dim_scale=None if np.allclose(ra_scale, 1.0) else ra_scale,
        feature_weights=None, assist_coding=assist_coding, action_cost=action_cost,
    )


def load_inputs(pickle_path: str, csv_path: str, score_csv_path: str) -> Inputs:
    with open(pickle_path, "rb") as f:
        dataset: Dataset = pickle.load(f)
    df = pd.read_csv(csv_path)
    if len(df) != len(dataset):
        raise ValueError(f"CSV has {len(df)} rows but the pickle has {len(dataset)}.")
    if (df.groupby("pair_id")["comfort_used"].nunique() > 1).any():
        raise ValueError("comfort_used is not constant within a pair; the block-level permutation assumes it is.")

    feat_cols, act_cols, _ = _resolve_reward_columns(df, make_cfg())
    pair_ids = df["pair_id"].astype(str).to_numpy()
    pair_order = list(pd.unique(pair_ids))
    index = {pid: k for k, pid in enumerate(pair_order)}
    first = df.groupby("pair_id", sort=False).first().loc[pair_order]
    inp = Inputs(
        transitions=list(dataset), pair_ids=pair_ids, pair_order=pair_order,
        pair_of_row=np.array([index[p] for p in pair_ids]), pair_blocks=extract_pair_blocks(df),
        X=df[feat_cols].to_numpy(float), A=df[act_cols].to_numpy(float),
        comfort_pair=first["comfort_used"].to_numpy(float), assist_pair=first["assistance_used"].to_numpy(float),
        score_csv=str(score_csv_path),
    )

    # Safety check: with the build-time config, the recomputed rewards must equal the stored ones.
    r_stored = np.array([t.r for t in inp.transitions])
    drift = float(np.max(np.abs(rewards_for(inp, make_cfg(assist_coding="raw")) - r_stored)))
    if drift > 1e-5:
        raise ValueError(f"Reward recomputation does not reproduce the stored rewards (max |dr| = {drift:.2e}). "
                         "Check that --score-csv is the file used to build the dataset.")
    return inp


def rewards_for(inp: Inputs, cfg: OPIConfig, comfort_pair: Optional[np.ndarray] = None,
                assist_pair: Optional[np.ndarray] = None, stats_pairs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reward of every row under cfg. comfort_pair / assist_pair replace the ratings of each pair (permutation test).
    The ratings are z-scored over all blocks of the score CSV, or over the pairs listed in stats_pairs (CV fold).
    """
    c_pair = inp.comfort_pair if comfort_pair is None else comfort_pair
    a_pair = inp.assist_pair if assist_pair is None else assist_pair
    if stats_pairs is None:
        rc = RewardComputer.fit_from_scores_csv(cfg, inp.score_csv)
    else:
        rc = RewardComputer(cfg).fit_human_stats(c_pair[stats_pairs], a_pair[stats_pairs])
    return rc.compute_batch(inp.X, inp.A, c_pair[inp.pair_of_row], a_pair[inp.pair_of_row])


def fold_reward_fn(inp: Inputs, cfg: OPIConfig, comfort_pair: Optional[np.ndarray] = None,
                   assist_pair: Optional[np.ndarray] = None):
    """fold_rewards(train_idx) for OPI_validation.cross_validate / fast_opi.grouped_cv_overfit."""
    def fold_rewards(train_idx: np.ndarray) -> np.ndarray:
        return rewards_for(inp, cfg, comfort_pair, assist_pair, stats_pairs=np.unique(inp.pair_of_row[train_idx]))
    return fold_rewards


def build_dataset(inp: Inputs, r: np.ndarray) -> Dataset:
    """Dataset of all rows, in their original order, with rewards r."""
    ds = Dataset()
    for t, ri in zip(inp.transitions, r):
        ds.add(replace(t, r=float(ri)))
    return ds


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def done_ids(path: Path, key: str):
    """Records already written to `path` (lets an interrupted job resume)."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)[key] for line in f if line.strip()}

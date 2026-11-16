"""
Validation Module for the OPI Controller

Author : Raphaël VASSART
=====================================================================
This module answers one critical question that the rest of the code cannot:
"Is the learned policy actually good?"
 
Since we have no patient to test on and no simulator, every check here is purely data-driven.

It only uses the offline dataset and the learned S matrix.
This is therefore the only safety net before any clinical deployment.
"""

from __future__ import annotations

import copy
import logging
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Sequence

import numpy as np

from elsa_configOPI import OPIConfig
from OPI_core import Dataset, OfflinePolicyIteration, Transition

logger = logging.getLogger(__name__)

# Physical parameter names (for readable output)
_PARAM_NAMES = ["d_dors [Nm·s/rad]", "θ_trig [rad]", "τ_lim [Nm]"]

# ============================================================================
# |                                                                          |
# |  1) RESULT DATACLASSES                                                   |
# |                                                                          |
# ============================================================================

@dataclass
class BellmanMetrics:
    """
    Results of the Bellman consistency check.
    A residual of 0 means the Bellman equation is perfectly satisfied for that transition.
    """
    residuals: np.ndarray # (N,) raw per-transition residuals ε_n
    rmse: float # Root Mean Squared Error of residuals
    mean: float # Mean residual (signed — bias indicator)
    std: float # Std of residuals (spread indicator)
    max_abs: float # Max absolute residual (worst-case sample)
    rel_rmse: float # RMSE / mean(U) — normalised, easier to interpret
    
    
@dataclass
class ActionMetrics:
    """
    Results of the action distribution sanity check.
    """
    # Per-parameter statistics of predicted actions (physical units)
    pred_mean: np.ndarray    
    pred_std: np.ndarray    
    # Same for the original dataset actions
    data_mean: np.ndarray   
    data_std: np.ndarray    
    # Per-parameter Mean Absolute Error between prediction and data
    mae_per_param: np.ndarray    
    # % of samples where the resulting p_k+1 is within 5 % of a physical bound.  High values = policy stuck at walls
    pct_near_bound: float
    # % of samples where ‖a_pred‖ < 1 % of the action range. High values = policy learned to do nothing.
    pct_zero_action: float   
    # Action-space  saturation diagnostic
    pct_lo_bound_per_dim: np.ndarray
    pct_hi_bound_per_dim: np.ndarray
    pct_any_bound_per_dim: np.ndarray
    pct_any_action_saturated: float
    
@dataclass
class PolicyImprovementMetrics:
    """ 
    Results of the policy improvement verification.
    The greedy policy should satisfy Q̂(x, π(x)) ≤ Q̂(x, a_data) for all x.
    """
    pct_improved: float # % of samples where policy Q ≤ dataset Q (we want 100%)
    mean_q_reduction: float # average Q reduction brought by the policy (positive = good)
    q_policy: np.ndarray # (N,) Q-values under the learned policy
    q_dataset: np.ndarray # (N,) Q-values under the dataset actions
    

@dataclass
class StabilityMetrics:
    """
    Results of the robustness / stability test.

    sensitivity_n = ‖π(x+δ) - π(x)‖₂ / ‖δ‖₂ for each sample n.
    A value > 1 means the action changes faster than the state perturbation.
    The "_norm" values divide each action change by the half-range of its parameter, so that the three parameters weigh equally whatever their units.
    """
    sensitivities: np.ndarray # per-sample sensitivity values
    mean: float
    max: float
    p95: float # 95th percentile
    # % of samples where at least one action dimension flips sign under perturbation
    pct_sign_change: float 
    noise_std_used: float # The noise level (physical units) used
    sensitivities_norm: Optional[np.ndarray] = None
    mean_norm: float = float("nan")
    p95_norm: float = float("nan")
    
    
@dataclass
class KFoldFoldMetrics:
    """
    Metrics for one fold of grouped K-fold cross-validation.
    """
    fold_index: int
    train_rmse: float
    val_rmse: float
    overfitting_ratio: float
    train_bellman: BellmanMetrics
    val_bellman: BellmanMetrics
    train_actions: ActionMetrics
    val_actions: ActionMetrics
    n_train: int
    n_val: int
    reassigned_pair_ids: List[str]
    n_train_pairs: int = 0
    n_val_pairs: int = 0


@dataclass
class CrossValMetrics:
    """
    Results of grouped K-fold cross-validation.

    The split is first defined over pseudo-transition rows, then corrected so that all pseudo-transitions from the same real block pair (same pair_id) end up entirely in either train or validation for the considered fold.
    """
    k_folds: int
    folds: List[KFoldFoldMetrics]
    mean_train_rmse: float
    mean_val_rmse: float
    mean_overfitting_ratio: float
    std_val_rmse: float
    total_n_train: int
    total_n_val: int
    purge: str = "none"
    fold_rewards: bool = False # True if the rewards were recomputed from the training pairs of each fold
    
    
@dataclass
class EvaluationReport:
    """
    Aggregated report produced by run_full_evaluation().
    Holds the results of all five checks in a single object.
    """
    n_samples: int
    bellman: BellmanMetrics
    actions: ActionMetrics
    policy_improvement: PolicyImprovementMetrics
    stability: StabilityMetrics
    cross_val: CrossValMetrics
    
    


# ============================================================================
# |                                                                          |
# |  2) EVALUATOR CLASS                                                      |
# |                                                                          |
# ============================================================================

class OPIEvaluator:
    """
    Wraps a fitted OfflinePolicyIteration model and provides all evaluation checks.

    The evaluator never modifies the model or the dataset.
    All methods can be called independently.
    """

    def __init__(self, model: OfflinePolicyIteration):
        if model.S is None:
            raise RuntimeError(
                "The model has not been fitted yet. Call model.fit(dataset) first."
            )
        self.model = model
        self.cfg   = model.cfg
        self._S = model.S
        self._gamma = model.cfg.gamma
        self._cost_shift = model.cost_shift_ # learned during fit() on the training set
        
    
    # -- HELPERS Function ----------------------------------------------
    
    def _q_value(self, x_state: np.ndarray, a_norm: np.ndarray):
        """
        Evaluate the learned Q-function at one (state, normalised-action) pair.
            Q̂(x, a) = μᵀ S μ  where μ = [1 (bias?); x; a]
        Returns the scalar Q-value (in cost space, i.e. U units).
        """
        mu = self.model._joint_vector(x_state, a_norm)
        return float(mu @ self._S @ mu)


    def _extract_arrays(self, dataset: Dataset):
        """
        Convert a dataset to NumPy arrays and build the normalised internal representations (states, actions, costs) that all checks need.

        Returns a dict with keys:
            - x_feat, a, x1_feat, r, p, p1: raw arrays
            - Xn, X1n: normalised state matrices
            - An: normalised action matrix
            - U: cost vector 
        """
        x_feat, a, x1_feat, r, p, p1 = dataset.arrays(self.cfg)

        # Normalise using the normalisers fitted on the training set.
        Xn = self.model._build_state_matrix(x_feat, p)
        X1n = self.model._build_state_matrix(x1_feat, p1)
        An = self.model.action_norm.normalize(a)

        # Convert rewards to costs using the TRAINING cost_shift_.
        # I must use the same shift as during training so that U values are on the same scale as the Q-values stored in S.
        U = self._cost_shift - r

        return dict(x_feat=x_feat, a=a, x1_feat=x1_feat, r=r, p=p, p1=p1, Xn=Xn, X1n=X1n, An=An, U=U)
        
        
    # -------------------------------------------------------------------------
    # CHECK 1 — BELLMAN ERROR
    # -------------------------------------------------------------------------
    
    def bellman_error(self, dataset: Dataset):
        """
        Measure how well the learned Q-function satisfies the Bellman equation.
        For each transition n in the dataset:
            ε_n = Q̂(xₙ, aₙ) - γ · Q̂(x'ₙ, π(x'ₙ)) - Uₙ
        A residual of zero means that the Bellman equation is exactly satisfied.
        In practice we want RMSE(ε) to be small relative to the scale of U.
        """
        arrays = self._extract_arrays(dataset)
        Xn, X1n, An, U, p1 = (
            arrays["Xn"], arrays["X1n"], arrays["An"],
            arrays["U"],  arrays["p1"],
        )
        N = Xn.shape[0]

        residuals = np.empty(N)
        for n in range(N):
            # Q-value at the observed (state, action)
            q_current = self._q_value(Xn[n], An[n])

            # Apply the learned policy at the NEXT state to get the "target"
            # I call my model.predict() then re-normalise the returned physical action for _q_value.
            x1_feat_n = arrays["x1_feat"][n]
            p1_n = p1[n]
            a1_n_phys, _ = self.model.predict(x1_feat_n, p1_n)
            a1_n_norm = self.model.action_norm.normalize(a1_n_phys)

            # Q-value at the next (state, policy-action)
            q_next = self._q_value(X1n[n], a1_n_norm)

            # Bellman residual
            residuals[n] = q_current - self._gamma * q_next - U[n]

        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mean_U = float(np.mean(np.abs(U)))
        rel_rmse = rmse / mean_U if mean_U > 1e-12 else float("nan")

        return BellmanMetrics(residuals, rmse, float(np.mean(residuals)), float(np.std(residuals)), float(np.max(np.abs(residuals))), rel_rmse)
    
    
    
    # -------------------------------------------------------------------------
    # CHECK 2 — ACTION DISTRIBUTION
    # -------------------------------------------------------------------------

    def action_distribution(self, dataset: Dataset):
        """
        Analyse the distribution of actions proposed by the learned policy and compare them to the dataset actions.
        Is policy hallucinating about aberrant actions?

        Two red flags I need to pay attention:
          - High pct_near_bound: the policy systematically pushes parameters to their physical limits -> it has learned to exploit the constraint rather than the biomechanical signal.
          - High MAE vs dataset: the policy differs a lot from clinical behaviour, which is risky if the data coverage is poor.
        """
        x_feat, a, _, _, p, _ = dataset.arrays(self.cfg)
        N = len(dataset)

        p_min = self.cfg.prosthesis.p_min
        p_max = self.cfg.prosthesis.p_max
        p_range = p_max - p_min
        half_range = self.cfg.prosthesis.p_half_range

        # Predict actions for every state in the dataset.
        a_pred = np.empty_like(a)
        p_next = np.empty_like(p)
        for n in range(N):
            a_pred[n], p_next[n] = self.model.predict(x_feat[n], p[n])

        # --- % near physical bound 
        # I check the resulting parameter p_k+1 = clip(p_k + a_k, p_min, p_max).
        # "Near bound" = within 5 % of the physical range on either side.
        bound_tol = 0.05 * p_range # tolerance vector
        near_lo = np.any(p_next - p_min < bound_tol, axis=1)
        near_hi = np.any(p_max  - p_next < bound_tol, axis=1)
        pct_near_bound = float(np.mean(near_lo | near_hi) * 100.0)

        # --- % zero action 
        # "Zero action" = norm of the normalised action < 1 % of the action range.
        a_pred_norm = self.model.action_norm.normalize(a_pred)
        pct_zero = float(np.mean(np.linalg.norm(a_pred_norm, axis=1) < 0.01) * 100.0)
        
        # --- Action-space saturation diagnostics
        # State-dependent admissible action polytope in PHYSICAL units:
        #     a_lo_phys(p) = p_min - p
        #     a_hi_phys(p) = p_max - p
        # mapped into the normalised action space via division by the per-parameter half-range (same scaling as ActionNormalizer).
        a_lo_phys = p_min - p
        a_hi_phys = p_max - p
        a_lo_n = a_lo_phys / half_range
        a_hi_n = a_hi_phys / half_range
        # Tolerance in normalised action space. 
        sat_tol = 1e-3
        hits_lo = (a_pred_norm - a_lo_n) < sat_tol
        hits_hi = (a_hi_n - a_pred_norm) < sat_tol
        hits_any_dim = hits_lo | hits_hi
        
        pct_lo_per_dim = hits_lo.mean(axis=0) * 100.0
        pct_hi_per_dim = hits_hi.mean(axis=0) * 100.0
        pct_any_per_dim = hits_any_dim.mean(axis=0) * 100.0
        pct_any_action_saturated = float(np.any(hits_any_dim, axis=1).mean() * 100.0)
        

        return ActionMetrics(
            pred_mean = a_pred.mean(axis=0),
            pred_std = a_pred.std(axis=0),
            data_mean = a.mean(axis=0),
            data_std = a.std(axis=0),
            mae_per_param = np.abs(a_pred - a).mean(axis=0),
            pct_near_bound = pct_near_bound,
            pct_zero_action = pct_zero,
            pct_lo_bound_per_dim = pct_lo_per_dim,
            pct_hi_bound_per_dim = pct_hi_per_dim,
            pct_any_bound_per_dim = pct_any_per_dim,
            pct_any_action_saturated = pct_any_action_saturated,
        )
        
    
    
    # -------------------------------------------------------------------------
    # CHECK 3 — POLICY IMPROVEMENT
    # -------------------------------------------------------------------------

    def policy_improvement_check(self, dataset: Dataset):
        """
        Check the overall coherence of the learned policy with the Q-function stored in S.
        We know that the policy π is defined as: π(x) = argmin_a Q̂(x, a)
        By construction it should satisfy: Q̂(x, π(x)) ≤ Q̂(x, a_dataset)
        We should reach 100% of samples otherwise it indicates a numerical problem (no bugs !!).
        """
        arrays = self._extract_arrays(dataset)
        Xn, An = arrays["Xn"], arrays["An"]
        x_feat, a, p = arrays["x_feat"], arrays["a"], arrays["p"]
        N = Xn.shape[0]

        q_policy = np.empty(N)
        q_dataset = np.empty(N)

        for n in range(N):
            # Q-value under the learned policy (what the policy would recommend)
            a_n_phys, _ = self.model.predict(x_feat[n], p[n])
            a_n_norm = self.model.action_norm.normalize(a_n_phys)
            q_policy[n] = self._q_value(Xn[n], a_n_norm)

            # Q-value under the dataset action (what actually happened during the experimental protocol)
            q_dataset[n] = self._q_value(Xn[n], An[n])

        improved = q_policy <= q_dataset + 1e-8 # small tolerance for float equality
        reduction = q_dataset - q_policy # positive = policy is better

        return PolicyImprovementMetrics(
            pct_improved = float(np.mean(improved) * 100.0),
            mean_q_reduction = float(np.mean(reduction)),
            q_policy = q_policy,
            q_dataset = q_dataset,
        )


    # -------------------------------------------------------------------------
    # CHECK 4 — STABILITY / ROBUSTNESS
    # -------------------------------------------------------------------------

    def stability_test(self, dataset: Dataset, noise_fraction: float = 0.02, n_perturbations: int = 5, seed: int = 0):
        """
        Measure the sensitivity of the policy to small perturbations of the input features (simulating measurement noise or session-to-session biomechanical variability).

        For each sample n, I add Gaussian noise δ to x_feat_n and compute:
            sensitivity_n = ‖π(x + δ) - π(x)‖₂ / ‖δ‖₂

        The perturbation magnitude is noise_fraction * std(x_feat) per dimension, which corresponds to simulating realistic measurement noise relative to the observed variability in the dataset.
        I repeat n_perturbations times per sample and average, to reduce the influence of lucky/unlucky noise draws.
        """
        rng = np.random.default_rng(seed)
        x_feat, _, _, _, p, _ = dataset.arrays(self.cfg)
        N = len(dataset)

        # Noise magnitude: fraction of the per-feature std across the dataset.
        feature_std = x_feat.std(axis=0)
        noise_std = noise_fraction * np.maximum(feature_std, 1e-8)
        # Scalar representative noise std for reporting.
        noise_std_scalar = float(noise_std.mean())

        half_range = self.cfg.prosthesis.p_half_range
        sensitivities = np.zeros(N)
        sensitivities_norm = np.zeros(N)
        sign_changes = np.zeros(N, dtype=bool)

        for n in range(N):
            # Baseline policy action in physical units.
            a_base, _ = self.model.predict(x_feat[n], p[n])

            sens_draws, sens_norm_draws = [], []
            for _ in range(n_perturbations):
                # Perturb the biomechanical features.
                delta = rng.normal(0.0, noise_std)
                x_perturbed = x_feat[n] + delta

                # Policy action under perturbed state.
                a_pert, _ = self.model.predict(x_perturbed, p[n])

                # Sensitivity = ‖Δa‖ / ‖δ‖
                delta_a_norm = np.linalg.norm(a_pert - a_base)
                delta_x_norm = np.linalg.norm(delta)
                if delta_x_norm > 1e-12:
                    sens_draws.append(delta_a_norm / delta_x_norm)
                    sens_norm_draws.append(np.linalg.norm((a_pert - a_base) / half_range) / delta_x_norm)

                # Check if any action dimension flips its sign.
                if np.any(np.sign(a_pert) != np.sign(a_base)):
                    sign_changes[n] = True

            sensitivities[n] = np.mean(sens_draws) if sens_draws else 0.0
            sensitivities_norm[n] = np.mean(sens_norm_draws) if sens_norm_draws else 0.0

        return StabilityMetrics(
            sensitivities = sensitivities,
            mean = float(np.mean(sensitivities)),
            max = float(np.max(sensitivities)),
            p95 = float(np.percentile(sensitivities, 95)),
            pct_sign_change = float(np.mean(sign_changes) * 100.0),
            noise_std_used = noise_std_scalar,
            sensitivities_norm = sensitivities_norm,
            mean_norm = float(np.mean(sensitivities_norm)),
            p95_norm = float(np.percentile(sensitivities_norm, 95)),
        )


# ============================================================================
# |                                                                          |
# |  3) CROSS-VALIDATION                                                     |
# |                                                                          |
# ============================================================================

def _dataset_from_indices(all_transitions: List, indices: np.ndarray, rewards: Optional[np.ndarray] = None):
    out = Dataset()
    for idx in indices:
        t = all_transitions[int(idx)]
        out.add(t if rewards is None else replace(t, r=float(rewards[int(idx)])))
    return out


def _make_ordered_kfold_indices(n_samples: int, k_folds: int):
    """
    Build deterministic K folds over row indices, preserving dataset order.
    Every row appears exactly once in a validation fold.
    """
    indices = np.arange(n_samples, dtype=int)
    return [arr.astype(int) for arr in np.array_split(indices, k_folds)]

def _majority_pair_reassignment(pair_ids: Sequence[str], initial_val_idx: np.ndarray):
    """
    Start from an initial row-based validation split, then enforce that every pair_id is entirely in either train or validation.
      - if a pair is split across train/val, I move ALL pseudo-transitions of that pair to the side containing the majority of that pair.
      - if there is a tie, send the whole pair to validation so the fold still has a chance to validate that real transition.
    But in practice it will never happen as the dataset contains 15 transitions with exactly 1600 pseudo-transitions each, and the initial split is done at the transition level, so all pseudo-transitions from a given transition will already be in the same fold.
    """
    pair_ids = np.asarray(pair_ids)
    n = len(pair_ids)
    val_mask = np.zeros(n, dtype=bool)
    val_mask[np.asarray(initial_val_idx, dtype=int)] = True

    reassigned_pairs: List[str] = []

    unique_pairs = np.unique(pair_ids)
    for pid in unique_pairs:
        pair_mask = (pair_ids == pid)
        pair_idx = np.flatnonzero(pair_mask)

        n_val = int(np.sum(val_mask[pair_idx]))
        n_train = int(len(pair_idx) - n_val)

        if n_val == 0 or n_train == 0:
            continue

        reassigned_pairs.append(str(pid))
        if n_val >= n_train:
            val_mask[pair_idx] = True
        else:
            val_mask[pair_idx] = False

    final_val_idx = np.flatnonzero(val_mask)
    final_train_idx = np.flatnonzero(~val_mask)
    return final_train_idx, final_val_idx, reassigned_pairs


CV_PURGES = ("none", "validation", "training")


def make_cv_splits(pair_ids: Sequence[str], k_folds: int, *, pair_blocks: Optional[Mapping[str, Tuple[int, int]]] = None, purge: str = "none"):
    """
    Build the K grouped folds of cross_validate as (train_idx, val_idx, reassigned_pairs).

    Two consecutive block pairs (k -> k+1 and k+1 -> k+2) share the raw block k+1, so the last training pair and the first validation pair of a fold can share a block.
    With purge != "none", every pair that shares a raw block with the other side of the split is removed:
      - purge = "validation" : from the validation fold, the training set stays complete;
      - purge = "training" : from the training set, the validation fold stays complete.
    pair_blocks maps each pair_id to its two block indices (k, k+1).
    """
    if purge not in CV_PURGES:
        raise ValueError(f"purge must be one of {CV_PURGES}, got {purge!r}.")
    if purge != "none" and pair_blocks is None:
        raise ValueError("pair_blocks is required to purge the folds.")

    pair_ids = np.asarray(pair_ids)
    splits = []
    for initial_val_idx in _make_ordered_kfold_indices(len(pair_ids), k_folds):
        train_idx, val_idx, reassigned_pairs = _majority_pair_reassignment(pair_ids, initial_val_idx)

        if purge != "none":
            purged_side, other_side = (val_idx, train_idx) if purge == "validation" else (train_idx, val_idx)
            other_blocks = set().union(*(pair_blocks[pid] for pid in np.unique(pair_ids[other_side])))
            shared = [pid for pid in np.unique(pair_ids[purged_side]) if set(pair_blocks[pid]) & other_blocks]
            kept = purged_side[~np.isin(pair_ids[purged_side], shared)]
            if purge == "validation":
                val_idx = kept
            else:
                train_idx = kept

        splits.append((train_idx, val_idx, reassigned_pairs))
    return splits


def cross_validate(cfg: OPIConfig, dataset: Dataset, *, pair_ids: Sequence[str], k_folds: int = 5, pair_blocks: Optional[Mapping[str, Tuple[int, int]]] = None, purge: str = "none", fold_reward_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    """
    Perform grouped K-fold cross-validation with the "majority regrouping" safeguard.

    Why I implement this logic
    ----------------------------
    In an all-to-all dataset, many pseudo-transitions come from the same real block pair.
    I therefore want to avoid a situation where pseudo-transitions from the same pair_id are split between train and validation.

    Implemented logic
    -----------------
    1) build a deterministic ordered K-fold split over ROWS;
    2) for each fold, if a pair_id is split across train/val, move the whole pair_id to the side containing the majority of its pseudo-transitions;
    3) train a fresh OPI model on the resulting training subset and evaluate on both sets.

    But in my current dataset (24000 rows = 15 pairs x 1600 rows, with ordered rows), folds should naturally align to full pairs already.

    purge : removes the pairs that share a raw block with the other side of the split (see make_cv_splits).
    fold_reward_fn : if given, fold_reward_fn(train_idx) returns the (N,) rewards of the fold, so that the z-score of the ratings uses the training pairs only.
    """
    N = len(dataset)
    if N < k_folds:
        raise ValueError(f"K-fold CV needs at least k_folds samples; got N={N}, k_folds={k_folds}.")

    if len(pair_ids) != N:
        raise ValueError(
            f"pair_ids must have the same length as dataset. Got {len(pair_ids)} pair_ids for {N} samples."
        )

    all_transitions = list(dataset)
    splits = make_cv_splits(pair_ids, k_folds, pair_blocks=pair_blocks, purge=purge)
    pair_ids_arr = np.asarray(pair_ids)

    fold_reports: List[KFoldFoldMetrics] = []

    for fold_idx, (train_idx, val_idx, reassigned_pairs) in enumerate(splits):
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(
                f"Fold {fold_idx + 1}/{k_folds} produced an empty train or validation set after pair regrouping and purge. "
                f"train={len(train_idx)}, val={len(val_idx)}"
            )

        fold_rewards = fold_reward_fn(train_idx) if fold_reward_fn is not None else None
        train_dataset = _dataset_from_indices(all_transitions, train_idx, fold_rewards)
        val_dataset = _dataset_from_indices(all_transitions, val_idx, fold_rewards)

        logger.info(
            "Grouped K-fold CV — fold %d/%d: %d train / %d val / %d reassigned pairs",
            fold_idx + 1, k_folds, len(train_idx), len(val_idx), len(reassigned_pairs),
        )

        train_model = OfflinePolicyIteration(copy.deepcopy(cfg))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_model.fit(train_dataset)

        evaluator = OPIEvaluator(train_model)

        train_bellman = evaluator.bellman_error(train_dataset)
        val_bellman = evaluator.bellman_error(val_dataset)
        train_actions = evaluator.action_distribution(train_dataset)
        val_actions = evaluator.action_distribution(val_dataset)

        overfitting_ratio = (
            val_bellman.rmse / train_bellman.rmse
            if train_bellman.rmse > 1e-12 else float("nan")
        )

        fold_reports.append(
            KFoldFoldMetrics(
                fold_index=fold_idx + 1,
                train_rmse=train_bellman.rmse,
                val_rmse=val_bellman.rmse,
                overfitting_ratio=overfitting_ratio,
                train_bellman=train_bellman,
                val_bellman=val_bellman,
                train_actions=train_actions,
                val_actions=val_actions,
                n_train=len(train_idx),
                n_val=len(val_idx),
                reassigned_pair_ids=reassigned_pairs,
                n_train_pairs=len(np.unique(pair_ids_arr[train_idx])),
                n_val_pairs=len(np.unique(pair_ids_arr[val_idx])),
            )
        )

    mean_train_rmse = float(np.mean([f.train_rmse for f in fold_reports]))
    mean_val_rmse = float(np.mean([f.val_rmse for f in fold_reports]))
    mean_overfit = float(np.mean([f.overfitting_ratio for f in fold_reports]))
    std_val_rmse = float(np.std([f.val_rmse for f in fold_reports]))

    total_n_train = int(np.sum([f.n_train for f in fold_reports]))
    total_n_val = int(np.sum([f.n_val for f in fold_reports]))

    return CrossValMetrics(
        k_folds=k_folds,
        folds=fold_reports,
        mean_train_rmse=mean_train_rmse,
        mean_val_rmse=mean_val_rmse,
        mean_overfitting_ratio=mean_overfit,
        std_val_rmse=std_val_rmse,
        total_n_train=total_n_train,
        total_n_val=total_n_val,
        purge=purge,
        fold_rewards=fold_reward_fn is not None,
    )

# ============================================================================
# |                                                                          |
# |  4) RUN COMPLETE EVALUATION                                              |
# |                                                                          |
# ============================================================================


def run_full_evaluation(model: OfflinePolicyIteration, dataset: Dataset, *, pair_ids: Sequence[str], noise_fraction: float = 0.02, n_perturbations: int = 5, cv_k_folds: int = 5, stability_seed: int = 0, cv_pair_blocks: Optional[Mapping[str, Tuple[int, int]]] = None, cv_purge: str = "none", cv_fold_reward_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    """
    Run all five validation checks on a fitted OPI model and return a consolidated EvaluationReport.

    This is the main entry point. Call this after model.fit(dataset) to get a complete picture of the model quality before any clinical use.
    """
    logger.info("Starting full OPI evaluation on %d samples.", len(dataset))
    evaluator = OPIEvaluator(model)

    logger.info("  [1/5] Bellman error…")
    bellman = evaluator.bellman_error(dataset)

    logger.info("  [2/5] Action distribution…")
    actions = evaluator.action_distribution(dataset)

    logger.info("  [3/5] Policy improvement check…")
    pi_check = evaluator.policy_improvement_check(dataset)

    logger.info("  [4/5] Stability test…")
    stability = evaluator.stability_test(dataset, noise_fraction, n_perturbations, stability_seed)

    logger.info("  [5/5] Group-safe K-fold cross-validation (K=%d)…", cv_k_folds)
    cv = cross_validate(model.cfg, dataset, pair_ids=pair_ids, k_folds=cv_k_folds, pair_blocks=cv_pair_blocks, purge=cv_purge, fold_reward_fn=cv_fold_reward_fn)

    logger.info("Evaluation complete.")
    return EvaluationReport(len(dataset), bellman, actions, pi_check, stability, cv)


# ============================================================================
# |                                                                          |
# |  5) HUMAN-READABLE REPORT PRINTER FOR ME (generated by Generative AI)    |
# |                                                                          |
# ============================================================================

def print_report(report: EvaluationReport):
    """Print a structured, human-readable summary of an EvaluationReport."""
    sep  = "═" * 72
    sep2 = "─" * 72

    print(f"\n{sep}")
    print("  OPI EVALUATION REPORT")
    print(f"  Dataset size : {report.n_samples} transitions")
    print(sep)

    # ── 1. Bellman Error ─────────────────────────────────────────────────────
    b = report.bellman
    print(f"\n{sep2}")
    print("  1. BELLMAN CONSISTENCY")
    print(sep2)
    print(f"  RMSE of residuals        : {b.rmse:.6f}")
    print(f"  Relative RMSE (/ mean U) : {b.rel_rmse:.6f}")
    print(f"  Mean residual            : {b.mean:.6f}")
    print(f"  Std of residuals         : {b.std:.6f}")
    print(f"  Max absolute residual    : {b.max_abs:.6f}")

    # ── 2. Action Distribution ───────────────────────────────────────────────
    am = report.actions
    print(f"\n{sep2}")
    print("  2. ACTION DISTRIBUTION")
    print(sep2)
    print(f"  {'Parameter':<22}  {'Data mean':>12}  {'Data std':>10}  {'Pred mean':>12}  {'Pred std':>10}  {'MAE':>10}")
    print(f"  {'-'*82}")
    for i, name in enumerate(_PARAM_NAMES):
        print(
            f"  {name:<22}  {am.data_mean[i]:>12.6f}  {am.data_std[i]:>10.6f}"
            f"  {am.pred_mean[i]:>12.6f}  {am.pred_std[i]:>10.6f}  {am.mae_per_param[i]:>10.6f}"
        )
    print(f"\n  % predicted params near physical bounds : {am.pct_near_bound:.2f} %")
    print(f"  % predicted actions ~ zero               : {am.pct_zero_action:.2f} %")
    
    print()
    print(f"  Action-space saturation — % samples where the normalised")
    print(f"  action lies on a face of the admissible polytope (per dim):")
    print(f"  {'Parameter':<22}  {'lower bound':>12}  {'upper bound':>12}  {'either':>10}")
    print(f"  {'-'*62}")
    for i, name in enumerate(_PARAM_NAMES):
        print(
            f"  {name:<22}  {am.pct_lo_bound_per_dim[i]:>11.2f} %"
            f"  {am.pct_hi_bound_per_dim[i]:>11.2f} %"
            f"  {am.pct_any_bound_per_dim[i]:>9.2f} %"
        )
    print(f"  % samples with ≥ 1 action dim saturated : {am.pct_any_action_saturated:.2f} %")

    # ── 3. Policy Improvement ────────────────────────────────────────────────
    pi = report.policy_improvement
    print(f"\n{sep2}")
    print("  3. POLICY IMPROVEMENT")
    print(sep2)
    print(f"  % samples where Q(x,π(x)) ≤ Q(x,a_data) : {pi.pct_improved:.2f} %")
    print(f"  Mean Q reduction by policy                : {pi.mean_q_reduction:.6f}")

    # ── 4. Stability ─────────────────────────────────────────────────────────
    st = report.stability
    print(f"\n{sep2}")
    print("  4. STABILITY / ROBUSTNESS")
    print(sep2)
    print(f"  Noise level used (mean feature std × fraction) : {st.noise_std_used:.6f}")
    print(f"  Mean sensitivity  ‖Δa‖/‖δ‖                    : {st.mean:.6f}")
    print(f"  Max  sensitivity                               : {st.max:.6f}")
    print(f"  95th-percentile sensitivity                    : {st.p95:.6f}")
    print(f"  95th-percentile sensitivity (half-range units) : {st.p95_norm:.6f}")
    print(f"  % samples with action sign flip                : {st.pct_sign_change:.2f} %")

    # ── 5. Cross-Validation ──────────────────────────────────────────────────
    cv = report.cross_val
    print(f"\n{sep2}")
    print(f"  5. GROUP-SAFE K-FOLD CROSS-VALIDATION  (K={cv.k_folds})")
    print(sep2)
    print(f"  Purge of shared blocks : {cv.purge}")
    print(f"  Rewards per fold       : {'recomputed on the training pairs' if cv.fold_rewards else 'as in the dataset'}")
    print(f"  Mean train RMSE        : {cv.mean_train_rmse:.6f}")
    print(f"  Mean validation RMSE   : {cv.mean_val_rmse:.6f}")
    print(f"  Std validation RMSE    : {cv.std_val_rmse:.6f}")
    print(f"  Mean overfitting ratio : {cv.mean_overfitting_ratio:.6f}")
    print()

    for fold in cv.folds:
        print(f"  Fold {fold.fold_index}/{cv.k_folds}")
        print(f"    train / val sizes     : {fold.n_train} / {fold.n_val}  ({fold.n_train_pairs} / {fold.n_val_pairs} pairs)")
        print(f"    train RMSE            : {fold.train_rmse:.6f}")
        print(f"    val RMSE              : {fold.val_rmse:.6f}")
        print(f"    overfitting ratio     : {fold.overfitting_ratio:.6f}")
        print(f"    reassigned pair_ids   : {len(fold.reassigned_pair_ids)}")
        if fold.reassigned_pair_ids:
            print(f"      {fold.reassigned_pair_ids}")
        print()

    print(f"\n{sep}\n")
    
    
# ============================================================================
# |                                                                          |
# |  6) MACHINE-READABLE EXPORT  (for later aggregation & plotting)          |
# |                                                                          |
# ============================================================================

def report_to_dict(report: EvaluationReport):
    """
    Convert an EvaluationReport into a plain Python dict that is JSON-serialisable.

    All numpy arrays are either:
      - flattened to lists of Python floats (small arrays: eigenvalues, per-param stats),
      - summarised by a few scalars (large arrays: residuals, sensitivities, Q-values), with a reference to the side-car .npy file where the full array lives.

    Large arrays are NOT dumped here; use save_report() below to persist them.
    """
    def _arr(a):
        """Convert small arrays to lists; return None for None."""
        if a is None:
            return None
        a = np.asarray(a)
        return a.tolist()

    def _bell(b: BellmanMetrics):
        return {
            "rmse": float(b.rmse),
            "mean": float(b.mean),
            "std": float(b.std),
            "max_abs": float(b.max_abs),
            "rel_rmse": float(b.rel_rmse) if np.isfinite(b.rel_rmse) else None,
            "n": int(len(b.residuals)),
        }

    def _act(a: ActionMetrics):
        return {
            "pred_mean": _arr(a.pred_mean),
            "pred_std": _arr(a.pred_std),
            "data_mean": _arr(a.data_mean),
            "data_std": _arr(a.data_std),
            "mae_per_param": _arr(a.mae_per_param),
            "pct_near_bound": float(a.pct_near_bound),
            "pct_zero_action": float(a.pct_zero_action),
            "pct_lo_bound_per_dim": _arr(a.pct_lo_bound_per_dim),
            "pct_hi_bound_per_dim": _arr(a.pct_hi_bound_per_dim),
            "pct_any_bound_per_dim": _arr(a.pct_any_bound_per_dim),
            "pct_any_action_saturated": float(a.pct_any_action_saturated),
        }

    return {
        "n_samples": int(report.n_samples),
        "bellman": _bell(report.bellman),
        "actions": _act(report.actions),
        "policy_improvement": {
            "pct_improved": float(report.policy_improvement.pct_improved),
            "mean_q_reduction": float(report.policy_improvement.mean_q_reduction),
        },
        "stability": {
            "mean": float(report.stability.mean),
            "max": float(report.stability.max),
            "p95": float(report.stability.p95),
            "pct_sign_change": float(report.stability.pct_sign_change),
            "noise_std_used": float(report.stability.noise_std_used),
            "mean_norm": float(report.stability.mean_norm),
            "p95_norm": float(report.stability.p95_norm),
        },
        "cross_val": {
            "k_folds": int(report.cross_val.k_folds),
            "mean_train_rmse": float(report.cross_val.mean_train_rmse),
            "mean_val_rmse": float(report.cross_val.mean_val_rmse),
            "std_val_rmse": float(report.cross_val.std_val_rmse),
            "mean_overfitting_ratio": float(report.cross_val.mean_overfitting_ratio),
            "total_n_train": int(report.cross_val.total_n_train),
            "total_n_val": int(report.cross_val.total_n_val),
            "purge": report.cross_val.purge,
            "fold_rewards": bool(report.cross_val.fold_rewards),
            "folds": [
                {
                    "fold_index": int(f.fold_index),
                    "train_rmse": float(f.train_rmse),
                    "val_rmse": float(f.val_rmse),
                    "overfitting_ratio": float(f.overfitting_ratio) if np.isfinite(f.overfitting_ratio) else None,
                    "n_train": int(f.n_train),
                    "n_val": int(f.n_val),
                    "n_train_pairs": int(f.n_train_pairs),
                    "n_val_pairs": int(f.n_val_pairs),
                    "reassigned_pair_ids": list(f.reassigned_pair_ids),
                    "train_bellman": _bell(f.train_bellman),
                    "val_bellman": _bell(f.val_bellman),
                    "train_actions": _act(f.train_actions),
                    "val_actions": _act(f.val_actions),
                }
                for f in report.cross_val.folds
            ],
        },
    }


def save_report(report: EvaluationReport, out_dir):
    """
    Persist an EvaluationReport on disk in three complementary formats:

        <out_dir>/evaluation_report.json      — all scalar metrics, JSON
        <out_dir>/evaluation_report.txt       — human-readable summary
        <out_dir>/arrays/bellman_residuals.npy
        <out_dir>/arrays/sensitivities.npy
        <out_dir>/arrays/sensitivities_norm.npy
        <out_dir>/arrays/q_policy.npy
        <out_dir>/arrays/q_dataset.npy

    The .npy files make it easy to later draw histograms and scatter plots from multiple experiments without re-running anything.
    """
    import io
    import json
    from contextlib import redirect_stdout
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr_dir = out_dir / "arrays"
    arr_dir.mkdir(exist_ok=True)

    # JSON (scalar metrics only)
    with (out_dir / "evaluation_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_to_dict(report), f, indent=2)

    # Human-readable txt (same content as print_report)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(report)
    (out_dir / "evaluation_report.txt").write_text(buf.getvalue(), encoding="utf-8")

    # Side-car arrays (raw per-sample values, for later plotting)
    np.save(arr_dir / "bellman_residuals.npy", report.bellman.residuals)
    np.save(arr_dir / "sensitivities.npy", report.stability.sensitivities)
    np.save(arr_dir / "sensitivities_norm.npy", report.stability.sensitivities_norm)
    np.save(arr_dir / "q_policy.npy", report.policy_improvement.q_policy)
    np.save(arr_dir / "q_dataset.npy", report.policy_improvement.q_dataset)

    return {
        "json": str(out_dir / "evaluation_report.json"),
        "txt": str(out_dir / "evaluation_report.txt"),
        "arrays": str(arr_dir),
    }
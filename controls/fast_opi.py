"""
Fast, numerically equivalent OPI training and evaluation
=========================================================

Why this module exists
----------------------
The permutation test and the lambda_h sweep need several hundred OPI fits
(each with a 5-fold CV). With the original implementation one fit of the
all-to-all dataset (N = 83,200 pseudo-transitions) takes 27 min to 2 h 20 on
CECI, which makes these controls impractical.

Profiling the original code shows two bottlenecks, both of which can be removed
WITHOUT changing the optimisation problems being solved:

1) Policy evaluation (SDP).
   The objective is  ||D vec(S) - U||^2 + lambda_reg ||S||_F^2  with D of size
   (N, n^2). With a thin QR factorisation D = Q R (Q: N x n^2, R: n^2 x n^2),
       ||D s - U||^2 = ||R s - Q^T U||^2 + (||U||^2 - ||Q^T U||^2),
   and the second term does not depend on S. The SDP therefore keeps exactly the
   same minimiser but is handed n^2 = 100 residuals instead of N = 83,200.

2) Policy improvement (box-constrained QP, solved row by row with CVXPY).
   For each state the problem is  min_a a^T H a + 2 g^T a  s.t. lo <= a <= hi,
   with H = S_aa positive definite and only 3 action dimensions. Its exact
   solution can be found by enumerating the 3^3 = 27 active sets (each dimension
   free, at its lower bound or at its upper bound): for a strictly convex QP the
   optimum is the feasible candidate with the lowest objective. This is done for
   all N states at once with NumPy.

Everything else (normalisation, cost shift, initial zero policy, convergence
and early-stopping rules, choice of S_best, metric definitions, CV folds, purge
and per-fold rewards) is inherited unchanged from OPI_core.py / OPI_validation.py.

Use `check_equivalence.py` to verify on your machine that this module
reproduces the original implementation before trusting it.
"""
from __future__ import annotations

import copy
import itertools
import logging
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np

from OPI_core import Dataset, OfflinePolicyIteration, _solve
from OPI_validation import _dataset_from_indices, make_cv_splits

logger = logging.getLogger(__name__)

# All 27 active-set patterns for a 3-D box: 0 = free, 1 = at lower bound, 2 = at upper bound.
_PATTERNS_CACHE: Dict[int, list] = {}


def _patterns(n_act: int):
    if n_act not in _PATTERNS_CACHE:
        _PATTERNS_CACHE[n_act] = list(itertools.product((0, 1, 2), repeat=n_act))
    return _PATTERNS_CACHE[n_act]


def box_qp_batch(H: np.ndarray, G: np.ndarray, LO: np.ndarray, HI: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Exact solution of  min_a  a^T H a + 2 g_n^T a   s.t.  lo_n <= a <= hi_n
    for every row n, with H symmetric positive definite (shared by all rows).

    Parameters
    ----------
    H : (m, m) PD matrix (here S_aa).
    G : (N, m) linear terms g_n.
    LO, HI : (N, m) bounds.

    Returns
    -------
    A : (N, m) optimal actions, clipped to the bounds (as in OPI_core).
    """
    N, m = G.shape
    best_val = np.full(N, np.inf)
    best_A = np.zeros((N, m))
    for pat in _patterns(m):
        pat = np.asarray(pat)
        free = np.flatnonzero(pat == 0)
        fixed = np.flatnonzero(pat != 0)
        A = np.zeros((N, m))
        if fixed.size:
            A[:, fixed] = np.where(pat[fixed] == 1, LO[:, fixed], HI[:, fixed])
        if free.size:
            H_ff = H[np.ix_(free, free)]
            rhs = -G[:, free]
            if fixed.size:
                rhs = rhs - A[:, fixed] @ H[np.ix_(fixed, free)]
            A[:, free] = np.linalg.solve(H_ff, rhs.T).T
            feasible = np.all((A[:, free] >= LO[:, free] - tol) & (A[:, free] <= HI[:, free] + tol), axis=1)
        else:
            feasible = np.ones(N, dtype=bool)
        val = np.einsum("ni,ij,nj->n", A, H, A) + 2.0 * np.einsum("ni,ni->n", G, A)
        better = feasible & (val < best_val)
        best_val[better] = val[better]
        best_A[better] = A[better]
    return np.clip(best_A, LO, HI)


class _GreedyPolicy:
    """Marker object so that _batch_apply can evaluate the greedy policy for all rows at once."""

    def __init__(self, model: "FastOPI", S: np.ndarray):
        self.model, self.S = model, S

    def __call__(self, x_state, p_k):  # single-row fallback, same contract as OPI_core
        return self.model._greedy_actions_n(self.S, np.atleast_2d(x_state), np.atleast_2d(p_k))[0]


class _ZeroPolicy:
    def __init__(self, model: "FastOPI"):
        self.model = model

    def __call__(self, x_state, p_k):
        lo, hi = self.model._normalised_action_bounds(p_k)
        return np.clip(np.zeros(self.model.n_act), lo, hi)


class FastOPI(OfflinePolicyIteration):
    """Drop-in replacement for OfflinePolicyIteration with the two exact speed-ups described above."""

    # -- policies ---------------------------------------------------------
    def _make_greedy_policy(self, S):
        return _GreedyPolicy(self, S)

    def _make_zero_policy(self):
        return _ZeroPolicy(self)

    def _batch_apply(self, policy, Xn, P):
        if isinstance(policy, _GreedyPolicy):
            return self._greedy_actions_n(policy.S, Xn, P)
        if isinstance(policy, _ZeroPolicy):
            lo, hi = self._normalised_action_bounds(P)
            return np.clip(np.zeros_like(lo), lo, hi)
        return super()._batch_apply(policy, Xn, P)

    def _greedy_actions_n(self, S, Xn, P):
        """Greedy normalised actions for a batch of (already normalised) states."""
        s0 = 1 if self.cfg.use_bias else 0
        a0 = s0 + self.n_state
        S_aa = 0.5 * (S[a0:, a0:] + S[a0:, a0:].T)
        S_as = S[a0:, s0:a0]
        s_a0 = S[a0:, 0] if self.cfg.use_bias else np.zeros(self.n_act)
        G = Xn @ S_as.T + s_a0
        lo, hi = self._normalised_action_bounds(P)  # broadcasts over rows
        return box_qp_batch(S_aa, G, lo, hi)

    # -- policy evaluation --------------------------------------------------
    def _policy_evaluation(self, Xn, An, X1n, A1n_pi, U, S_fallback):
        n = self.n_joint
        gamma = self.cfg.gamma
        ones = np.ones((Xn.shape[0], 1))
        if self.cfg.use_bias:
            mu = np.hstack([ones, Xn, An])
            mu1 = np.hstack([ones, X1n, A1n_pi])
        else:
            mu = np.hstack([Xn, An])
            mu1 = np.hstack([X1n, A1n_pi])
        D = (np.einsum("ni,nj->nij", mu, mu) - gamma * np.einsum("ni,nj->nij", mu1, mu1)).reshape(-1, n * n)

        # Thin QR: same minimiser, n^2 residuals instead of N.
        Q, R = np.linalg.qr(D, mode="reduced")
        b = Q.T @ U

        S_var = cp.Variable((n, n), symmetric=True)
        S_flat = cp.reshape(S_var, (n * n,), order="C")
        obj = cp.Minimize(cp.sum_squares(R @ S_flat - b) + cp.sum_squares(np.sqrt(self.cfg.lambda_reg) * S_var))
        a_idx = (1 if self.cfg.use_bias else 0) + self.n_state
        cons = [S_var[a_idx:, a_idx:] >> self.cfg.SDP_eps * np.eye(self.n_act)]
        prob = cp.Problem(obj, cons)
        status = _solve(prob, "policy_evaluation_fast")
        if S_var.value is None:
            logger.warning("Policy evaluation failed; reusing previous S.")
            return S_fallback.copy(), status
        S = 0.5 * (S_var.value + S_var.value.T)
        return S, status

    # -- batch prediction in physical units (used by the metrics) ---------
    def predict_batch(self, x_feat: np.ndarray, p: np.ndarray):
        Xn = self._build_state_matrix(np.atleast_2d(x_feat), np.atleast_2d(p))
        a_n = self._greedy_actions_n(self.S, Xn, np.atleast_2d(p))
        a = self.action_norm.denormalize(a_n)
        p1 = np.clip(p + a, self.cfg.prosthesis.p_min, self.cfg.prosthesis.p_max)
        return a, p1


# ============================================================================
# Metrics: same definitions as OPI_validation.py, vectorised.
# ============================================================================

def _q_values(model: FastOPI, Xn, An):
    ones = np.ones((Xn.shape[0], 1))
    mu = np.hstack([ones, Xn, An]) if model.cfg.use_bias else np.hstack([Xn, An])
    return np.einsum("ni,ij,nj->n", mu, model.S, mu)


def bellman_rmse(model: FastOPI, dataset: Dataset) -> float:
    """OPIEvaluator.bellman_error(...).rmse"""
    x_feat, a, x1_feat, r, p, p1 = dataset.arrays(model.cfg)
    Xn = model._build_state_matrix(x_feat, p)
    X1n = model._build_state_matrix(x1_feat, p1)
    An = model.action_norm.normalize(a)
    U = model.cost_shift_ - r
    a1, _ = model.predict_batch(x1_feat, p1)
    A1n = model.action_norm.normalize(a1)
    res = _q_values(model, Xn, An) - model.cfg.gamma * _q_values(model, X1n, A1n) - U
    return float(np.sqrt(np.mean(res ** 2)))


def saturation(model: FastOPI, dataset: Dataset) -> Dict[str, float]:
    """Saturation percentages of OPIEvaluator.action_distribution(...)."""
    x_feat, a, _, _, p, _ = dataset.arrays(model.cfg)
    pr = model.cfg.prosthesis
    a_pred, _ = model.predict_batch(x_feat, p)
    a_pred_n = model.action_norm.normalize(a_pred)
    lo_n = (pr.p_min - p) / pr.p_half_range
    hi_n = (pr.p_max - p) / pr.p_half_range
    hits = ((a_pred_n - lo_n) < 1e-3) | ((hi_n - a_pred_n) < 1e-3)
    per_dim = hits.mean(axis=0) * 100.0
    return {
        "sat_any_pct": float(np.any(hits, axis=1).mean() * 100.0),
        "sat_d_plant_pct": float(per_dim[0]),
        "sat_theta_trig_pct": float(per_dim[1]),
        "sat_tau_lim_pct": float(per_dim[2]),
    }


def stability(model: FastOPI, dataset: Dataset, noise_fraction: float = 0.02, n_perturbations: int = 5, seed: int = 0) -> Dict[str, float]:
    """
    OPIEvaluator.stability_test(...). The random draws are generated in the
    same order as the original double loop (sample n, then perturbation k),
    so the same seed gives the same perturbations.
    """
    rng = np.random.default_rng(seed)
    x_feat, _, _, _, p, _ = dataset.arrays(model.cfg)
    N, F = x_feat.shape
    noise_std = noise_fraction * np.maximum(x_feat.std(axis=0), 1e-8)
    delta = rng.normal(0.0, np.broadcast_to(noise_std, (N, n_perturbations, F)))
    a_base, _ = model.predict_batch(x_feat, p)
    sens = np.zeros((N, n_perturbations))
    for k in range(n_perturbations):
        a_pert, _ = model.predict_batch(x_feat + delta[:, k, :], p)
        sens[:, k] = np.linalg.norm(a_pert - a_base, axis=1) / np.linalg.norm(delta[:, k, :], axis=1)
    s = sens.mean(axis=1)
    out = {"sens_mean": float(s.mean()), "sens_p95": float(np.percentile(s, 95)), "sens_max": float(s.max())}

    # Unit-free variant (same as OPI_validation.stability_test, "_norm" values): the action
    # change is divided by the half-range of each parameter. Same perturbations as above.
    half = model.cfg.prosthesis.p_half_range
    sens_n = np.zeros((N, n_perturbations))
    share = np.zeros(3)  # mean share of each parameter in ||Delta a||^2 (physical units)
    for k in range(n_perturbations):
        a_pert, _ = model.predict_batch(x_feat + delta[:, k, :], p)
        da = a_pert - a_base
        dnorm = np.linalg.norm(delta[:, k, :], axis=1)
        sens_n[:, k] = np.linalg.norm(da / half, axis=1) / dnorm
        sq = da ** 2
        tot = sq.sum(axis=1)
        moving = tot > 0
        if moving.any():
            share += (sq[moving] / tot[moving, None]).mean(axis=0) / n_perturbations
    sn = sens_n.mean(axis=1)
    out.update({"sens_norm_mean": float(sn.mean()), "sens_norm_median": float(np.median(sn)),
                "sens_norm_p95": float(np.percentile(sn, 95)), "sens_norm_max": float(sn.max()),
                "sens_median": float(np.median(s)),
                "sens_share_d_plant": float(share[0]), "sens_share_theta_trig": float(share[1]),
                "sens_share_tau_lim": float(share[2])})
    return out


def grouped_cv_overfit(cfg, dataset: Dataset, pair_ids: Sequence[str], k_folds: int = 5, *,
                       pair_blocks: Optional[Mapping[str, Tuple[int, int]]] = None, purge: str = "none",
                       fold_reward_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Dict[str, float]:
    """Same folds, purge and per-fold rewards as OPI_validation.cross_validate."""
    rows = list(dataset)
    pair_ids = np.asarray(pair_ids)
    ratios, val_rmse, train_rmse, n_tr, n_va = [], [], [], [], []
    for tr_idx, va_idx, _ in make_cv_splits(pair_ids, k_folds, pair_blocks=pair_blocks, purge=purge):
        r = fold_reward_fn(tr_idx) if fold_reward_fn is not None else None
        tr, va = _dataset_from_indices(rows, tr_idx, r), _dataset_from_indices(rows, va_idx, r)
        m = FastOPI(copy.deepcopy(cfg)).fit(tr)
        t, v = bellman_rmse(m, tr), bellman_rmse(m, va)
        train_rmse.append(t)
        val_rmse.append(v)
        ratios.append(v / t if t > 1e-12 else float("nan"))
        n_tr.append(len(np.unique(pair_ids[tr_idx])))
        n_va.append(len(np.unique(pair_ids[va_idx])))
    return {
        "cv_overfit_ratio": float(np.mean(ratios)),
        "cv_val_rmse": float(np.mean(val_rmse)),
        "cv_train_rmse": float(np.mean(train_rmse)),
        "cv_fold_ratios": [float(x) for x in ratios],
        "cv_n_train_pairs": n_tr,
        "cv_n_val_pairs": n_va,
    }


def evaluate_all(cfg, dataset: Dataset, pair_ids: Sequence[str], *, k_folds: int = 5, seed: int = 0,
                 pair_blocks: Optional[Mapping[str, Tuple[int, int]]] = None, purge: str = "none",
                 fold_reward_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Dict[str, float]:
    """Fit on the full dataset, then compute every scale-free diagnostic used in the paper."""
    model = FastOPI(copy.deepcopy(cfg)).fit(dataset)
    out = {"best_iter": int(model.best_iter), "n_iter": len(model.history["frobenius"]),
           "bellman_rmse": bellman_rmse(model, dataset)}
    out.update(saturation(model, dataset))
    out.update(stability(model, dataset, seed=seed))
    out.update(grouped_cv_overfit(cfg, dataset, pair_ids, k_folds=k_folds, pair_blocks=pair_blocks,
                                  purge=purge, fold_reward_fn=fold_reward_fn))
    # Learned matrix of the full-data fit (10 x 10), kept so that the policy of any
    # replicate can be recomputed afterwards (what the ratings change in the actions).
    out["S"] = np.asarray(model.S).tolist()
    return out

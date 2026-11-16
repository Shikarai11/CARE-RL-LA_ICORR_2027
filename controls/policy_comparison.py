"""
Do the real ratings change the recommended updates? (Section VI-C of the paper)
================================================================================

The policies are recomputed from the S matrices saved by permutation_test.py
(real ratings and 109 shuffled ratings, lambda_h = 0.5) and by sweep_lambda.py
(objective-only policy, lambda_h = 0), and compared on the SAME states: the 2080
distinct states of the dataset. Targets p_k + pi(s_k) are in half-range units
from the middle of the admissible range. These are fits on the 52 transitions,
so the comparison does not depend on the CV protocol.

  0) eigenvalues of S_aa of the real-rating policy;
  1) mean per-state distance of the real-rating policy to the shuffled ones,
     against the same distance for each shuffled policy to the others;
  2) share of states where the real-rating target leaves the 95 % band of the
     shuffled targets (leave-one-out reference for the shuffled policies);
  3) shift from the objective-only policy, real vs shuffled, and the correlation
     across states between the real shift and the mean shuffled shift.

Example (from the repository root)
------------------------------
    python controls/policy_comparison.py --pickle data/all_to_all_opi_dataset.pkl \
        --csv data/all_to_all_transitions.csv --score-csv data/ExpS1_Full_scores.csv \
        --perm results/controls/permutation.jsonl --sweep results/controls/sweep_ra1.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import warnings

import numpy as np

import controls_common as C
import fast_opi as F

NAMES = ["d_dors", "theta_trig", "tau_lim"]


def _load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _p_high(null, obs):
    return (1 + np.sum(null >= obs)) / (1 + len(null))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--score-csv", required=True)
    ap.add_argument("--perm", required=True, help="permutation_test.py output (lambda_h = 0.5)")
    ap.add_argument("--sweep", required=True, help="sweep_lambda.py output containing lambda_h = 0")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    inp = C.load_inputs(args.pickle, args.csv, args.score_csv)
    cfg = C.make_cfg(lambda_human=0.5)
    x, _, x1, _, p, _ = C.build_dataset(inp, C.rewards_for(inp, cfg)).arrays(cfg)
    model = F.FastOPI(copy.deepcopy(cfg))
    model.feature_norm.fit(np.vstack([x, x1]))  # same normalisation as in OfflinePolicyIteration.fit
    _, first = np.unique(np.hstack([x, p]).round(10), axis=0, return_index=True)
    states = np.sort(first)  # each state (cycle of block k, p_k) once
    xs, ps = x[states], p[states]
    pr = cfg.prosthesis
    center, half = (pr.p_min + pr.p_max) / 2, pr.p_half_range

    def targets(S):
        model.S = np.asarray(S)
        _, p_next = model.predict_batch(xs, ps)
        return (p_next - center) / half

    perm = _load(args.perm)
    real = [r for r in perm if r["real_ratings"]][0]
    T_real = targets(real["S"])
    T_null = np.array([targets(r["S"]) for r in perm if not r["real_ratings"]])  # (K, n_states, 3)
    T_obj = targets([r for r in _load(args.sweep) if r["lambda_human"] == 0.0][0]["S"])
    K = len(T_null)
    print(f"{len(states)} distinct states, {K} shuffled-rating policies")

    S_aa = np.asarray(real["S"])[-3:, -3:]
    print("\n0) Eigenvalues of S_aa (real ratings):", np.round(np.linalg.eigvalsh(0.5 * (S_aa + S_aa.T)), 3))

    d_real = np.abs(T_null - T_real).mean(axis=1)
    d_null = np.array([np.abs(np.delete(T_null, k, 0) - T_null[k]).mean(axis=1).mean(0) for k in range(K)])
    print("\n1) Mean per-state distance to the shuffled-rating policies (half-range)")
    for j, n in enumerate(NAMES):
        print(f"  {n:10s} real {d_real[:, j].mean():.3f} | shuffled to the others: median {np.median(d_null[:, j]):.3f}"
              f" [{np.percentile(d_null[:, j], 2.5):.3f}, {np.percentile(d_null[:, j], 97.5):.3f}]"
              f"  p = {_p_high(d_null[:, j], d_real[:, j].mean()):.3f}")
    o, nn = d_real.mean(), d_null.mean(1)
    print(f"  all three  real {o:.3f} | median {np.median(nn):.3f} [{np.percentile(nn, 2.5):.3f}, "
          f"{np.percentile(nn, 97.5):.3f}]  p = {_p_high(nn, o):.3f}")

    lo, hi = np.percentile(T_null, 2.5, axis=0), np.percentile(T_null, 97.5, axis=0)
    out_real = ((T_real < lo - 1e-9) | (T_real > hi + 1e-9)).mean(0)
    out_null = []
    for k in range(K):
        rest = np.delete(T_null, k, 0)
        l2, h2 = np.percentile(rest, 2.5, axis=0), np.percentile(rest, 97.5, axis=0)
        out_null.append(((T_null[k] < l2 - 1e-9) | (T_null[k] > h2 + 1e-9)).mean(0))
    out_null = np.array(out_null)
    print("\n2) Share of states where the target leaves the 95 % band of the shuffled policies")
    for j, n in enumerate(NAMES):
        print(f"  {n:10s} real {100 * out_real[j]:.1f} % | shuffled median {100 * np.median(out_null[:, j]):.1f} %"
              f" [{100 * np.percentile(out_null[:, j], 2.5):.1f}, {100 * np.percentile(out_null[:, j], 97.5):.1f}]"
              f"  p = {_p_high(out_null[:, j], out_real[j]):.3f}")

    shift_real, shift_null = T_real - T_obj, T_null - T_obj
    print("\n3) Shift from the objective-only policy (half-range), mean over states")
    for j, n in enumerate(NAMES):
        v = shift_null[:, :, j].mean(1)
        corr = np.corrcoef(shift_real[:, j], shift_null[:, :, j].mean(0))[0, 1]
        print(f"  {n:10s} real {shift_real[:, j].mean():+.3f} | shuffled mean {v.mean():+.3f}"
              f" [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]  p_high = {_p_high(v, shift_real[:, j].mean()):.3f}"
              f" | corr. across states with the mean shuffled shift {corr:+.2f}")


if __name__ == "__main__":
    main()

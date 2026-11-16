""" 
Generic Offline Policy Iteration (OPI) algorithm

Author : Raphaël VASSART
================================================================
Contains everything that comes directly from the state of the art:
Li et al., "Offline Policy Iteration Based Reinforcement Learning Controller for Online Robotic Knee Prosthesis Parameter Tuning", ICRA 2019, pp. 2831-2837.

Algorithm Main Loop
---------------------
  Q-function approx:  Q̂(x, a) = [x; a]ᵀ S [x; a],  S ∈ ℝⁿˣⁿ, S ≻ 0
 
  π⁽⁰⁾ = 0  (zero-action warm start)
  
  For i = 0, 1, …:
    1) Policy Evaluation  (SDP, convex)
       S⁽ⁱ⁾ ← argmin_S  Σₙ [ μₙᵀ S μₙ - γ (μ′ₙ)ᵀ S μ′ₙ - Uₙ ]²
                        + λ_reg · ‖S‖²_F      (Tikhonov regularisation)
              s.t.   S_aa ⪰ ε · I
       
    2) Policy Improvement  (QP, convex)
       π⁽ⁱ⁺¹⁾(x) = argmin_{a⁻≤a≤a⁺}  Q̂⁽ⁱ⁾(x, a)
       
    3) Convergence:  ‖S⁽ⁱ⁺¹⁾ - S⁽ⁱ⁾‖_F < tol
    
"""

from __future__ import annotations
 
import logging
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
 
import cvxpy as cp
import numpy as np
 
from elsa_configOPI import OPIConfig
 
logger = logging.getLogger(__name__)


# ============================================================================
# |                                                                          |
# |  1) NORMALISERS                                                          |
# |                                                                          |
# ============================================================================

class MinMaxNormalizer:
    """
    Min-max normaliser that maps each feature to [-1, 1].
    From Li et al. 2019: "all samples were normalized into the range between -1 and 1 to avoid ill-conditioning issues during application of convex optimization."
    
    For each dimension j:
        x_n[j] = 2 * (x[j] - lo[j]) / (hi[j] - lo[j]) - 1

    An epsilon is used to prevent division by zero when a feature is constant.

    The biomechanical features are fitted on `vstack([X, X1])`, i.e. on current and next states together. 
    This ensures both sides of the Bellman equation live in the same normalised coordinate system.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self._lo: Optional[np.ndarray] = None
        self._hi: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray):
        """Estimate per-feature min and max values from the supplied matrix X."""
        X = np.asarray(X, dtype=float)
        self._lo = X.min(axis=0)
        self._hi = X.max(axis=0)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray):
        """Map raw values into [-1, 1] using the fitted min-max statistics."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        rng = np.maximum(self._hi - self._lo, self.eps)
        return 2.0 * (X - self._lo) / rng - 1.0
    
    
class ActionNormalizer:
    """
    Normaliser for INCREMENTAL actions.
    
    Actions are divided by the half-range of the corresponding parameter:
        a_n = a / ((p_max - p_min) / 2)

    Why this scaling?
    ---------------------
    The three action dimensions may have very different physical magnitudes (for example mm vs ° vs Nm). 
    Without normalisation, the learned matrix S can become badly conditioned because some coordinates are numerically much larger than others. 
    We use the half-range to keep values centered around [-1, 1].
    """

    def __init__(self, cfg: OPIConfig):
        self.scale = np.maximum(cfg.prosthesis.p_half_range, 1e-8)

    def normalize(self, a: np.ndarray):
        """Map physical incremental actions into the normalised action space."""
        a = np.asarray(a, dtype=float)
        return a / self.scale

    def denormalize(self, an: np.ndarray):
        """Map normalised actions back into physical units."""
        an = np.asarray(an, dtype=float)
        return an * self.scale

    def normalize_bounds(self, a_lo: np.ndarray, a_hi: np.ndarray):
        """
        Convert physical lower/upper action bounds into normalised bounds.
        Returns lower and upper bounds in the normalised action space.
        """
        return self.normalize(a_lo), self.normalize(a_hi)
    
    

# ============================================================================
# |                                                                          |
# |  2) DATASET                                                              |
# |                                                                          |
# ============================================================================

@dataclass
class Transition: 
    """
    One offline (s, a, s', r) tuple for the OPI dataset.  
 
    x_feat : biomechanical block-level features during block k : xₖ = [f1ₖ-f1ʳᵉᶠ, f2ₖ-f2ʳᵉᶠ, ...]
    a : incremental action applied at the end of block k, leading to block k+1.
    x1_feat : biomechanical features during block k+1
    r : shaped reward associated with the transition k -> k+1
    p : absolute parameters active during block k (needed for action bounds)
    p1 : absolute parameters active during block k+1, we can provide it directly or can be reconstructed as clip(p + a).
    """
    x_feat: np.ndarray
    a: np.ndarray
    x1_feat: np.ndarray
    r: float
    p: np.ndarray
    p1: Optional[np.ndarray] = None
    
    
    
class Dataset:
    """Ordered collection of offline transitions."""

    def __init__(self):
        self._data: List[Transition] = []

    def add(self, t: Transition):
        self._data.append(t)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def arrays(self, cfg: OPIConfig):
        """
        Convert the dataset into dense NumPy arrays.
        
        x_feat: Matrix of current biomechanical features, shape (N, n_features).
        a: Matrix of incremental actions, shape (N, n_actions).
        x1_feat: Matrix of next biomechanical features, shape (N, n_features).
        r: Reward vector, shape (N,).
        p: Matrix of current absolute parameters, shape (N, n_actions).
        p1: Matrix of next absolute parameters, shape (N, n_actions).
        """
        x_feat = np.array([t.x_feat for t in self._data], dtype=float)
        a = np.array([t.a for t in self._data], dtype=float)
        x1_feat = np.array([t.x1_feat for t in self._data], dtype=float)
        r = np.array([t.r for t in self._data], dtype=float)
        p = np.array([t.p for t in self._data], dtype=float)

        p1_list = []
        for t in self._data:
            if t.p1 is not None:
                p1_list.append(np.asarray(t.p1, dtype=float))
            else:
                # If a transition does not explicitly provide `p1`, it is reconstructed using the action and clipped to the physical parameter bounds.
                p1_list.append(np.clip(t.p + t.a, cfg.prosthesis.p_min, cfg.prosthesis.p_max))
        p1 = np.array(p1_list, dtype=float)

        return x_feat, a, x1_feat, r, p, p1
    


# ============================================================================
# |                                                                          |
# |  3) CVXPY SOLVER HELPER                                                  |
# |                                                                          |
# ============================================================================

def _solve(prob: cp.Problem, tag: str):
    """
    Try several CVXPY solvers in sequence and return the final status string.
    cf. https://www.cvxpy.org/api_reference/cvxpy.html

    Parameters
    ------------
    prob: CVXPY problem to solve.
    tag: Small label used in logs to identify the caller.

    Returns the final CVXPY status string.
    """
    last_status = "unknown" # variable to memorize the last CVXPYstatus
    for solver in [cp.CLARABEL, cp.SCS]:
        try:
            prob.solve(solver=solver, verbose=False)
            last_status = str(prob.status)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                # Only `OPTIMAL` and `OPTIMAL_INACCURATE` are treated as acceptable solved states. 
                # Everything else is considered a failure for the purposes of this controller.
                return last_status
        except Exception as exc: 
            logger.info("[%s] solver %s failed: %s", tag, solver, exc)
    logger.warning("[%s] all solvers failed or returned non-optimal status (%s)", tag, last_status)
    return last_status



# ============================================================================
# |                                                                          |
# |  4) OFFLINE POLICY ITERATION (core algorithm)                            |
# |                                                                          |
# ============================================================================

class OfflinePolicyIteration:
    """
    Approximate Offline Policy Iteration for prosthesis parameter tuning.
    
    Training alternates between:
        - policy evaluation:
              estimate a PSD matrix S such that
                  Q̂⁽ⁱ⁾(s, a) ≈ μ(s, a)ᵀ S⁽ⁱ⁾ μ(s, a)
              satisfies the Bellman relation on the offline dataset (plus a Tikhonov term on ‖S‖²_F);

        - policy improvement:
              derive the greedy action from the current S by solving a bounded quadratic program, strictly convex thanks to S_aa ≻ 0.
    """
    
    def __init__(self, cfg: OPIConfig):
        self.cfg = cfg
        self.n_feat = cfg.n_features
        self.n_state = cfg.n_state_total # n_features (+ n_params = n_actions)
        self.n_act = cfg.n_actions
        self.n_joint = cfg.n_joint
        
        self.S: Optional[np.ndarray] = None # learned PSD matrix defining the quadratic Q approximation
        self.cost_shift_: Optional[float] = None # Cost shift learned from the reward distribution during fit()
        
        
        # Best-S tracker 
        self.S_best: Optional[np.ndarray] = None
        self.best_frob_: float = float('inf')
        self.best_iter: int = -1
        self.strict_convergence_: bool = False
        
        # Normalisation 
        self.feature_norm = MinMaxNormalizer()
        self.param_norm = cfg.build_param_normalizer() # ELSA-aware, injected via config
        self.action_norm = ActionNormalizer(cfg)
        
        self.history = {
            "frobenius": [],
            "sdp_status": [],
            "min_eig_Saa": [],
        } # Used to compute Frobenius norm and check the convergence status
        
    
    # -- Public API -----------------------------------------------------------
    
    def fit(self, dataset: Dataset):
        """
        Train the OPI model from an offline dataset of (x, a, x', r, p) tuples.
 
        Steps:
        -------
        1. Normalise states to [-1,1] and actions by half-range.
        2. Convert rewards to costs: U = -reward (+ cost_shift).
        3. Initialise π⁽⁰⁾ = 0 (zero policy - safe, always feasible).
        4. Iterate policy evaluation -> policy improvement until convergence.
        5. Store the converged S matrix.
        """
        x_feat, a, x1_feat, r, p, p1 = dataset.arrays(self.cfg) # extraction 
        N = len(dataset)
        
        if N == 0:
            raise ValueError("Dataset is empty")
        
        # A symmetric n x n matrix has n(n+1)/2 unique parameters.
        # This warning flags datasets likely too small for a stable S estimate. (which will be my case for this master's thesis unfortunately)
        n_sym_params = self.n_joint * (self.n_joint + 1) // 2
        if N < n_sym_params:
            warnings.warn(f"Dataset has {N} transitions but S has {n_sym_params} unique symmetric parameters. Expect policy evaluation to be poorly conditioned unless you collect more data.")
            
        logger.info("OPI.fit: N=%d samples,  n=%d joint-space dim", N, self.n_joint) # For debug purpose
        
        # --- Step 1: Normalize ---
        # Fit feature normaliser on BOTH current and next states.
        self.feature_norm.fit(np.vstack([x_feat, x1_feat]))
        
        # Build full state matrices.
        # If include_params_in_state=True, the state is:
        #   s_k = [x_feat_k_normalised ; p_k_normalised]
        Xn = self._build_state_matrix(x_feat, p)
        X1n = self._build_state_matrix(x1_feat, p1)
        
        # Action normalisation.
        An = self.action_norm.normalize(a)   
        
        # --- Step 2: Rewards -> Costs ---
        # Reward -> non-negative cost via constant shift.
        U = self._reward_to_cost(r)
        
        # --- Step 3: Initialise ---
        policy = self._make_zero_policy()
        S_prev = np.eye(self.n_joint) # init S
        
        # --- Step 4: Main iteration: Pol Eval -> Pol Imp -> Pol Eval -> ... ---
        for i in range(self.cfg.n_iterations):
            # 1) Compute π⁽ⁱ⁾(x'ₙ) for every next state in the dataset to compute the "future part" of the Bellman equation
            A1n_pi = self._batch_apply(policy, X1n, p1)
            
            # 2) Policy Evaluation: solve SDP for S⁽ⁱ⁾ and estimate a new quadratic Q approximation (new S).
            S_new, status = self._policy_evaluation(Xn, An, X1n, A1n_pi, U, S_prev)
            
            # 3) Policy Improvement: define π⁽ⁱ⁺¹⁾ = greedy according to S_new
            policy = self._make_greedy_policy(S_new)
            
            # 4) Check Convergence status
            frob = float(np.linalg.norm(S_new - S_prev, ord="fro"))
            # Extract S_aa from the converged S to monitor the PSD constraint.
            a_idx = (1 if self.cfg.use_bias else 0) + self.n_state
            S_aa_new = S_new[a_idx:, a_idx:]
            min_eig_Saa = float(np.min(np.linalg.eigvalsh(S_aa_new)))
            self.history["frobenius"].append(frob)
            self.history["sdp_status"].append(status)
            self.history["min_eig_Saa"].append(min_eig_Saa)
            logger.info(
                "iter=%d  status=%s  ||ΔS||_F=%.3e  λ_min(S)=%.3e",
                i + 1, status, frob, self.history["min_eig_Saa"][-1],
            ) # debug
            
            # Update the best-S tracker
            if frob < self.best_frob_:
                self.S_best = S_new.copy()
                self.best_frob_ = frob
                self.best_iter = i+1
                
            
            # 5a) Strict convergence check on S.
            S_prev = S_new
            if frob < self.cfg.tol:
                logger.info("OPI converged at iteration %d.", i + 1)
                self.strict_convergence_ = True
                break
            
            # 5b) Early stopping on cycles or plateaus in the Frobenius norm history.
            if (self.cfg.cycle_window > 0 and (i+1) >= 2 * self.cfg.cycle_window):
                window = self.history["frobenius"][-self.cfg.cycle_window:]
                amp = max(window) - min(window)
                med = float(np.median(window))
                rel = amp / max(med, 1e-12) # avoid division by zero
                if amp < self.cfg.cycle_tol_abs and rel < self.cfg.cycle_tol_rel:
                    logger.info(
                        "OPI plateau detected at iter %d: ||ΔS||_F stagnates "
                        "at %.3e (absolute span %.2e, relative span %.2e) over "
                        "the last %d iterations. Stopping early.",
                        i + 1, med, amp, rel, self.cfg.cycle_window,
                    )
                    break
                
            # 5c) Early stopping on limit cycles with non-negligible amplitude.
            cycle_info = self._detect_limit_cycle(i)
            if cycle_info is not None:
                period, span = cycle_info
                logger.info(
                    "OPI limit cycle detected at iter %d: ||ΔS||_F oscillates "
                    "with period %d (peak-to-peak %.3e over last %d iters). "
                    "Stopping early.",
                    i + 1, period, span, 3 * period,
                )
                break
            
            # 5d) Early stopping on non-periodic stagnation of progress.
            stag_info = self._detect_no_progress(i)
            if stag_info is not None:
                recent_min, prior_min = stag_info
                delta_pct = (recent_min / prior_min - 1.0) * 100.0
                logger.info(
                    "OPI no-progress detected at iter %d: best ||ΔS||_F "
                    "over last %d iters is %.3e vs %.3e previously "
                    "(%+.2f %%). Stopping early.",
                    i + 1, self.cfg.stagnation_window,
                    recent_min, prior_min, delta_pct,
                )
                break
            
        else:
            logger.warning("OPI did not converge within %d iterations.", self.cfg.n_iterations)
 
        # --- Step 5: Store the converged S matrix or the best S if early stopping was triggered by a cycle or stagnation ---
        if self.strict_convergence_ or not self.cfg.return_best_S_on_early_stop:
            self.S = S_prev
        else:
            self.S = (self.S_best.copy() if self.S_best is not None else S_prev)
            if self.S_best is not None:
                last_frob = (self.history["frobenius"][-1]
                             if self.history["frobenius"] else float("nan"))
                logger.info(
                    "Using S_best from iter %d (||ΔS||_F=%.3e) as final "
                    "model instead of last iterate (||ΔS||_F=%.3e).",
                    self.best_iter, self.best_frob_, last_frob,
                )
        return self
    
    
    def predict(self, x_feat_k: np.ndarray, p_k: np.ndarray):
        """
        Predict the greedy incremental action for one state.
        
        Returns
        ---------
            - a_k : Greedy incremental action in physical units.
            - p_k1 : new absolute parameters = clip(p_k + a_k, p_min, p_max)
        """
        if self.S is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        x_state = self._build_single_state(x_feat_k, p_k)
        a_lo_n, a_hi_n = self._normalised_action_bounds(p_k)
        a_n = self._policy_improvement_qp(x_state, self.S, a_lo_n, a_hi_n)
        a_k = self.action_norm.denormalize(a_n)
        p_k1 = np.clip(p_k + a_k, self.cfg.prosthesis.p_min, self.cfg.prosthesis.p_max)
        return a_k, p_k1
    
    
    # -- Core OPI steps -----------------------------------------------------------
    
    def _policy_evaluation(self, Xn: np.ndarray, An: np.ndarray, X1n: np.ndarray, A1n_pi: np.ndarray, U: np.ndarray, S_fallback: np.ndarray):
        """
        Solve the policy evaluation step as a least-squares SDP.
        
        Bellman approximation
        ------------------------------
        For each transition n, the code builds:
            - μ_n = μ(s_n, a_n)
            - μ'_n = μ(s'_n, π(s'_n))
        and enforces in least-squares sense: μ_n^T S μ_n - γ μ'_n^T S μ'_n ≈ U_n

        Because S is symmetric, this can be written as a linear expression in S
        for each sample. The code then solves a convex optimisation problem:
            minimise   Σ_n residual_n²  +  λ_reg · ‖S‖²_F
            subject to S_aa ⪰ SDP_eps * I
            
        Implemented via CVXPY pakage (cf. https://www.cvxpy.org/api_reference/cvxpy.html):
            residualₙ = trace(Δₙ S)  - Uₙ  = cp.sum(cp.multiply(Δₙ, S_var)) - Uₙ
        (valid because Δₙ and S are both symmetric)
        
        If the solver fails, the previous S_fallback is returned unchanged so that the outer policy-iteration loop can keep making progress.
        
        Params
        --------
        Xn, An: Current states and observed actions from the dataset.
        X1n, A1n_pi: Next states and actions given by the current policy at those states.
        U: Instantaneous non-negative costs.
        S_fallback: Previous S used as a safe fallback if the solver fails.
        """
        
        N = Xn.shape[0]
        n = self.n_joint # size of joint space
        gamma = self.cfg.gamma
        
        # Joint vectors: 
        mu = np.array([self._joint_vector(Xn[i], An[i]) for i in range(N)]) # μₙ
        mu1 = np.array([self._joint_vector(X1n[i], A1n_pi[i]) for i in range(N)]) # μ'ₙ
        
        # For each sample n, delta[n] is the matrix such that:
        #   Δₙ = μₙ μₙᵀ - γ μ'ₙ μ'ₙᵀ (vectorised with einsum)
        delta = (
            np.einsum("ni,nj->nij", mu, mu) 
            - gamma * np.einsum("ni,nj->nij", mu1, mu1) 
        ) # Use of einsum to avoid the iteration over N (cf. doc : https://stackoverflow.com/questions/26089893/understanding-numpys-einsum)
        
        
        # Build CVXPY problem
        # S⁽ⁱ⁾ ← argmin_{S_aa≻0}  Σₙ [ μₙᵀ S μₙ - γ (μ′ₙ)ᵀ S μ′ₙ - Uₙ ]²
        S_var = cp.Variable((n, n), symmetric=True) # the unknown variable to optimise
        
        # -- Vectorised residual construction (critical for scalability) -----
        # Each residual is an affine function of vec(S):
        #       rₙ = trace(Δₙ · S) - Uₙ = vec(Δₙ)ᵀ · vec(S) - Uₙ
        # Stacking all N residuals in one matrix-vector product gives:
        #       r = D · vec(S) - U  with D = [vec(Δ₁); vec(Δ₂); ...; vec(Δ_N)]
        # This replaces the Python-level list comprehension over N CVXPY expressions (which triggers the "too many subexpressions" warning and makes problem compilation O(N²)) by a single linear expression, making compilation O(N) and typically 50–200× faster on N ≥ 10⁴.
        # The optimisation problem is mathematically unchanged.
        D = delta.reshape(N, n * n) # (N, n²)
        S_flat = cp.reshape(S_var, (n * n,), order='C') # vec(S), row-major
        residuals_vec = D @ S_flat - U # (N,) affine in S_var
        
        # Objective: least-squares Bellman error + Tikhonov regularisation.
        # The regularisation term λ_reg ‖S‖²_F = λ_reg · cp.sum_squares(S_var) penalises large entries in S and prevents overfitting when N is small relative to the n(n+1)/2 free parameters of a symmetric n×n matrix.
        data_fit = cp.sum_squares(residuals_vec)
        reg_term = cp.sum_squares(np.sqrt(self.cfg.lambda_reg) * S_var)
        obj = cp.Minimize(data_fit + reg_term)
        
        # Constraint: S_aa ≻ 0 (action-action block only)
        # Index of the first action dimension inside the joint vector μ.
        #   μ = [1 (bias?); x (state); a (action)]
        a_idx   = (1 if self.cfg.use_bias else 0) + self.n_state
        S_aa_var = S_var[a_idx:, a_idx:]   # (n_act × n_act) sub-block
        constraints = [S_aa_var >> self.cfg.SDP_eps * np.eye(self.n_act)] # under the constraint S_aa⪰εI (or S_aa​−εI⪰0)
        
        prob = cp.Problem(obj, constraints) # def of the problem
        status = _solve(prob, "policy_evaluation")
        
        # If the solver fails, keep the previous valid matrix rather than silently resetting to an arbitrary identity matrix.
        if S_var.value is None:
            logger.warning("Policy evaluation failed; reusing previous S.")
            return S_fallback.copy(), status

        # Symmetrise explicitly to avoid tiny numerical asymmetries produced by CVXPY due to floating-point precision.
        S = 0.5 * (S_var.value + S_var.value.T)
        return S, status
    
    
    
    def _policy_improvement_qp(self, x_state: np.ndarray, S: np.ndarray, a_lo_n: np.ndarray, a_hi_n: np.ndarray):
        """ 
        Solve the policy improvement QP for one state:
            min_a  Q̂(x, a) = aᵀ S_aa a  +  2 (S_ax x)ᵀ a  +  xᵀ S_xx x
            s.t.   a⁻ ≤ a ≤ a⁺

        The term xᵀ S_xx x is constant in a as it's fixed and thus I can drop it.
        S_aa ≻ 0 (since S ≻ 0), so the QP is strictly convex → unique solution !

        Strategy :
        -----------
        I first try the unconstrained analytical minimiser. 
        The unconstrained minimum is analytic:
            a* = -S_aa⁻¹ S_ax x_full ∈ [a_lo, a_hi],

        If a* ∈ [a⁻, a⁺], return it directly (no solver call needed).

        Otherwise, call and try more complex solvers from CVXPY for the constrained QP.
        """
        
        s_start = 1 if self.cfg.use_bias else 0
        a_start = s_start + self.n_state
        
        # Extract the good sub-blocks
        S_aa = 0.5 * (S[a_start:, a_start:] + S[a_start:, a_start:].T) # Block action-action 
        S_as = S[a_start:, s_start:a_start] #Block action-state S_ax
        
        # With a bias term, the linear term in a includes S_a0 due to the cost shifting technique
        if self.cfg.use_bias:
            s_a0 = S[a_start:, 0]
        else:
            s_a0 = np.zeros(self.n_act)

        lin = S_as @ x_state + s_a0 # (S_ax x)
        
        # First try the unconstrained optimum
        try:
            a_unc = -np.linalg.solve(S_aa, lin)
            if np.all(a_unc >= a_lo_n - 1e-8) and np.all(a_unc <= a_hi_n + 1e-8):
                # if the unconstrained minimiser is already in the admissible range then we can return it directly
                return np.clip(a_unc, a_lo_n, a_hi_n) # No solver used -> much faster !
        except np.linalg.LinAlgError:
            pass
        
        # Safe fallback: convex bounded QP via CVXPY solvers 
        a_var = cp.Variable(self.n_act) # the unknown action vector
        objective = cp.Minimize(cp.quad_form(a_var, S_aa) + 2.0 * lin @ a_var) # min_a  Q̂(x, a) = aᵀ S_aa a  +  2 (S_ax x)ᵀ a
        constraints = [a_var >= a_lo_n, a_var <= a_hi_n] # under constraint a⁻ ≤ a ≤ a⁺
        prob = cp.Problem(objective, constraints)
        status = _solve(prob, "policy_improvement")

        if a_var.value is None:
            logger.warning("Policy improvement failed (status=%s); returning clipped zero action.", status)
            # If the solver fails we return the null action (do nothing -> safe)
            return np.clip(np.zeros(self.n_act), a_lo_n, a_hi_n)

        return np.clip(np.asarray(a_var.value).ravel(), a_lo_n, a_hi_n)
    
    
    
    # -- State construction ---------------------------------------------------
    
    def _build_state_matrix(self, x_feat: np.ndarray, p: np.ndarray):
        """
        Build the internal state matrix used by OPI.
        If parameters are included in the state, the returned matrix is:
            [normalised biomechanical features | normalised parameters]

        Otherwise only the normalised biomechanical features are returned.
        """
        x_feat_n = self.feature_norm.transform(x_feat)
        if self.cfg.include_params_in_state:
            p_n = self.param_norm.normalize(p)
            return np.hstack([x_feat_n, p_n])
        return x_feat_n
    
    
    def _build_single_state(self, x_feat_k: np.ndarray, p_k: np.ndarray):
        """Single-sample wrapper around '_build_state_matrix' returning a 1D vector."""
        x_feat_k = np.asarray(x_feat_k, dtype=float).reshape(1, -1)
        p_k = np.asarray(p_k, dtype=float).reshape(1, -1)
        return self._build_state_matrix(x_feat_k, p_k).ravel()
    
    
    def _joint_vector(self, x_state: np.ndarray, a_n: np.ndarray) -> np.ndarray:
        """
        Build the joint vector μ used inside the quadratic approximator.
        Depending on the configuration, μ is either:
            [x_state ; a_n] or [1 ; x_state ; a_n]
        """
        if self.cfg.use_bias:
            return np.concatenate([np.ones(1), x_state, a_n])
        return np.concatenate([x_state, a_n])
    
    
    
    # -- Reward → cost --------------------------------------------------------
    
    def _reward_to_cost(self, r: np.ndarray):
        """
        Convert rewards into strictly positive costs.
        The transformation is:
            U = cost_shift - r
        where 'cost_shift' is learned from the reward vector as max(r) + margin.
        """
        r = np.asarray(r, dtype=float)
        self.cost_shift_ = float(np.max(r) + self.cfg.cost_shift_margin)
        U = self.cost_shift_ - r
        return U
    
    
    
    
    # -- Policy helpers -------------------------------------------------------
    
    def _make_greedy_policy(self, S: np.ndarray):
        """ Return the greedy policy induced by matrix S. """
        
        def _policy(x_state: np.ndarray, p_k: np.ndarray):
            a_lo_n, a_hi_n = self._normalised_action_bounds(p_k)
            return self._policy_improvement_qp(x_state, S, a_lo_n, a_hi_n)

        return _policy
    
    
    def _make_zero_policy(self):
        """
        Return the initial safe policy π⁽⁰⁾ that always proposes the zero
        incremental action, clipped to the admissible bounds for the current p_k.
        """

        def _policy(x_state: np.ndarray, p_k: np.ndarray):
            del x_state
            a_lo_n, a_hi_n = self._normalised_action_bounds(p_k)
            return np.clip(np.zeros(self.n_act), a_lo_n, a_hi_n)

        return _policy
    
    
    def _normalised_action_bounds(self, p_k: np.ndarray):
        """
        Compute state-dependent action bounds in the normalised action space.
        
        The physical incremental action bounds are:
            - a_lo = p_min - p_k
            - a_hi = p_max - p_k
        because p_k + a must remain inside the prosthesis physical limits.
        
        The method then maps these bounds into the normalised action space.
        """
        a_lo_phys = self.cfg.prosthesis.p_min - np.asarray(p_k, dtype=float)
        a_hi_phys = self.cfg.prosthesis.p_max - np.asarray(p_k, dtype=float)
        return self.action_norm.normalize_bounds(a_lo_phys, a_hi_phys)
    
    
    def _batch_apply(self, policy: Callable[[np.ndarray, np.ndarray], np.ndarray], Xn: np.ndarray, P: np.ndarray):
        """
        Apply a policy row-by-row to a batch of states and parameter vectors.
        This helper function is used during policy evaluation to build π(s'_n) for all next states in the dataset.
        """
        return np.array([policy(Xn[i], P[i]) for i in range(len(Xn))], dtype=float)
    
    
    
    # -- Early-stopping detectors -------------------------------------------------------

    def _detect_limit_cycle(self, iter_idx: int):
        """
        Detect limit cycles of period p in the history of ||ΔS||_F.

        A limit cycle of period p means the policy iteration alternates between p ≈ stationary regimes without ever contracting strictly.
        This is distinct from the plateau case handled separately: the plateau detector requires amplitude < cycle_tol_abs, whereas here the amplitude of the oscillation is explicitly allowed to be NON-negligible as long as the orbit is periodic.

        Detection principle
        -------------------
        For each candidate period p ∈ {2, ..., cycle_period_max}:
          - at least 3·p iterations of history are required (2 windows to compare, plus a 1·p buffer to evaluate the peak-to-peak span reliably);
          - let w₁ = f[-2p:-p] and w₂ = f[-p:] be the two most recent consecutive windows of size p;
          - let span = max(f[-3p:]) - min(f[-3p:]) be the full peak-to-peak amplitude observed over the three last windows;
          - declare a p-cycle iff
                max_i |w₁[i] - w₂[i]|  <  cycle_period_tol_rel · span,
            i.e. the two windows overlap tightly relative to the overall amplitude of the oscillation.

        A sanity floor span >= cycle_tol_abs is enforced so that strict plateau cases (quasi-zero amplitude, period undefined) are not reported here, they are the province of the plateau detector above.

        Returns
        -------
        None if no cycle is detected, else a tuple (p, span) giving the detected period and its peak-to-peak amplitude.
        """
        f = self.history["frobenius"]
        # Iterate from the shortest period upwards: if an orbit of period 2 exists it is also an orbit of period 4, so reporting the smallest period is the most informative choice.
        for p in range(2, self.cfg.cycle_period_max + 1):
            if iter_idx + 1 < 3 * p:
                continue

            recent = np.asarray(f[-3 * p:], dtype=float)
            span = float(recent.max() - recent.min())

            # Skip if amplitude is below the plateau threshold: that case is (and should be) caught by the dedicated plateau detector.
            if span < self.cfg.cycle_tol_abs:
                continue

            w1 = np.asarray(f[-2 * p:-p], dtype=float)
            w2 = np.asarray(f[-p:],       dtype=float)
            per_index_diff = float(np.max(np.abs(w1 - w2)))

            if per_index_diff < self.cfg.cycle_period_tol_rel * span:
                return p, span

        return None


    def _detect_no_progress(self, iter_idx: int):
        """
        Declare that progress has stalled when the best ||ΔS||_F achieved over the last 'stagnation_window' iterations fails to improve upon the best achieved over the preceding window of equal length.

        Unlike the plateau detector (which requires amp < cycle_tol_abs) and the limit-cycle detector (which requires strict periodicity), this criterion captures CHAOTIC but bounded behaviour of the policy-evaluation / improvement alternation. 
        In such regimes the per-iteration ||ΔS||_F fluctuates within a wide band without converging, yet beyond some point no new information is being extracted (empirically confirmed on CV folds 1, 2, 3 and 5 of run 793553 where the running minimum flattens out after ~25 iterations).

        Stopping criterion
        ------------------
            min(f[-w:])  >=  (1 - tol) · min(f[-2w:-w])

        with w = stagnation_window and tol = stagnation_tol_improve.

        Returns (recent_min, prior_min) if stagnation is declared, else None. 
        Setting stagnation_window = 0 disables this detector.
        """
        w = self.cfg.stagnation_window
        if w <= 0 or (iter_idx + 1) < 2 * w:
            return None
        f = self.history["frobenius"]
        recent_min = min(f[-w:])
        prior_min  = min(f[-2 * w:-w])
        tol = self.cfg.stagnation_tol_improve
        if recent_min >= (1.0 - tol) * prior_min:
            return recent_min, prior_min
        return None
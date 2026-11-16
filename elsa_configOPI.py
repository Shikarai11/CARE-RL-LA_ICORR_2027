from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Optional, Tuple
 
import numpy as np

# ============================================================================
# |                                                                          |
# |  1) PROSTHESIS PHYSICAL BOUNDS                                           |
# |                                                                          |
# ============================================================================
@dataclass
class ProsthesisParams:
    """
    Physical bounds for the three ELSA parameters that are tuned.
        d_dors : damping level in dorsiflexion [Nm·s/rad] -> bounds: [9.74, 28.65]
                 (stored under the historical name d_plant in attributes, CSV columns and CLI flags)
        θ_trig : push-off trigger angle [rad] -> bounds: [0.052, 0.192]
        τ_lim : torque limit during push-off [Nm] -> bounds: [10, 60]
    """
    d_plant_bounds: Tuple[float, float] = (9.74, 28.65)
    theta_trig_bounds: Tuple[float, float] = (0.052, 0.192)
    tau_lim_bounds: Tuple[float, float] = (10.0, 60.0)

    @property
    def p_min(self):
        """Lower bounds as a vector [d_dors_min, theta_min, tau_min]."""
        return np.array([self.d_plant_bounds[0], self.theta_trig_bounds[0], self.tau_lim_bounds[0]])

    @property
    def p_max(self):
        """Upper bounds as a vector [d_dors_max, theta_max, tau_max]."""
        return np.array([self.d_plant_bounds[1], self.theta_trig_bounds[1], self.tau_lim_bounds[1]])

    @property
    def p_range(self):
        """Total physical span of each parameter."""
        return self.p_max - self.p_min

    @property
    def p_half_range(self):
        """
        Half the span of each parameter.
        Used to normalise actions so that a normalised action of +1 corresponds to a change equal to half the total admissible range.
        """
        return self.p_range / 2.0
    
    
    
# ============================================================================
# |                                                                          |
# |  2) ELSA-SPECIFIC PARAMETER NORMALISER                                   |
# |                                                                          |
# ============================================================================
class ParamNormalizer:
    """
    Normaliser for ABSOLUTE prosthesis parameters.

    Uses known physical bounds rather than dataset statistics:
        p_n = (p - center) / half_range  ∈ [-1, 1]

    This is preferred over a data-driven normaliser for parameters because their admissible range is fixed by hardware and is known in advance.
    """

    def __init__(self, cfg: "OPIConfig"):
        self.p_min    = cfg.prosthesis.p_min
        self.p_max    = cfg.prosthesis.p_max
        self.scale    = np.maximum(cfg.prosthesis.p_half_range, 1e-8)
        self.center   = (self.p_min + self.p_max) / 2.0
 
    def normalize(self, p: np.ndarray):
        """Map absolute prosthesis parameters from physical units to [-1, 1]."""
        return (np.asarray(p, dtype=float) - self.center) / self.scale
    
    
    
    
# ============================================================================
# |                                                                          |
# |  3)  MASTER CONFIGURATION                                                |
# |                                                                          |
# ============================================================================
@dataclass
class OPIConfig:
    """
    Master configuration for the OPI algorithm applied to ELSA.
    
    Dimensions
    -----------
      n_features : number of biomechanical features in x_k. Default 6 → {f1, ..., f6}. Use 8 to include std of features 1 and 2.
      include_params_in_state : if True, the current absolute prosthesis parameters p_k are appended to the state vector.
      n_state_total : n_features (+ n_actions if params are in state).
      n_actions : always 3 = (Δd_dors, Δθ_trig, Δτ_lim).
      n_joint : dimension of μ = [bias?; state; action].

    RL hyper-parameters
    -------------------------
      gamma : discount factor - Li et al. recommend γ ∈ [0.8, 0.95].
      n_iterations : maximum policy-iteration steps.
      tol : Frobenius-norm convergence threshold.

    Reward shaping
    -----------------
      lambda_ : balance between prosthesis and human reward.
      alpha : balance between comfort and perceived assistance.
      feature_weights : wᵢ - diagonal weight matrix on state deviations.
      R_a : (n_actions x n_actions) PD - penalises large param jumps.
      assist_coding : "raw" uses the assistance score as recorded, "centered" uses -|A - assist_optimum| since the assistance scale is bipolar (5 = appropriate assistance).
      action_cost : "mixed" keeps aᵀR_a a inside r^{prosthesis}, weighted by (1 - λ); "separate" keeps it at full weight whatever λ.

    Solver
    -------
      SDP_eps : small ε for the S ⪻ εI PSD constraint.
      cost_shift_margin: added to max(r) to keep costs strictly positive.
      use_bias : if True, prepend a constant 1 to μ so that the approximator can represent a constant cost offset.
      lambda_reg : Tikhonov regularisation weight on ‖S‖²_F added to the policy-evaluation objective. Prevents overfitting when the dataset is small relative to the number of parameters in S. Set to 0.0 to disable regularisation.
    
    Cycle / plateau early stopping
    --------------------------------
    During my baseline experiments, I observe that the alternation between policy evaluation and policy improvement may reach a LIMIT CYCLE rather than a strict fixed point. I saw two types of limit cycles in particular:
        - a clean 2-cycle where ||ΔS||_F alternates between two values,
        - a noisy plateau where ||ΔS||_F fluctuates within a narrow band dominated by solver tolerance, without ever dropping below "tol".
    In both cases, continuing to iterate does NOT improve S (I verified this empirically on the baseline run 791411: ΔS stagnates from iter ~22 onwards while the default n_iterations=100 wastes ~70 % of the fit run time).
    
    Early stopping is declared when BOTH of the following hold AND the iteration index is at least 2·cycle_window (guaranteeing enough history):
        max(window) - min(window) <  cycle_tol_abs
        (max - min) / median(window) <  cycle_tol_rel
    where 'window' denotes the last 'cycle_window' values of ||ΔS||_F.
    The two tolerances cover complementary cases:
      - cycle_tol_abs catches clean cycles at small ||ΔS|| (e.g. 6.56 +- 0.00).
      - cycle_tol_rel catches plateaus at LARGE ||ΔS|| (e.g. the fold-4 plateau at ||ΔS||=141, observed  would not trigger on absolute amplitude alone).
    Set cycle_window = 0 to disable the detection entirely and recover the legacy behaviour (run until strict convergence or n_iterations, whichever comes first).
    """

    # -- Dimensions --------------------------------------------------------
    n_features: int = 6 # Change HERE to add bonus features (e.g. stds)
    n_actions: int = 3

    include_params_in_state: bool = False

    # -- ELSA hardware bounds -----------------------------------------------
    prosthesis: ProsthesisParams = field(default_factory=ProsthesisParams)

    # -- RL hyper-parameters ------------------------------------------------
    gamma: float = 0.9
    n_iterations: int = 100
    tol: float = 1e-4

    # -- Reward shaping -----------------------------------------------------
    lambda_: float = 0.5 
    alpha: float = 0.5
    feature_weights: Optional[np.ndarray] = None # wᵢ (n_features,) PD
    R_a: Optional[np.ndarray] = None # (n_actions, n_actions) PD
    assist_coding: str = "raw" # coding used when the dataset rewards were frozen
    assist_optimum: float = 5.0
    action_cost: str = "mixed"

    # -- Solver settings ----------------------------------------------------
    SDP_eps: float = 1e-6
    cost_shift_margin: float = 1e-3
    use_bias: bool  = True
    # Tikhonov regularisation on S during policy evaluation.
    # Penalises large entries in S (λ_reg ‖S‖²_F added to the objective).
    # Tune upward if S is unstable or actions are erratic; set to 0 to disable.
    lambda_reg: float = 0.0
    
    # -- Cycle / plateau early stopping -------------------------------------
    cycle_window: int = 5
    cycle_tol_abs: float = 1e-3
    cycle_tol_rel: float = 1e-3
    # Limit cycle-detection 
    # Detection rule (see OPI_core._detect_limit_cycle): 
    # for each candidate period p ∈ {2, ..., cycle_period_max}, declare a p-cycle iff the two most recent windows of size p differ, element-wise, by less than cycle_period_tol_rel × peak-to-peak(last 3·p iterations).
    cycle_period_max: int = 8 # maximum period to check for in the limit cycle detection
    cycle_period_tol_rel: float = 0.02
    # No-progress early stopping :
    # Catches CHAOTIC (non-periodic but bounded) oscillations of ||ΔS||_F. Empirically observed on run 793553 for CV folds 1, 2, 3 and 5, where the running minimum stops improving after ~25-30 iterations while the per-iteration norm keeps fluctuating widely.
    # Detection rule (see OPI_core._detect_no_progress):
    #       min(f[-w:]) >= (1 - tol) · min(f[-2w:-w])
    # i.e. the best ||ΔS||_F over the most recent window fails to improve the best over the preceding window of equal length by more than `stagnation_tol_improve`. 
    # tol = 0.0 means "any strict improvement counts"; positive values tolerate some degradation.
    # Set stagnation_window = 0 to disable this detector entirely.
    stagnation_window: int = 25
    stagnation_tol_improve: float = 0.0 # minimum relative improvement in ||ΔS||_F over the stagnation window to avoid early stopping
    return_best_S_on_early_stop: bool = True # if True, return the best S found so far when early stopping is triggered; otherwise return the last S

    # -- Per-dimension R_a rescaling ----------------------------------------
    # Element-wise multiplier applied to the diagonal of R_a after its default construction from prosthesis half-ranges.
    # Default = None (no rescaling; legacy isotropic R_a).
    #
    # Motivation: when the greedy QP saturates one specific face of the admissible polytope on many samples 
    # (e.g. 47 % of samples hit the lower bound on τ_lim in run 795205), increasing R_a selectively on that dimension,
    # without modifying the others, makes the parasitic optimum suboptimal and forces the policy to rely on the biomechanical signal rather than on the polytope face.
    #
    # Example:
    #     R_a_per_dim_scale = np.array([1.0, 1.0, 5.0])
    # triples the cost of the τ_lim action dimension without modifying d_dors and θ_trig.
    R_a_per_dim_scale: Optional[np.ndarray] = None
    



    def __post_init__(self):
        """
        Validate and standardise configuration arrays.
        This method ensures that optional arrays are automaticaly (thanks to @dataclass package cf. https://docs.python.org/3/library/dataclasses.html) initialised to default values and that their shapes are compatible with the chosen state and action dimensions.
        """
        if self.feature_weights is None:
            self.feature_weights = np.ones(self.n_features) # equal weights
        if self.R_a is None:
            # R_a = diag(1 / half_range²) makes the action penalty ISOTROPIC in the normalized action space:
            # a step of size ε·half_range on each parameter costs exactly ε², regardless of the parameter.
            # I make this to avoid bias resulting from differences in scale between parameters
            half = self.prosthesis.p_half_range
            R_a_diag = 1.0 / np.maximum(half**2, 1e-12)
            
            # Apply per-dimension rescaling if requested.
            if self.R_a_per_dim_scale is not None:
                scale = np.asarray(self.R_a_per_dim_scale, dtype=float)
                if scale.shape != (self.n_actions,):
                    raise ValueError(
                        f"R_a_per_dim_scale must have shape ({self.n_actions},), "
                        f"got {scale.shape}"
                    )
                if np.any(scale <= 0):
                    raise ValueError(
                        "R_a_per_dim_scale entries must all be strictly positive"
                    )
                R_a_diag = R_a_diag * scale

            self.R_a = np.diag(R_a_diag)

        self.R_a = np.asarray(self.R_a, dtype=float)
        self.feature_weights = np.asarray(self.feature_weights, dtype=float)

        if self.R_a.shape != (self.n_actions, self.n_actions):
            raise ValueError("R_a must have shape (n_actions, n_actions).")
        if self.feature_weights.shape != (self.n_features,):
            raise ValueError("feature_weights must have shape (n_features,).")
        if self.assist_coding not in ("raw", "centered"):
            raise ValueError("assist_coding must be 'raw' or 'centered'.")
        if self.action_cost not in ("mixed", "separate"):
            raise ValueError("action_cost must be 'mixed' or 'separate'.")
        
        
    @property
    def n_state_total(self):
        """Total state dimension seen by the controller."""
        return self.n_features + (self.n_actions if self.include_params_in_state else 0)
    
    @property
    def n_joint(self):
        """
        Dimension of the joint vector μ = [bias?; state; action].
        μ contains, in order:
          - optional bias term 1 (for the cost shift),
          - full state vector,
          - action vector.
        """
        return (1 if self.use_bias else 0) + self.n_state_total + self.n_actions


    def build_param_normalizer(self):
        """Instantiate the ELSA-specific parameter normaliser from this config."""
        return ParamNormalizer(self)
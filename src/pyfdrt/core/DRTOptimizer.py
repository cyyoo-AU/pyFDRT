import numpy as np
from scipy.optimize import minimize_scalar, curve_fit, lsq_linear
from scipy.signal import find_peaks, find_peaks_cwt

from .calcFDRT import calcFDRT
from .reverseDRT import reverse_drt
from ..io.eis_utils import lcurve_corner


# ---------------------------------------------------------------------------
# Empirically tuned deconvolution constants (task #35).
# Each value encodes real tuning against the 15-dataset reference corpus and
# the synthetic study (docs/specs/drt-deconvolution.md); they are collected
# here so the tuning is discoverable in one place instead of scattered as
# magic numbers. Changing any of them changes pinned behaviour under
# docs/specs/drt-conformance.md.

# Measured-window clamp margin, in log10(tau) decades: half a grid step at the
# fixed 10 points/decade - wide enough to keep the edge samples, narrow enough
# to exclude the extrapolated region (drt-deconvolution.md 3.2).
WINDOW_MARGIN_DECADES = 0.05

# NNLS target: total least-squares influence of the synthetic LF-extrapolated
# rows as a fraction of the measured rows (drt-core.md 5.2, task #64). The rows
# anchor the DRT at the slow edge but are synthetic; unweighted they drag the
# fit off the measured data whenever the linear LF tail is biased, and their
# influence scales with decade count x interpolation density. Each synthetic
# row is weighted w = min(1, sqrt(this * n_meas / n_syn)).
SYNTH_LF_ROW_INFLUENCE = 0.1

# Tail-cut detection: a sample still counts as part of the monotonic rise while
# its neighbour is within this multiplicative band (2%), so measurement ripple
# does not break the walk (drt-deconvolution.md 2.1).
TAIL_CUT_NOISE_FACTOR = 1.02

# Tail-cut detection: the rise into the low-frequency edge must exceed this
# fraction of the in-window maximum before it is treated as a diffusion branch;
# smaller rises are noise (drt-deconvolution.md 2.1).
TAIL_CUT_RISE_FRACTION = 0.05

# LASSO: dictionary atoms below this fraction of the largest coefficient are
# discarded before clustering (drt-deconvolution.md 4.1).
LASSO_ATOM_THRESHOLD_FRACTION = 0.02

# LASSO: atoms whose centres are closer than this many decades merge into one
# peak candidate (drt-deconvolution.md 4.1).
LASSO_CLUSTER_DECADES = 0.35

# CWT: detected peaks closer than this many decades merge into one candidate
# (drt-deconvolution.md 4.2).
CWT_CLUSTER_DECADES = 0.30

# Peaks whose R is below this fraction of the largest peak's R are dropped from
# the report (drt-deconvolution.md 6.2).
NEGLIGIBLE_PEAK_FRACTION = 0.02

# Misfit-rescue stage (drt-deconvolution.md 7.3, task #84). The stage fires
# only when the refined R-CPE model has visibly failed against the measured
# impedance (misfit above the gate; corpus margin: Flow_15wt 6.2 % vs next
# highest 1.15 %), seeds one element in the DRT residual away from existing
# peaks (the exclusion keeps it off the near-peak kernel mismatch that would
# otherwise split the main arc), and adopts it only on a large improvement.
RESCUE_MISFIT_GATE = 0.03
RESCUE_EXCLUSION_DECADES = 0.7
RESCUE_MIN_IMPROVEMENT = 2.0

# Composite diffusion-branch refit (drt-deconvolution.md 2.4.2, task #89):
# adopt the slow-ZARC + Warburg composite only when it beats the single
# Warburg-open refit by this factor on the same residual target, and the
# near-optimal multistart solutions agree on the slow element's tau within
# this many decades. Both guards protect against reporting a ridge.
COMPOSITE_MIN_IMPROVEMENT = 3.0
COMPOSITE_STABILITY_DECADES = 0.2
# How close to the 1.0 Debye ceiling the composite's slow-ZARC phi must sit
# before the log marks it as not-a-measured-shape (drt-deconvolution.md 2.4.2,
# task #99). Diagnostic only: the ceiling remains an admissible solution, so
# this threshold cannot change which fits are adopted.
COMPOSITE_PHI_CEILING_TOL = 1e-3

# R-budget diagnostic (task #87): warn when the reported relaxation elements'
# total resistance differs from the NNLS mass inside the deconvolution window
# by more than this fraction. Corpus reference: 13/15 datasets sit within 5%;
# the two outliers (550C_BZYPtFC, Ciucci/LIB_data, both ~1.17) are exactly
# the datasets whose bound-limited slowest element (task #83) absorbs
# impedance beyond the measured window. Logging only - nothing pinned moves.
R_BUDGET_TOL = 0.15

# Minimum diffusion-branch contribution, as a fraction of max|Z|, for the
# jointly fitted Warburg column to mean anything (drt-core.md 5.5). It gates
# two things that must agree: whether the pipeline reports the element at all,
# and whether the joint record may claim an identified knee (task #98). Both
# read this name - the reporting gate in pipeline.py imports it rather than
# repeating the value (task #117), so the two cannot be tuned apart. A clean
# non-diffusion spectrum drives R_W to ~0 and lands below it.
WARBURG_MIN_CONTRIBUTION = 1e-3


class DRTOptimizer:
    """
    Refines FFT-based DRT results and extracts final ECM parameters.
    Pipeline: Auto-Tune -> NNLS Polish -> Gaussian Deconvolution
    """

    def __init__(self, EIS_preprocessed: np.ndarray, EIS_raw: np.ndarray = None, R_inf: float = None):
        """
        :param EIS_preprocessed: interpolated/extrapolated data for FFT
        :param EIS_raw: optional raw data used as NNLS fitting target
        :param R_inf: optional external ohmic resistance value (e.g. from extrapolation)
        """
        self.EIS    = EIS_preprocessed
        self.freq   = EIS_preprocessed[:, 0]
        self.Z_re   = EIS_preprocessed[:, 1]
        self.Z_im   = EIS_preprocessed[:, 2]

        self.target_data = EIS_raw if EIS_raw is not None else EIS_preprocessed

        # Fixed log-spaced tau grid at 10 pts/decade.
        #
        # Low-frequency bound  → preprocessed (extrapolated) minimum frequency.
        #   Extending into the low-freq extrapolated region allows the DRT to
        #   represent open arcs whose peak lies below the measurement window
        #   (e.g. a dominant capacitive arc that has not closed at f_min_meas).
        #
        # High-frequency bound → measured maximum frequency (target_data).
        #   Deliberately NOT extended into the high-freq extrapolated region.
        #   If tau basis functions are placed above f_meas_max, the NNLS fits
        #   the extrapolated (RC-model) data there and distributes DRT weight at
        #   those tau values.  Even though those relaxations are above the
        #   measurement window, their Voigt contributions sum over ALL measured
        #   frequencies and produce a systematic positive Z_re offset that cannot
        #   be corrected by R_inf subtraction alone.  Stopping the tau grid at
        #   the measured high-frequency limit eliminates this artifact entirely.
        _log10_2pi  = np.log10(2.0 * np.pi)
        _log_f_min  = float(np.log10(EIS_preprocessed[:, 0].min()))   # low-freq ext
        _log_f_max  = float(np.log10(self.target_data[:, 0].max()))    # measured max
        _n_tau      = max(50, int(round((_log_f_max - _log_f_min) * 10)) + 1)
        self.tau     = 10 ** np.linspace(-_log_f_max - _log10_2pi,
                                          -_log_f_min - _log10_2pi, _n_tau)
        self.log_tau = np.log10(self.tau)

        # log-tau window spanned by the *measured* frequencies (EIS_raw). Peaks
        # found outside this band live purely in the extrapolated region and are
        # numerical artefacts, so peak detection is restricted to it. This keeps
        # the extracted ECM peaks invariant to the extrapolation range.
        meas_f          = self.target_data[:, 0]
        meas_log_tau    = np.log10(1.0 / (2 * np.pi * meas_f))
        self.meas_log_tau_min = float(np.min(meas_log_tau))
        self.meas_log_tau_max = float(np.max(meas_log_tau))

        self.best_noise   = None
        self.base_drt     = None
        self.polished_drt = None
        # Joint Warburg-open branch fitted alongside the RC basis by
        # polish_spectrum_joint (task #56); None on the default post-hoc path.
        self.joint_warburg = None
        self.R_inf_injected = R_inf
        # Shared ohmic resistance for NNLS fitting, reconstruction and ECM
        # extraction.  Set by polish_spectrum from np.min(self.EIS[:,1])
        # (the preprocessed/extrapolated spectrum minimum, which converges to
        # the true series resistance) and reused consistently everywhere so that
        # the polished DRT, its Nyquist reconstruction, and the extracted ECM
        # parameters all share one coherent R_inf.
        self.R_inf        = None
        self.solver_method = 'bvls'

    def auto_tune(self) -> tuple:
        """
        Step 1: Find the optimal Wiener filter noise strength that minimises
        reconstruction MSE against the original impedance data.
        """
        print("-> [Step 1] Auto-tuning noise filter...")

        def objective_function(noise_param):
            try:
                gEIS, _   = calcFDRT(self.EIS, windowfunction2='wiener', addjustnoise=noise_param)
                simulated = reverse_drt(self.freq, gEIS[:, 3])
                return (np.sum((self.Z_re - simulated[:, 1])**2)
                        + np.sum((self.Z_im - simulated[:, 2])**2))
            except Exception:
                return np.inf

        res             = minimize_scalar(objective_function, bounds=(-2.0, 2.0), method='bounded')
        self.best_noise = res.x

        best_gEIS, _  = calcFDRT(self.EIS, windowfunction2='wiener', addjustnoise=self.best_noise)
        self.base_drt = best_gEIS[:, 3]

        print(f"   Done: optimal filter strength (addjustnoise) = {self.best_noise:.4f}")
        return self.base_drt, self.best_noise

    def _nnls_system(self, prior: np.ndarray = None):
        """Assemble the operators shared by optimize_lambda and polish_spectrum.

        Fitting target: low-frequency extrapolated data + raw measured data.
        High-frequency extrapolated data is excluded — including it creates a
        non-zero b_re in the extrapolated band that drives DRT pile-up at the
        high-freq tau boundary and causes systematic Z_re over-prediction
        there. R_inf is the injected extrapolation value (fallback: full EIS
        minimum). Using the raw measurements directly (not the interpolated
        self.EIS values) eliminates the interpolation-smoothing residual at
        the measurement boundary.

        Returns (A_stacked, b_stacked, L_mat, x_prior, R_inf); x_prior is the
        prior interpolated onto the tau grid, or None.
        """
        _f_meas_min = float(self.target_data[:, 0].min())
        _lo_mask    = self.EIS[:, 0] < _f_meas_min * 0.999
        _EIS_fit    = np.vstack([self.EIS[_lo_mask], self.target_data])
        _EIS_fit    = _EIS_fit[np.argsort(_EIS_fit[:, 0])]

        W     = (2 * np.pi * _EIS_fit[:, 0])[:, np.newaxis]
        T     = self.tau[np.newaxis, :]
        WT    = W * T
        denom = 1.0 + WT ** 2
        A_stacked = np.vstack((1.0 / denom, -WT / denom))

        R_inf = (self.R_inf_injected if self.R_inf_injected is not None
                 else float(np.min(self.EIS[:, 1])))
        b_stacked = np.concatenate((_EIS_fit[:, 1] - R_inf, _EIS_fit[:, 2]))

        # Down-weight the synthetic LF-extrapolated rows (drt-core.md 5.2,
        # task #64): cap their collective least-squares influence at
        # SYNTH_LF_ROW_INFLUENCE of the measured rows, independent of decade
        # count and interpolation density. Both the Re and the Im block of
        # each synthetic row are scaled; an empty block is a no-op and the
        # cap prevents up-weighting a sparse block.
        _syn = _EIS_fit[:, 0] < _f_meas_min * 0.999
        _n_syn = int(_syn.sum())
        if _n_syn:
            w_syn = min(1.0, float(np.sqrt(
                SYNTH_LF_ROW_INFLUENCE * (len(_EIS_fit) - _n_syn) / _n_syn)))
            if w_syn < 1.0:
                _m = np.concatenate([_syn, _syn])
                A_stacked[_m] *= w_syn
                b_stacked[_m] *= w_syn

        N_tau = len(self.tau)
        L_mat = np.zeros((N_tau, N_tau))
        for i in range(1, N_tau - 1):
            L_mat[i, i-1] = 1.0; L_mat[i, i] = -2.0; L_mat[i, i+1] = 1.0

        x_prior = None
        if prior is not None:
            if len(prior) == len(self.freq):
                log_tau_src = np.log10(1.0 / (2.0 * np.pi * self.freq))
            elif len(prior) == len(self.tau):
                log_tau_src = self.log_tau.copy()
            else:
                log_tau_src = np.linspace(self.log_tau[0], self.log_tau[-1], len(prior))
            sort_idx = np.argsort(log_tau_src)
            from scipy.interpolate import interp1d
            interp_func = interp1d(log_tau_src[sort_idx], prior[sort_idx],
                                   bounds_error=False, fill_value=0.0)
            x_prior = interp_func(self.log_tau)

        return A_stacked, b_stacked, L_mat, x_prior, R_inf

    def optimize_lambda(self, method: str = 'L-Curve', prior: np.ndarray = None, lambda_prior: float = 0.05) -> float:
        # Default matches the production criterion (task #68): the pipeline
        # passes 'L-Curve' explicitly, so a library caller omitting the
        # argument must not silently get a different selection rule.
        print(f"-> [Lambda optimisation] Searching optimal smoothing strength via {method}...")

        # Calibrate on the same problem polish_spectrum solves.
        A_stacked, y_stacked, L_mat, x_prior, _ = self._nnls_system(prior)
        N_tau = len(self.tau)

        if self.best_noise is not None:
            center  = -2.0 + float(self.best_noise)
            log_low = center - 2.0
            log_hi  = center + 3.0
            print(f"   - best_noise={self.best_noise:.3f} -> lambda scan 10^[{log_low:.1f}, {log_hi:.1f}]")
        else:
            log_low, log_hi = -4.0, 1.0

        lambdas                           = np.logspace(log_low, log_hi, 50)
        residuals, roughnesses, gcv_scores = [], [], []
        N_data                            = len(y_stacked)
        I_mat                             = np.eye(N_tau)

        for lam in lambdas:
            if x_prior is not None:
                res = lsq_linear(
                    np.vstack((A_stacked, np.sqrt(lam) * L_mat, np.sqrt(lambda_prior) * I_mat)),
                    np.concatenate((y_stacked, np.zeros(N_tau), np.sqrt(lambda_prior) * x_prior)),
                    bounds=(0, np.inf),
                    method=self.solver_method
                )
            else:
                res = lsq_linear(
                    np.vstack((A_stacked, np.sqrt(lam) * L_mat)),
                    np.concatenate((y_stacked, np.zeros(N_tau))),
                    bounds=(0, np.inf),
                    method=self.solver_method
                )
            x         = res.x
            res_norm  = np.linalg.norm(A_stacked @ x - y_stacked)**2
            reg_norm  = np.linalg.norm(L_mat @ x)**2
            residuals.append(res_norm)
            roughnesses.append(reg_norm)

            if method == 'GCV':
                df = np.sum(x > 1e-8)  # effective degrees of freedom for NNLS
                gcv_scores.append(res_norm / ((1 - df / N_data)**2))

        if method == 'GCV':
            best_lam = lambdas[np.argmin(gcv_scores)]
            print(f"   Done: GCV optimal Lambda = {best_lam:.4e}")
            return float(best_lam)

        elif method == 'L-Curve':
            best_lam = lcurve_corner(lambdas, residuals, roughnesses)
            # Defensive: a corner sitting on a scan edge means the
            # best_noise-centred window did not bracket the true corner
            # (drt-core.md 7.4). Log-only - log lines are unpinned.
            if best_lam in (lambdas[0], lambdas[-1]):
                edge = 'lower' if best_lam == lambdas[0] else 'upper'
                print(f"   Warning: L-Curve corner landed on the scan's {edge} "
                      f"edge (lambda = {best_lam:.4e}); the scan window may not "
                      f"bracket the true corner.")
            print(f"   Done: L-Curve optimal Lambda = {best_lam:.4e}")
            return float(best_lam)

        else:
            raise ValueError(f"method must be 'GCV' or 'L-Curve', got {method!r}")

    def polish_spectrum(self, lambda_reg: float = 0.01, prior: np.ndarray = None, lambda_prior: float = 0.05) -> np.ndarray:
        """
        Step 2: NNLS DRT Inversion with 2nd-derivative smoothness regularisation
        and optional prior guideline penalty:

            min_{x >= 0}  || A x - b ||^2 + lambda_reg * || L x ||^2 + lambda_prior * || x - x_prior ||^2
        """
        print("-> [Step 2] NNLS Polishing (smoothness & prior regularisation)...")

        A, b, L_mat, x_prior, R_inf = self._nnls_system(prior)
        self.R_inf = R_inf
        N_tau = len(self.tau)

        if x_prior is not None:
            I_mat = np.eye(N_tau)
            res = lsq_linear(np.vstack((A, np.sqrt(lambda_reg) * L_mat, np.sqrt(lambda_prior) * I_mat)),
                             np.concatenate((b, np.zeros(N_tau), np.sqrt(lambda_prior) * x_prior)),
                             bounds=(0, np.inf),
                             method=self.solver_method)
        else:
            res = lsq_linear(np.vstack((A, np.sqrt(lambda_reg) * L_mat)),
                             np.concatenate((b, np.zeros(N_tau))),
                             bounds=(0, np.inf),
                             method=self.solver_method)

        self.polished_drt = res.x
        print("   Done.")
        return self.polished_drt

    def _joint_solve(self, lambda_reg, prior, lambda_prior, warburg_tau,
                     prior_mask_tau=None):
        """One bounded solve of the RC basis + a single Warburg-open column.

        The target is the MEASURED rows only: the Warburg column is the
        low-frequency model, so the synthetic LF-extrapolated rows — which
        close the arc that an open diffusion branch keeps rising past — are
        dropped (drt-core.md 5.5, task #56). The column is excluded from both
        the smoothness and the prior regularizer; R_W is bounded >= 0. The
        row mask must mirror the _EIS_fit assembly in _nnls_system.

        ``prior_mask_tau`` zeroes x_prior beyond that tau (task #62): the
        RC-vs-Warburg split of an open diffusion arc is residual-degenerate,
        and an unmasked prior — which carries the diffusion tail — decides it
        against the Warburg column. Masked, the prior term instead penalizes
        slow-tau RC weight and the Warburg column takes the diffusion.
        """
        A, b, L_mat, x_prior, R_inf = self._nnls_system(prior)
        if x_prior is not None and prior_mask_tau is not None:
            x_prior = x_prior.copy()
            x_prior[self.log_tau > np.log10(prior_mask_tau)] = 0.0
        _f_meas_min = float(self.target_data[:, 0].min())
        _lo_mask    = self.EIS[:, 0] < _f_meas_min * 0.999
        _EIS_fit    = np.vstack([self.EIS[_lo_mask], self.target_data])
        _EIS_fit    = _EIS_fit[np.argsort(_EIS_fit[:, 0])]
        meas  = _EIS_fit[:, 0] >= _f_meas_min * 0.999
        rmask = np.concatenate([meas, meas])
        A, b  = A[rmask], b[rmask]

        w_meas = 2 * np.pi * _EIS_fit[meas, 0]
        zwo    = self._warburg_open(w_meas, 1.0, warburg_tau)
        A_aug  = np.hstack([A, np.concatenate([zwo.real, zwo.imag])[:, None]])

        N_tau = len(self.tau)
        L_aug = np.hstack([L_mat, np.zeros((N_tau, 1))])
        rows  = [A_aug, np.sqrt(lambda_reg) * L_aug]
        rhs   = [b, np.zeros(N_tau)]
        if x_prior is not None:
            I_aug = np.hstack([np.eye(N_tau), np.zeros((N_tau, 1))])
            rows.append(np.sqrt(lambda_prior) * I_aug)
            rhs.append(np.sqrt(lambda_prior) * x_prior)
        res = lsq_linear(np.vstack(rows), np.concatenate(rhs),
                         bounds=(0, np.inf), method=self.solver_method)
        resid = A_aug @ res.x - b
        return res.x[:N_tau], float(res.x[N_tau]), float(np.sum(resid ** 2)), R_inf

    def polish_spectrum_joint(self, lambda_reg: float, lambda_prior: float = 0.05,
                              tau_grid: np.ndarray = None) -> np.ndarray:
        """Joint Warburg-open + RC NNLS polish (drt-core.md 5.5, task #56).

        Two passes mirroring the standard protocol (prior = base_drt, then
        prior = first pass), each solved by _joint_solve with the prior masked
        beyond the diffusion rise detected in the prior itself (task #62; see
        _joint_solve). tau_W is nonlinear, so an outer grid picks the residual
        minimum; when the diffusion knee lies beyond the measured window the
        column degenerates to a pure capacitor and only C_lim = tau_W / R_W is
        identified — the record flags this via 'knee_resolved'. Sets
        polished_drt to the RC part and stores the branch in
        self.joint_warburg (same key set as the post-hoc fit, plus
        'knee_resolved' and 'contribution').
        """
        print("-> [Step 2] Joint NNLS polishing (RC basis + Warburg-open column)...")
        f_meas = self.target_data[:, 0]
        t_min  = 1.0 / (2 * np.pi * float(f_meas.max()))
        t_max  = 1.0 / (2 * np.pi * float(f_meas.min()))
        if tau_grid is None:
            tau_grid = np.logspace(np.log10(t_min), np.log10(t_max * 1e3), 30)

        # Prior masking (task #62, drt-core.md 5.5): run the tail-cut detector
        # on the pass-1 prior itself (base_drt on the tau grid) and zero the
        # prior beyond the detected diffusion rise in BOTH passes. Detection on
        # the prior keeps the rule self-contained, so the AUTO upgrade and a
        # directly configured JOINT run behave identically, and a prior with
        # no diffusion rise is used unmasked (no-op on clean spectra).
        x_prior_base = self._nnls_system(self.base_drt)[3]
        prior_mask_tau = self._resolve_tail_cut('auto', drt=x_prior_base)
        if prior_mask_tau is not None:
            print(f"   Prior masked beyond tau = {prior_mask_tau:.3g} s "
                  "(diffusion rise detected in the prior)")

        best = None
        tau_grid = np.asarray(tau_grid, dtype=float)
        for t_w in tau_grid:
            x1, r_w, ssr, _ = self._joint_solve(lambda_reg, self.base_drt,
                                                lambda_prior, t_w,
                                                prior_mask_tau=prior_mask_tau)
            if best is None or ssr < best[0]:
                best = (ssr, t_w, x1)
        _, tau_W, x1 = best
        x2, R_W, ssr2, R_inf = self._joint_solve(lambda_reg, x1, lambda_prior, tau_W,
                                                 prior_mask_tau=prior_mask_tau)

        self.R_inf = R_inf
        self.polished_drt = x2

        w_meas = 2 * np.pi * f_meas
        z_meas = self.target_data[:, 1] + 1j * self.target_data[:, 2]
        zwo    = R_W * self._warburg_open(w_meas, 1.0, tau_W)
        contribution = float(np.max(np.abs(zwo)) / np.max(np.abs(z_meas)))
        misfit = float(np.sqrt(ssr2) /
                       np.linalg.norm(np.concatenate([z_meas.real - R_inf, z_meas.imag])))
        # Validity precondition (drt-core.md 5.5, task #98). Where the branch is
        # not separable from the RC basis the tau_W scan returns the grid FLOOR
        # with R_W driven to ~0, and that solution satisfies `tau_W <= t_max`
        # trivially. Reporting it as a resolved knee, with a finite C_lim built
        # from two meaningless numbers, is indistinguishable in the record from
        # a measurement. Only the floor counts: the grid's upper end is where a
        # knee genuinely beyond the window comes to rest, and there C_lim is the
        # quantity to report. A diffusion knee cannot sit at the fastest
        # measured relaxation time; it can sit arbitrarily far beyond the
        # slowest.
        on_floor = bool(tau_W <= tau_grid[0])
        degenerate = bool(on_floor or contribution <= WARBURG_MIN_CONTRIBUTION)
        self.joint_warburg = {
            'model': 'warburg_open_joint', 'R_W': R_W, 'Tau_W': tau_W,
            'C_lim': None if degenerate else tau_W / max(R_W, 1e-300),
            'Freq': None if degenerate else 1.0 / (2 * np.pi * tau_W),
            'n_points': len(f_meas), 'misfit': misfit,
            'knee_resolved': False if degenerate else bool(tau_W <= t_max),
            'contribution': contribution,
        }
        if degenerate:
            print(f"   Done: R_W = {R_W:.4g} Ohm, tau_W = {tau_W:.3g} s "
                  f"(DEGENERATE: the diffusion column collapsed onto the "
                  f"{'search-grid floor' if on_floor else 'negligible-amplitude limit'}; "
                  f"no knee identified, C_lim withheld)")
        else:
            print(f"   Done: R_W = {R_W:.4g} Ohm, tau_W = {tau_W:.3g} s, "
                  f"C_lim = {self.joint_warburg['C_lim']:.3g} F "
                  f"({'knee resolved' if self.joint_warburg['knee_resolved'] else 'knee beyond window: only C_lim identified'})")
        return x2

    def _empty_result(self):
        """Shared 'no peaks found' result used by the deconvolution methods."""
        R_inf_empty = max(0.0, float(np.min(self.target_data[:, 1])))
        self.refined_ecm_params = {'R_inf': R_inf_empty, 'Peaks': []}
        self.zarc_params = None
        self.popt = np.array([])
        return self.refined_ecm_params

    def _resolve_tail_cut(self, tail_cut, drt=None):
        """Resolve the diffusion tail-cut request into a tau value (or None).

        Numeric -> manual cut at that tau. 'auto' -> detect DRT mass rising
        monotonically into the low-frequency edge of the measured window
        (2% noise tolerance; only when the rise exceeds 5% of the in-window
        maximum) and cut at the last local minimum before the rise. Anything
        else ('off') -> no cut.

        ``drt`` selects the spectrum the 'auto' detector reads (a tau-grid
        array); default is ``polished_drt``. The joint polish passes its
        pass-1 prior here, before any polished DRT exists (task #62).
        """
        if isinstance(tail_cut, (int, float)) and not isinstance(tail_cut, bool):
            return float(tail_cut)
        if tail_cut != 'auto':
            return None
        if drt is None:
            drt = self.polished_drt
        margin = WINDOW_MARGIN_DECADES
        win = ((self.log_tau >= self.meas_log_tau_min - margin) &
               (self.log_tau <= self.meas_log_tau_max + margin))
        x, y = self.log_tau[win], drt[win]
        m = len(y) - 1
        while m > 0 and y[m - 1] <= y[m] * TAIL_CUT_NOISE_FACTOR:
            m -= 1
        if m < len(y) - 1 and (y[-1] - y[m]) > TAIL_CUT_RISE_FRACTION * y.max():
            return 10 ** x[m]
        return None

    def _measured_window_slice(self):
        """The (log_tau, polished_drt) slice every deconvolution method fits over.

        It is the measured log-tau window (0.05 decade margin) with the active
        diffusion tail-cut applied. All three methods (ZARC, LASSO, CWT) build
        their dictionary / candidates / model-selection score on this same
        domain, so the BIC/AICc `N` is comparable across methods and no method
        places weight at tau values the window clamp deliberately excludes
        (task #16; see drt-core.md section 3.1).

        Applies the same <5-point tail-cut guard side effect as the ZARC path:
        a cut that would leave fewer than five points is disabled
        (`self.deconv_tail_cut_tau = None`) with a warning.
        """
        margin = WINDOW_MARGIN_DECADES
        win = ((self.log_tau >= self.meas_log_tau_min - margin) &
               (self.log_tau <= self.meas_log_tau_max + margin))
        x_fit, y_fit = self.log_tau[win], self.polished_drt[win]

        cut = self.deconv_tail_cut_tau
        if cut is not None:
            keep = x_fit <= np.log10(cut)
            if keep.sum() >= 5:
                x_fit, y_fit = x_fit[keep], y_fit[keep]
            else:
                print(f"   Warning: tail-cut at tau = {cut:.2e} s leaves "
                      f"<5 points; ignored.")
                self.deconv_tail_cut_tau = None
        return x_fit, y_fit

    def _deconv_zarc(self, max_peaks, criterion):
        """ZARC-kernel deconvolution with BIC/AICc model selection.

        Analytic ZARC (Cole-Cole) DRT kernels replace the symmetric Gaussian
        basis: one kernel reproduces a broad/depressed arc exactly
        (parameters R, log tau0, phi), so wide arcs are not split into
        several Gaussians. Same 3 parameters per peak as the Gaussian basis,
        so the BIC/AICc penalty and the curve_fit cost are directly
        comparable. The fit is clamped to the measured log-tau window and to
        the active diffusion tail-cut (self.deconv_tail_cut_tau).
        """
        delta = abs(np.mean(np.diff(self.log_tau)))
        x_fit, y_fit = self._measured_window_slice()

        pk, _ = find_peaks(y_fit, height=y_fit.max() * 0.01)
        cands = sorted(pk, key=lambda i: y_fit[i], reverse=True)
        if not cands:
            cands = [int(np.argmax(y_fit))]

        def zarc_model(x, *p):
            out = np.zeros_like(x)
            for i in range(0, len(p), 3):
                out += p[i] * self._zarc_kernel(x, p[i + 1], p[i + 2]) * delta
            return out

        N_win = len(x_fit)
        k0 = self._zarc_kernel(0.0, 0.0, 0.85)
        best_score, best_popt, best_k = np.inf, None, 1
        for k in range(1, min(max_peaks, len(cands)) + 1):
            p0, lo, hi = [], [], []
            for idx in cands[:k]:
                p0.extend([max(y_fit[idx] / (k0 * delta), 1e-12),
                           x_fit[idx], 0.85])
                lo.extend([0.0, x_fit[idx] - 0.5, 0.30])
                hi.extend([np.inf, x_fit[idx] + 0.5, 0.9999])
            try:
                popt_k, _ = curve_fit(zarc_model, x_fit, y_fit, p0=p0,
                                      bounds=(lo, hi), max_nfev=20000)
            except Exception as e:
                print(f"   Warning: ZARC curve_fit failed for k={k} ({e}); "
                      f"scoring the initial guess.")
                popt_k = np.array(p0, dtype=float)
            rss = max(np.sum((y_fit - zarc_model(x_fit, *popt_k)) ** 2), 1e-15)
            n_param = 3 * k
            if str(criterion).upper() == 'BIC':
                score = N_win * np.log(rss / N_win) + n_param * np.log(N_win)
            elif N_win - n_param - 1 > 0:
                score = (N_win * np.log(rss / N_win) + 2 * n_param
                         + (2 * n_param * (n_param + 1)) / (N_win - n_param - 1))
            else:
                score = np.inf
            if score < best_score:
                best_score, best_k, best_popt = score, k, popt_k

        print(f"   - Fitting {best_k} ZARC kernel(s) to DRT spectrum "
              f"(selected by {criterion}, window-clamped)")
        return best_popt

    def _deconv_lasso(self, max_peaks, lambda_lasso, lasso_channels):
        """Two-stage LASSO deconvolution (legacy): non-negative L1-regularised
        fit on a Gaussian dictionary, clustering, then Gaussian curve-fit.
        The dictionary and both fits are clamped to the measured window + active
        tail-cut, matching the ZARC path (task #16). Returns the popt triplets,
        or None when no candidates are found."""
        x_fit, y_fit = self._measured_window_slice()
        N_data = len(x_fit)
        mu_grid = np.linspace(x_fit.min(), x_fit.max(), lasso_channels)
        D = np.zeros((N_data, lasso_channels))
        for j in range(lasso_channels):
            D[:, j] = np.exp(-((x_fit - mu_grid[j])**2) / (2 * 0.45**2))

        A_stacked = np.vstack((D, np.sqrt(lambda_lasso) * np.eye(lasso_channels)))
        b_stacked = np.concatenate((y_fit, np.zeros(lasso_channels)))
        res_lsq = lsq_linear(A_stacked, b_stacked, bounds=(0, np.inf), method='bvls')
        x_lasso = res_lsq.x

        # Threshold basis peaks and merge local clusters
        threshold = x_lasso.max() * LASSO_ATOM_THRESHOLD_FRACTION
        active_idx = np.where(x_lasso > threshold)[0]
        clusters, current_cluster = [], []
        for idx in sorted(active_idx):
            if not current_cluster or (mu_grid[idx] - mu_grid[current_cluster[-1]]) <= LASSO_CLUSTER_DECADES:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        if current_cluster:
            clusters.append(current_cluster)

        peak_guides = [{'mu': mu_grid[c[np.argmax(x_lasso[c])]],
                        'amp': x_lasso[c[np.argmax(x_lasso[c])]]} for c in clusters]

        # The tail-cut is already applied by the window slice; sort candidates by
        # intensity and cap at max_peaks.
        peak_guides = sorted(peak_guides, key=lambda pg: pg['amp'], reverse=True)[:max_peaks]
        if not peak_guides:
            print("   Warning: no candidate peaks found via LASSO.")
            return None

        initial_guess, bounds_lower, bounds_upper = [], [], []
        for pg in peak_guides:
            initial_guess.extend([pg['amp'], pg['mu'], 0.45])
            bounds_lower.extend([0.0, pg['mu'] - 0.4, 0.01])
            bounds_upper.extend([np.inf, pg['mu'] + 0.4, 5.0])
        try:
            popt, _ = curve_fit(self._gaussian_mix, x_fit, y_fit,
                                p0=initial_guess, bounds=(bounds_lower, bounds_upper),
                                max_nfev=20000)
        except Exception as e:
            print(f"   Warning: Stage 2 curve_fit failed ({e}); using LASSO guides.")
            popt = np.array(initial_guess, dtype=float)
        print(f"   - Fitting {len(peak_guides)} Gaussian peaks to DRT spectrum "
              f"(selected by LASSO + Curve-Fit, window-clamped)...")
        return popt

    def _deconv_cwt(self, max_peaks, criterion):
        """CWT candidate detection + BIC/AICc incremental selection (legacy).
        Detection, the information-criterion loop and the curve-fit all run on
        the measured window + active tail-cut slice, matching the ZARC path
        (task #16). Returns the popt triplets, or None when no candidates are
        found."""
        x_fit, y_fit = self._measured_window_slice()
        N_data = len(x_fit)

        # CWT-based high-sensitivity peak detection (on the clamped slice).
        drt_norm = y_fit / (y_fit.max() + 1e-10) * 1000
        log_y = np.log10(np.maximum(1, drt_norm))
        # CWT widths in *physical* log-tau units (decades), converted to the
        # current grid's index scale so detection is invariant to how many
        # points the extrapolation/interpolation produced.
        pts_per_decade = (len(x_fit) - 1) / max(abs(x_fit[-1] - x_fit[0]), 1e-9)
        cwt_widths = np.maximum(1, np.arange(1, 6) / 5.0 * pts_per_decade).astype(int)
        raw_peak_indices = find_peaks_cwt(log_y, cwt_widths)
        filtered_indices = [i for i in raw_peak_indices if log_y[i] > 0.05]

        # Cluster peaks closer than a fixed physical separation (in decades).
        merge_decades = CWT_CLUSTER_DECADES
        min_idx_dist = max(1, int(round(merge_decades * pts_per_decade)))
        clusters, current_cluster = [], []
        for idx in sorted(filtered_indices):
            if not current_cluster or idx - current_cluster[-1] <= min_idx_dist:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        if current_cluster:
            clusters.append(current_cluster)

        # Indices are local to the clamped slice; the tail-cut is already applied.
        final_peak_indices = [c[np.argmax(log_y[c])] for c in clusters]
        candidate_peaks = sorted(final_peak_indices,
                                 key=lambda idx: y_fit[idx], reverse=True)
        if not candidate_peaks:
            print("   Warning: no candidate peaks found.")
            return None

        # BIC/AICc information-criterion selection loop
        best_k, best_score, best_popt = 1, np.inf, None
        for k in range(1, min(max_peaks, len(candidate_peaks)) + 1):
            initial_guess, bounds_lower, bounds_upper = [], [], []
            for idx in candidate_peaks[:k]:
                A_init, mu_init = y_fit[idx], x_fit[idx]
                initial_guess.extend([A_init, mu_init, 0.5])
                bounds_lower.extend([0.0, mu_init - 0.4, 0.01])
                bounds_upper.extend([np.inf, mu_init + 0.4, 5.0])
            try:
                popt_k, _ = curve_fit(self._gaussian_mix, x_fit, y_fit,
                                      p0=initial_guess, bounds=(bounds_lower, bounds_upper),
                                      max_nfev=20000)
            except Exception as e:
                print(f"   Warning: CWT Gaussian curve_fit failed for k={k} "
                      f"({e}); scoring the initial guess.")
                popt_k = np.array(initial_guess, dtype=float)
            # RSS in the DRT domain, over the clamped window
            rss = max(np.sum((y_fit - self._gaussian_mix(x_fit, *popt_k)) ** 2), 1e-15)
            n_param = 3 * k
            if str(criterion).upper() == 'BIC':
                score = N_data * np.log(rss / N_data) + n_param * np.log(N_data)
            elif N_data - n_param - 1 > 0:
                score = (N_data * np.log(rss / N_data) + 2 * n_param
                         + (2 * n_param * (n_param + 1)) / (N_data - n_param - 1))
            else:
                score = np.inf
            if score < best_score:
                best_score, best_k, best_popt = score, k, popt_k

        print(f"   - Fitting {best_k} Gaussian peaks to DRT spectrum "
              f"(selected by {criterion}, window-clamped)...")
        return best_popt

    @staticmethod
    def _gaussian_mix(x, *params):
        """Sum of Gaussian peaks used for DRT curve fitting."""
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp, mu, sigma = params[i], params[i+1], params[i+2]
            y += amp * np.exp(-((x - mu)**2) / (2 * sigma**2))
        return y

    @staticmethod
    def _zarc_kernel(x, x0, phi):
        """Unit-area (in log10 tau) analytic DRT of a ZARC element.

        For Z = R / (1 + (j*w*tau0)^phi) the exact DRT is
            gamma(ln tau) = (1/2pi) sin(phi*pi) / (cosh(phi*ln(tau/tau0)) + cos(phi*pi))
        expressed here per log10-tau unit (hence the ln(10) factor). phi
        controls the width/depression of the arc: phi -> 1 is a sharp ideal RC
        peak, smaller phi gives the broad heavy-tailed shape that a symmetric
        Gaussian basis can only mimic by splitting into several peaks.
        """
        phi = np.clip(phi, 0.05, 0.9999)
        ln10 = np.log(10.0)
        u = phi * ln10 * (x - x0)
        return (ln10 / (2 * np.pi)) * np.sin(phi * np.pi) / (np.cosh(u) + np.cos(phi * np.pi))

    @staticmethod
    def peak_curve(log_tau, p):
        """DRT curve of one extracted peak on the given log-tau grid.

        Uses the ZARC kernel when the peak came from the ZARC-basis
        deconvolution ('Phi_deconv' present), else the Gaussian shape. Shared
        by the GUI, the exporter and the batch runner so every consumer draws
        the same curve that was actually fitted.
        """
        if 'Phi_deconv' in p:
            k  = DRTOptimizer._zarc_kernel(log_tau, p['Mu'], p['Phi_deconv'])
            k0 = DRTOptimizer._zarc_kernel(p['Mu'], p['Mu'], p['Phi_deconv'])
            return p['Amplitude'] * k / k0
        return p['Amplitude'] * np.exp(
            -((log_tau - p['Mu']) ** 2) / (2 * p['Sigma'] ** 2))

    def deconvolve_peaks(self, max_peaks: int = 10, criterion: str = 'BIC',
                               manual_peaks: list = None,
                               method: str = 'ZARC + BIC',
                               lambda_lasso: float = 0.003,
                               lasso_channels: int = 100,
                               tail_cut='auto',
                               warburg='auto') -> dict:
        """tail_cut controls the diffusion tail-cut: 'auto' (default) detects
        a monotonic DRT rise into the low-frequency window edge, 'off'
        disables the cut, and a number is a manual cut in seconds. The cut
        applies to every deconvolution method (the region tau > cut is
        excluded from peak identification), and fit_zarc_elements excludes
        the same frequencies from the impedance-domain refinement. In
        manual_peaks mode only a numeric cut is honoured (no detection).

        warburg controls the Warburg-open fit of the excluded region
        (docs/specs/drt-deconvolution.md section 2.4): 'auto' (default) fits a
        finite-length Warburg-open element to the residual of the tail-cut
        region after refinement, 'off' disables the fit. The fit runs strictly
        after every conformance-pinned quantity is produced, so it never moves
        them."""
        if warburg not in ('auto', 'off'):
            raise ValueError(
                f"warburg must be 'auto' or 'off', got {warburg!r}.")
        # Reset per-run so a re-run on the same optimizer cannot leak a stale
        # diffusion element into the reconstruction.
        self.warburg_params = None
        self.r_budget = None
        print(f"\n-> [DRT peak deconvolution] Extracting peaks from DRT spectrum "
              f"(method = {method if manual_peaks is None else 'manual'})...")

        raw_re = self.target_data[:, 1]
        basis = 'gauss'   # parameterization of popt triplets; 'zarc' -> (R, mu, phi)
        # Tau above which the DRT is treated as a diffusion branch. Shared by
        # all deconvolution methods and by fit_zarc_elements (refinement mask).
        if manual_peaks is not None:
            self.deconv_tail_cut_tau = (
                float(tail_cut)
                if isinstance(tail_cut, (int, float)) and not isinstance(tail_cut, bool)
                else None)
        else:
            self.deconv_tail_cut_tau = self._resolve_tail_cut(tail_cut)
            if self.deconv_tail_cut_tau is not None:
                if isinstance(tail_cut, (int, float)) and not isinstance(tail_cut, bool):
                    print(f"   - Manual tail-cut: fit window truncated "
                          f"at tau = {self.deconv_tail_cut_tau:.2e} s")
                else:
                    print(f"   - Diffusion tail detected: fit window truncated "
                          f"at tau = {self.deconv_tail_cut_tau:.2e} s")

        if manual_peaks is not None:
            active_peaks = sorted(manual_peaks)
            print(f"   - Using {len(active_peaks)} manually specified peak indices.")
            # ZARC-kernel mixture, same basis as the automatic path (task #81,
            # drt-deconvolution.md 4.3). No model selection; the fit domain is
            # the full tau grid because a manually placed peak may sit outside
            # the measured window.
            delta = abs(np.mean(np.diff(self.log_tau)))
            k0 = self._zarc_kernel(0.0, 0.0, 0.85)

            def zarc_model(x, *p):
                out = np.zeros_like(x)
                for i in range(0, len(p), 3):
                    out += p[i] * self._zarc_kernel(x, p[i + 1], p[i + 2]) * delta
                return out

            initial_guess, bounds_lower, bounds_upper = [], [], []
            for idx in active_peaks:
                A_init, mu_init = self.polished_drt[idx], self.log_tau[idx]
                initial_guess.extend([max(A_init / (k0 * delta), 1e-12),
                                      mu_init, 0.85])
                bounds_lower.extend([0.0, mu_init - 0.4, 0.30])
                bounds_upper.extend([np.inf, mu_init + 0.4, 0.9999])
            try:
                popt, _ = curve_fit(zarc_model, self.log_tau, self.polished_drt,
                                    p0=initial_guess, bounds=(bounds_lower, bounds_upper),
                                    max_nfev=20000)
            except Exception as e:
                print(f"   Warning: ZARC fit failed ({e}); using initial guess.")
                popt = np.array(initial_guess, dtype=float)
            basis = 'zarc'
        elif "ZARC" in method:
            popt = self._deconv_zarc(max_peaks, criterion)
            basis = 'zarc'
        elif "LASSO" in method:
            popt = self._deconv_lasso(max_peaks, lambda_lasso, lasso_channels)
            if popt is None:
                return self._empty_result()
        else:
            popt = self._deconv_cwt(max_peaks, criterion)
            if popt is None:
                return self._empty_result()

        self.popt = popt
        # Use the same R_inf as the polished DRT so ECM parameters are
        # consistent with the NNLS reconstruction.
        R_inf_final = (max(0.0, self.R_inf) if self.R_inf is not None
                       else max(0.0, float(np.min(raw_re))))

        # R = (continuous Gaussian area) / Δlog_tau to match the discrete DRT grid
        delta_log_tau = abs(np.mean(np.diff(self.log_tau)))

        # Build candidate peaks, then drop ones whose resistance is negligible
        # relative to the strongest peak. These tiny satellite peaks sit near the
        # measurement-window edges and otherwise flicker in/out with the
        # extrapolation range, making the ECM peak count range-dependent.
        candidates = []
        for i in range(0, len(popt), 3):
            if basis == 'zarc':
                # ZARC triplet is (R, mu, phi): R is the element resistance
                # directly (the kernel has unit area). Store equivalent
                # Gaussian Amplitude/Sigma (kernel height / FWHM-derived) so
                # legacy consumers of those fields keep working.
                R_val, mu, phi_d = popt[i], popt[i+1], popt[i+2]
                Tau_val = 10 ** mu
                amp = R_val * self._zarc_kernel(mu, mu, phi_d) * delta_log_tau
                half = (np.arccosh(2.0 + np.cos(np.pi * phi_d))
                        / (phi_d * np.log(10.0)))
                sigma = 2.0 * half / 2.3548
                candidates.append({'R': R_val, 'Tau': Tau_val, 'Sigma': sigma,
                                   'Amplitude': amp, 'Mu': mu,
                                   'Phi_deconv': phi_d,
                                   'Freq': 1.0 / (2 * np.pi * Tau_val)})
                continue
            amp, mu, sigma = popt[i], popt[i+1], popt[i+2]
            R_val   = amp * sigma * np.sqrt(2 * np.pi) / delta_log_tau
            Tau_val = 10 ** mu
            candidates.append({'R': R_val, 'Tau': Tau_val, 'Sigma': sigma,
                               'Amplitude': amp, 'Mu': mu,
                               'Freq': 1.0 / (2 * np.pi * Tau_val)})

        rel_R_threshold = NEGLIGIBLE_PEAK_FRACTION   # keep peaks with R >= this fraction of the largest R
        R_max = max((c['R'] for c in candidates), default=0.0)
        kept  = [c for c in candidates if c['R'] >= rel_R_threshold * R_max] if R_max > 0 else []
        n_drop = len(candidates) - len(kept)
        if n_drop:
            print(f"   - Dropped {n_drop} negligible peak(s) (R < {rel_R_threshold:.0%} of R_max).")

        kept.sort(key=lambda c: c['Mu'])
        refined_params = {'R_inf': R_inf_final, 'Peaks': []}
        for k, c in enumerate(kept):
            c['Peak'] = k + 1
            refined_params['Peaks'].append(c)

        # Editable peak list (grid indices) for the GUI's manual edit mode:
        # user-supplied indices are kept verbatim; otherwise use the indices
        # closest to the extracted peak centers.
        if manual_peaks is not None:
            self.active_peaks = sorted(int(i) for i in manual_peaks)
        else:
            self.active_peaks = [int(np.argmin(np.abs(self.log_tau - c['Mu'])))
                                 for c in kept]

        self.refined_ecm_params = refined_params
        self.popt               = popt
        print("-> [Done] DRT deconvolution and ECM parameter extraction complete.")

        # Stage 2: refine each peak into a ZARC (R-CPE) element so broad/depressed
        # arcs are captured by a CPE exponent phi instead of being smeared across
        # many RC bins. Uses the Gaussian peaks above as initial guesses.
        self.fit_zarc_elements()

        # Stage 2b: misfit-rescue (drt-deconvolution.md 7.3, task #84). Only
        # in automatic mode - a manual peak count is the user's decision.
        if manual_peaks is None:
            self._misfit_rescue()

        # R-budget diagnostic (task #87): the reported elements' total R
        # against the NNLS mass inside the deconvolution window - the region
        # the peaks were fitted to. A large mismatch means the circuit and
        # the DRT disagree on how much impedance the window carries,
        # typically a bound-limited element absorbing impedance beyond the
        # measured window (task #83) or an unmodelled shoulder.
        _x_bud, _y_bud = self._measured_window_slice()
        _mass = float(np.sum(_y_bud))
        _sum_R = float(sum(p['R'] for p in refined_params['Peaks']))
        if _mass > 0 and refined_params['Peaks']:
            _ratio = _sum_R / _mass
            self.r_budget = {'sum_R': _sum_R, 'nnls_mass': _mass,
                             'ratio': _ratio}
            print(f"   - R-budget: reported elements carry {_ratio:.1%} of "
                  f"the windowed NNLS mass "
                  f"(sum R = {_sum_R:.4g}, window mass = {_mass:.4g})")
            if abs(_ratio - 1.0) > R_BUDGET_TOL:
                print(f"   Warning: R-budget ratio {_ratio:.2f} is outside "
                      f"[{1 - R_BUDGET_TOL:.2f}, {1 + R_BUDGET_TOL:.2f}]: "
                      f"the reported circuit and the windowed NNLS mass "
                      f"disagree - typically a bound-limited element "
                      f"absorbing impedance beyond the measured window, or "
                      f"an unmodelled shoulder.")

        # Stage 3: model the diffusion branch the tail-cut excluded. Runs after
        # refinement on the residual of the excluded points only, so every
        # quantity pinned by drt-conformance.md section 3 is already final.
        if warburg == 'auto':
            wb = self.fit_warburg_tail()
            if wb is not None:
                refined_params['Warburg'] = wb
        return refined_params

    @staticmethod
    def _zarc_sum(w, R_inf, params):
        """
        Series of ZARC (R parallel CPE) elements plus an ohmic R_inf.

            Z(w) = R_inf + sum_k  R_k / (1 + (j*w*tau_k)**phi_k)

        params : flat list [R_1, log_tau_1, phi_1, R_2, log_tau_2, phi_2, ...].
        phi = 1 reduces each element to an ideal RC (Debye) semicircle; phi < 1
        gives a broad/depressed arc.
        """
        Z = np.full_like(w, R_inf, dtype=complex)
        for i in range(0, len(params), 3):
            R, log_tau, phi = params[i], params[i + 1], params[i + 2]
            tau = 10.0 ** log_tau
            Z += R / (1.0 + (1j * w * tau) ** phi)
        return Z

    def fit_zarc_elements(self):
        """
        Refine the extracted peaks into ZARC (R-CPE) elements by fitting the
        measured impedance directly:  Z = R_inf + sum R_k/(1+(j w tau_k)^phi_k).

        Adds 'Phi' (CPE exponent) and 'CPE_Q' to each peak in
        self.refined_ecm_params, and stores the per-element results in
        self.zarc_params. Falls back gracefully (phi = 1) if the fit fails.
        """
        ref = getattr(self, 'refined_ecm_params', None)
        if not ref or not ref.get('Peaks'):
            self.zarc_params = None
            return None

        peaks  = ref['Peaks']
        w_meas = 2 * np.pi * self.target_data[:, 0]
        Z_re   = self.target_data[:, 1]
        Z_im   = self.target_data[:, 2]
        R_inf  = ref['R_inf']

        # Joint mode (task #63, drt-deconvolution.md 7): the diffusion branch
        # is carried by the jointly fitted Warburg column (drt-core.md 5.5),
        # so subtract it from the refinement target - otherwise the ZARC
        # elements are forced to absorb the diffusion resistance the NNLS
        # already assigned to that column, reporting the same physics twice.
        if self.joint_warburg is not None:
            zwo = self.joint_warburg['R_W'] * self._warburg_open(
                w_meas, 1.0, self.joint_warburg['Tau_W'])
            Z_re = Z_re - zwo.real
            Z_im = Z_im - zwo.imag
            print("   - ZARC refinement target: measured impedance minus the "
                  "joint Warburg branch")

        # When the deconvolution stage cut off a diffusion tail, exclude the
        # same region (f < 1/(2*pi*tau_cut)) from this refinement too: a
        # diffusion branch is not representable by ZARC elements, and fitting
        # across it drags R/tau/phi away from the relaxation arcs.
        cut_tau = getattr(self, 'deconv_tail_cut_tau', None)
        if cut_tau:
            keep = w_meas >= 1.0 / cut_tau
            if keep.sum() >= 8:   # keep enough points for a meaningful fit
                n_excl = int((~keep).sum())
                w_meas, Z_re, Z_im = w_meas[keep], Z_re[keep], Z_im[keep]
                print(f"   - ZARC refinement excludes {n_excl} low-frequency "
                      f"point(s) below the diffusion cut (tau > {cut_tau:.2e} s)")
            else:
                print(f"   Warning: diffusion cut at tau = {cut_tau:.2e} s would "
                      f"leave <8 points; refinement uses the FULL spectrum "
                      f"(deconvolution and refinement windows now differ).")

        # Initial guess / bounds from the deconvolution peaks. When the peak
        # came from the ZARC-basis deconvolution, its phi is already a good
        # estimate of the CPE exponent, so start from it.
        p0, lo, hi = [], [], []
        for p in peaks:
            phi0 = min(0.999, max(0.31, p.get('Phi_deconv', 0.85)))
            p0 += [max(p['R'], 1e-12), p['Mu'], phi0]
            lo += [0.0,            p['Mu'] - 1.0, 0.3]
            hi += [np.inf,         p['Mu'] + 1.0, 1.0]

        def model_ri(w, *params):
            Z = self._zarc_sum(w, R_inf, list(params))
            return np.concatenate((Z.real, Z.imag))

        y = np.concatenate((Z_re, Z_im))
        fit_ok = True
        try:
            popt, _ = curve_fit(model_ri, w_meas, y, p0=p0,
                                bounds=(lo, hi), maxfev=40000)
        except (RuntimeError, ValueError) as e:
            print(f"   Warning: ZARC refinement failed ({e}); keeping phi=1 (RC).")
            fit_ok = False
            popt = np.array(p0, dtype=float)
            for i in range(2, len(popt), 3):
                popt[i] = 1.0

        # Flag parameters the optimizer pushed onto an ARBITRARY fit bound
        # (task #83): a bound-saturated value is set by the box, not the data -
        # typically an element left under-determined by an open low-frequency
        # arc, whose R-tau ridge lets tau run until the bound stops it. The
        # phi upper bound 1.0 is deliberately NOT flagged: it is the physical
        # Debye limit, and a near-ideal RC element legitimately sits there.
        if fit_ok:
            for k in range(0, len(popt), 3):
                parts = []
                mu = popt[k + 1]
                if mu - lo[k + 1] < 1e-3:
                    parts.append("tau at the -1.0-decade bound from its seed")
                elif hi[k + 1] - mu < 1e-3:
                    parts.append("tau at the +1.0-decade bound from its seed")
                if popt[k + 2] - lo[k + 2] < 1e-4:
                    parts.append(f"phi at its lower bound {lo[k + 2]:.1f}")
                if parts:
                    print(f"   Warning: refined element {k // 3 + 1} is "
                          f"bound-limited ({'; '.join(parts)}): the reported "
                          f"value sits on the fit bound, not a data-determined "
                          f"optimum.")

        # Fit quality (modulus-weighted relative residual over measured window).
        Zf  = self._zarc_sum(w_meas, R_inf, list(popt))
        res = np.linalg.norm(np.concatenate((Zf.real - Z_re, Zf.imag - Z_im)))
        misfit = res / (np.linalg.norm(y) + 1e-30)

        zarc = {'R_inf': R_inf, 'misfit': misfit, 'Elements': []}
        for k in range(0, len(popt), 3):
            R, log_tau, phi = popt[k], popt[k + 1], popt[k + 2]
            tau = 10.0 ** log_tau
            # CPE admittance prefactor Q from tau = (R*Q)^(1/phi)  ->  Q = tau^phi / R
            Q = (tau ** phi) / R if R > 0 else 0.0
            # Brug effective capacitance: C_eff = Q^(1/phi) * R^(1/phi - 1) = tau / R.
            # For phi = 1 this collapses to the ideal capacitance C = tau / R.
            C_eff = (Q ** (1.0 / phi)) * (R ** (1.0 / phi - 1.0)) if (R > 0 and phi > 0) else 0.0
            idx = k // 3
            elem = {'Peak': idx + 1, 'R': R, 'Tau': tau, 'Phi': phi,
                    'CPE_Q': Q, 'C_eff': C_eff, 'Freq': 1.0 / (2 * np.pi * tau)}
            zarc['Elements'].append(elem)
            # Promote the ZARC-refined values onto the reported peak so that R,
            # Phi, CPE_Q, Tau and C_eff are all mutually consistent (taken from
            # the same R-CPE element). Keep the Gaussian-stage values separately.
            if idx < len(peaks):
                peaks[idx]['R_gauss']   = peaks[idx]['R']
                peaks[idx]['Tau_gauss'] = peaks[idx]['Tau']
                peaks[idx]['R']      = R
                peaks[idx]['Tau']    = tau
                peaks[idx]['Freq']   = 1.0 / (2 * np.pi * tau)
                peaks[idx]['Phi']    = phi
                peaks[idx]['CPE_Q']  = Q
                peaks[idx]['C_eff']  = C_eff

        self.zarc_params = zarc
        phis = ', '.join(f"{e['Phi']:.3f}" for e in zarc['Elements'])
        print(f"-> [ZARC] R-CPE refinement: phi = [{phis}], misfit = {misfit:.3f}")
        return zarc

    def _misfit_rescue(self):
        """One-shot misfit-rescue stage (drt-deconvolution.md 7.3, task #84).

        A broad DRT shoulder without a local maximum cannot become a
        find_peaks candidate and barely moves the DRT-domain model-selection
        criterion, yet can carry percent-level measured impedance. When the
        R-CPE refinement has visibly failed (misfit > RESCUE_MISFIT_GATE),
        seed one ZARC element at the largest smoothed DRT residual away from
        every existing peak, re-refine with k+1 elements, and adopt the
        result only on a >= RESCUE_MIN_IMPROVEMENT misfit improvement with a
        non-negligible rescued element. Otherwise restore everything.
        """
        zp = self.zarc_params
        ref = getattr(self, 'refined_ecm_params', None)
        if not zp or not ref or not ref.get('Peaks'):
            return
        old_misfit = zp['misfit']
        if old_misfit <= RESCUE_MISFIT_GATE:
            return
        peaks = ref['Peaks']

        # Positive DRT residual of the deconvolution-stage model over the fit
        # window, smoothed, with the neighbourhood of every existing peak
        # zeroed so the seed lands between peaks (drt-deconvolution.md 7.3
        # step 2: near-peak kernel mismatch would otherwise split the arc).
        x_fit, y_fit = self._measured_window_slice()
        total = np.zeros_like(x_fit)
        for p in peaks:
            total += self.peak_curve(x_fit, p)
        resid = np.maximum(y_fit - total, 0.0)
        if len(resid) >= 5:
            resid = np.convolve(resid, np.ones(5) / 5.0, mode='same')
        for p in peaks:
            resid[np.abs(x_fit - p['Mu']) < RESCUE_EXCLUSION_DECADES] = 0.0
        if not np.any(resid > 0):
            return
        i_seed = int(np.argmax(resid))
        mu_seed = float(x_fit[i_seed])
        print(f"-> [Rescue] R-CPE misfit {old_misfit:.4f} exceeds "
              f"{RESCUE_MISFIT_GATE}: testing one residual-seeded element "
              f"near tau = {10 ** mu_seed:.2e} s")

        # DRT-domain kernel fit of the residual under the manual-mode bounds
        # (drt-deconvolution.md 4.3) to obtain a full peak record.
        delta = abs(np.mean(np.diff(self.log_tau)))
        k0 = self._zarc_kernel(0.0, 0.0, 0.85)

        def one_zarc(x, R, mu, phi):
            return R * self._zarc_kernel(x, mu, phi) * delta

        try:
            popt1, _ = curve_fit(
                one_zarc, x_fit, resid,
                p0=[max(resid[i_seed] / (k0 * delta), 1e-12), mu_seed, 0.85],
                bounds=([0.0, mu_seed - 0.4, 0.30],
                        [np.inf, mu_seed + 0.4, 0.9999]),
                max_nfev=20000)
        except Exception as e:
            print(f"   Warning: rescue kernel fit failed ({e}); keeping the "
                  f"{len(peaks)}-element model.")
            return
        R_d, mu_d, phi_d = (float(v) for v in popt1)
        tau_d = 10.0 ** mu_d
        amp = R_d * self._zarc_kernel(mu_d, mu_d, phi_d) * delta
        half = (np.arccosh(2.0 + np.cos(np.pi * phi_d))
                / (phi_d * np.log(10.0)))
        cand = {'R': R_d, 'Tau': tau_d, 'Sigma': 2.0 * half / 2.3548,
                'Amplitude': amp, 'Mu': mu_d, 'Phi_deconv': phi_d,
                'Freq': 1.0 / (2 * np.pi * tau_d)}

        # Snapshot, then re-refine with k+1 deconvolution-stage seeds (reset
        # the promoted values so the second refinement starts clean).
        snap_peaks = [dict(p) for p in peaks]
        snap_zarc = zp
        for p in peaks:
            if 'R_gauss' in p:
                p['R'], p['Tau'] = p['R_gauss'], p['Tau_gauss']
        peaks.append(cand)
        peaks.sort(key=lambda c: c['Mu'])
        for i, p in enumerate(peaks):
            p['Peak'] = i + 1
        self.fit_zarc_elements()
        new_misfit = (self.zarc_params['misfit']
                      if self.zarc_params else np.inf)

        improved = (new_misfit > 0
                    and old_misfit / new_misfit >= RESCUE_MIN_IMPROVEMENT)
        R_max = max(p['R'] for p in peaks)
        significant = cand['R'] >= NEGLIGIBLE_PEAK_FRACTION * R_max
        if improved and significant:
            self.active_peaks = [int(np.argmin(np.abs(self.log_tau - p['Mu'])))
                                 for p in peaks]
            print(f"-> [Rescue] Adopted: misfit {old_misfit:.4f} -> "
                  f"{new_misfit:.4f} ({old_misfit / new_misfit:.1f}x); "
                  f"peak count {len(snap_peaks)} -> {len(peaks)}.")
        else:
            reason = ('improvement below threshold' if not improved
                      else 'rescued element negligible')
            peaks[:] = snap_peaks
            self.zarc_params = snap_zarc
            print(f"-> [Rescue] Rejected ({reason}: misfit {old_misfit:.4f} "
                  f"-> {new_misfit:.4f}); keeping the "
                  f"{len(snap_peaks)}-element model.")

    @staticmethod
    def _warburg_open(w, R_W, tau_W):
        """
        Finite-length Warburg element with a reflective (open-circuit) boundary:

            Z_WO(w) = R_W * coth(sqrt(j*w*tau_W)) / sqrt(j*w*tau_W)

        High frequency: the 45-degree Warburg line R_W / sqrt(j*w*tau_W).
        Low frequency:  R_W/3 + 1/(j*w*C_lim) with C_lim = tau_W / R_W — the
        blocked-diffusion (capacitive) signature of battery spectra.

        coth(s)/s is evaluated as 1/(s*tanh(s)); for |s| > 30 the coth -> 1
        asymptote is used so cosh/sinh cannot overflow on the extended
        reconstruction grid (frequencies up to ~1e8 Hz).
        """
        w = np.asarray(w, dtype=float)
        s = np.sqrt(1j * w * tau_W)
        Z = np.empty_like(s)
        big = np.abs(s) > 30.0
        Z[big] = R_W / s[big]
        sb = s[~big]
        Z[~big] = R_W / (sb * np.tanh(sb))
        return Z

    def fit_warburg_tail(self):
        """
        Fit a Warburg-open element to the low-frequency region excluded by the
        diffusion tail-cut (docs/specs/drt-deconvolution.md section 2.4).

        The fit target is the residual Z_meas - Z_ECM on the excluded points
        only, with the refined relaxation model held fixed, so nothing pinned
        by the conformance contract can move. Returns the parameter dict (also
        stored as self.warburg_params) or None when inactive or failed.
        """
        self.warburg_params = None
        cut_tau = getattr(self, 'deconv_tail_cut_tau', None)
        zarc = getattr(self, 'zarc_params', None)
        if not cut_tau or not zarc or not zarc.get('Elements'):
            return None

        w_all = 2 * np.pi * self.target_data[:, 0]
        lf = w_all < 1.0 / cut_tau
        n_lf = int(lf.sum())
        if n_lf < 4:
            print(f"   - [Warburg] Only {n_lf} point(s) below the diffusion cut; "
                  f"need >= 4. Skipping the Warburg-open fit.")
            return None

        w_lf = w_all[lf]
        Z_meas = self.target_data[lf, 1] + 1j * self.target_data[lf, 2]
        params = []
        for e in zarc['Elements']:
            params += [e['R'], np.log10(e['Tau']), e['Phi']]
        Z_res = Z_meas - self._zarc_sum(w_lf, zarc['R_inf'], params)

        # Initial guess from the lowest-frequency residual point: the LF limit
        # is capacitive with C_lim = tau_W/R_W, and the real span approaches
        # R_W/3.
        i0 = int(np.argmin(w_lf))
        R0 = 3.0 * max(float(np.max(Z_res.real)), 1e-9)
        if Z_res.imag[i0] < 0:
            C0 = -1.0 / (w_lf[i0] * Z_res.imag[i0])
            tau0 = max(R0 * C0, cut_tau)
        else:
            tau0 = 10.0 * cut_tau

        def model(w_, lgR, lgT):
            Z = self._warburg_open(w_, 10.0 ** lgR, 10.0 ** lgT)
            return np.concatenate((Z.real, Z.imag))

        y = np.concatenate((Z_res.real, Z_res.imag))
        p0 = [np.log10(R0), np.log10(tau0)]
        lo = [p0[0] - 6.0, np.log10(cut_tau) - 3.0]
        hi = [p0[0] + 6.0, np.log10(cut_tau) + 8.0]
        try:
            popt, _ = curve_fit(model, w_lf, y, p0=p0,
                                bounds=(lo, hi), maxfev=20000)
        except (RuntimeError, ValueError) as e:
            print(f"   Warning: Warburg-open fit failed ({e}); "
                  f"diffusion branch left unmodelled.")
            return None

        # A converged point pinned to a box boundary is a degenerate fit (the
        # residual is not Warburg-shaped, e.g. the relaxation model already
        # absorbed the branch). Report failure rather than a plausible-looking
        # element — see interface-design-standards rule 1.
        if any(abs(popt[i] - b) < 1e-6 for i in (0, 1) for b in (lo[i], hi[i])):
            print("   Warning: Warburg-open fit converged onto a parameter "
                  "bound (degenerate); diffusion branch left unmodelled.")
            return None

        R_W, tau_W = 10.0 ** popt[0], 10.0 ** popt[1]
        misfit = (np.linalg.norm(model(w_lf, *popt) - y)
                  / (np.linalg.norm(y) + 1e-30))
        self.warburg_params = {
            'model': 'warburg_open', 'R_W': R_W, 'Tau_W': tau_W,
            'C_lim': tau_W / R_W, 'Freq': 1.0 / (2 * np.pi * tau_W),
            'misfit': misfit, 'n_points': n_lf,
        }
        print(f"-> [Warburg] Diffusion branch fitted (open boundary): "
              f"R_W = {R_W:.4g} Ohm, tau_W = {tau_W:.3g} s, "
              f"C_lim = {tau_W / R_W:.3g} F, misfit = {misfit:.3f} "
              f"({n_lf} LF points)")
        return self.warburg_params

    def fit_diffusion_composite(self):
        """Composite diffusion-branch refit (drt-deconvolution.md 2.4.2,
        task #89): with the reported relaxation elements held fixed, co-fit
        one slow ZARC plus one Warburg-open against the full-spectrum
        residual. The excluded region can hide a slow near-RC relaxation
        overlapping the diffusion (LIB_data_2's analogue of LIB_data's
        tau ~ 30 s element); the full spectrum identifies it through its
        leading tail inside the kept window. Adopted only under the guards
        of section 2.4.2; on any failure the single-element record of the
        section 2.4.1 refit is kept and None is returned.
        """
        single = self.warburg_params
        cut_tau = getattr(self, 'deconv_tail_cut_tau', None)
        zarc = getattr(self, 'zarc_params', None)
        if not single or not cut_tau or not zarc or not zarc.get('Elements'):
            return None

        w = 2 * np.pi * self.target_data[:, 0]
        Zm = self.target_data[:, 1] + 1j * self.target_data[:, 2]
        params = []
        for e in zarc['Elements']:
            params += [e['R'], np.log10(e['Tau']), e['Phi']]
        D = Zm - self._zarc_sum(w, zarc['R_inf'], params)
        y = np.concatenate((D.real, D.imag))

        # Baseline: the single-element refit evaluated on the same target.
        z1 = self._warburg_open(w, single['R_W'], single['Tau_W'])
        mis1 = (np.linalg.norm(np.concatenate((z1.real, z1.imag)) - y)
                / (np.linalg.norm(y) + 1e-30))

        def model(wv, R_s, lgT, phi, lgRW, lgTW):
            Z = (R_s / (1.0 + (1j * wv * 10.0 ** lgT) ** phi)
                 + (10.0 ** lgRW) * self._warburg_open(wv, 1.0, 10.0 ** lgTW))
            return np.concatenate((Z.real, Z.imag))

        lo = [0.0, np.log10(cut_tau), 0.3,
              np.log10(single['R_W']) - 4.0, 0.0]
        hi = [np.inf, 3.0, 1.0,
              np.log10(single['R_W']) + 2.0, 6.0]
        sols = []
        for t_s in (1.0, 10.0, 100.0):
            for scale in (0.3, 1.0, 3.0):
                p0 = [scale * single['R_W'], np.log10(t_s), 0.85,
                      np.log10(single['R_W']), np.log10(single['Tau_W'])]
                p0 = [min(max(p, l), h) for p, l, h in zip(p0, lo, hi)]
                try:
                    popt, _ = curve_fit(model, w, y, p0=p0,
                                        bounds=(lo, hi), maxfev=80000)
                except Exception:
                    continue
                sols.append((float(np.sum((model(w, *popt) - y) ** 2)), popt))
        if len(sols) < 3:
            return None
        rss, pb = min(sols, key=lambda s: s[0])
        mis2 = np.sqrt(rss) / (np.linalg.norm(y) + 1e-30)
        good = [p for r, p in sols if r < 1.3 * rss]
        spread = max(p[1] for p in good) - min(p[1] for p in good)
        # phi's 1.0 ceiling is the physical Debye limit and allowed (task #83
        # rationale); only the tau box and the 0.3 phi floor count as bounds.
        interior = (pb[1] - lo[1] > 1e-3 and hi[1] - pb[1] > 1e-3
                    and pb[2] - 0.3 > 1e-3)
        improved = mis2 > 0 and mis1 / mis2 >= COMPOSITE_MIN_IMPROVEMENT
        if not (improved and interior and spread <= COMPOSITE_STABILITY_DECADES):
            reason = ('improvement below threshold' if not improved else
                      'slow element on a bound' if not interior else
                      'multistart tau unstable')
            print(f"   - Composite diffusion refit rejected ({reason}); "
                  f"keeping the single Warburg-open element.")
            return None

        R_s, tau_s, phi_s = float(pb[0]), 10.0 ** float(pb[1]), float(pb[2])
        R_W, tau_W = 10.0 ** float(pb[3]), 10.0 ** float(pb[4])
        rec = {'model': 'warburg_open_plus_slow_zarc',
               'R_W': R_W, 'Tau_W': tau_W, 'C_lim': tau_W / R_W,
               'Freq': 1.0 / (2 * np.pi * tau_W),
               'misfit': float(mis2), 'n_points': int(len(w)),
               'slow_zarc': {'R': R_s, 'Tau': tau_s, 'Phi': phi_s,
                             'C_eff': tau_s / R_s if R_s > 0 else 0.0,
                             'Freq': 1.0 / (2 * np.pi * tau_s)}}
        self.warburg_params = rec
        t_win = 1.0 / (2 * np.pi * float(self.target_data[:, 0].min()))
        beyond = " (tau beyond the measured window: R is mostly " \
                 "extrapolated; only the leading tail is measured)" \
                 if tau_s > t_win else ""
        print(f"-> [Warburg] Composite diffusion branch adopted "
              f"({mis1 / mis2:.1f}x over the single element): slow ZARC "
              f"R = {R_s:.4g} Ohm, tau = {tau_s:.3g} s, phi = {phi_s:.3f}"
              f"{beyond}; Warburg R_W = {R_W:.4g} Ohm, "
              f"tau_W = {tau_W:.3g} s.")
        # phi_s is weakly identified over the top of its range: with phi_s held
        # fixed and the other four parameters free the residual costs only
        # ~1.4 % of the optimum at 0.95 (drt-deconvolution.md 2.4.2). A value on
        # the ceiling is where a shallow optimum landed, not a measured shape,
        # and the task #83 bound warnings deliberately do not cover it - the
        # ceiling stays admissible because a genuinely Debye element must still
        # be reported. Say so instead of leaving the value unmarked (task #99).
        if hi[2] - phi_s <= COMPOSITE_PHI_CEILING_TOL:
            print(f"   Warning: the slow element's phi saturates the Debye "
                  f"ceiling ({phi_s:.3f}): phi is weakly identified there "
                  f"(0.95 costs ~1 % of the residual optimum), so read it as "
                  f"'not distinguishable from ideal RC', not as a measured "
                  f"shape.")
        return rec

    def reconstruct_impedance(self, freq: np.ndarray = None, include_warburg: bool = True):
        """
        Reconstruct both Nyquist curves on ONE common frequency grid and with
        ONE common R_inf, so the 'NNLS' and 'equivalent-circuit' results are
        directly comparable (and identical apart from the Gaussian-approximation
        error in the DRT itself).

        freq : grid to evaluate on. Defaults to the preprocessed grid (self.freq).
        include_warburg : when True (default), a fitted Warburg-open element is
            added to the ECM curve so it reproduces the excluded diffusion branch.
            When False the ECM curve is the ZARC relaxation elements only and
            ``ecm_model`` is 'zarc'; the Warburg parameters remain reported on
            ``warburg_params`` regardless. The plotters pass False so the ECM
            curve is drawn only over the region the ZARC elements were fitted
            (task #51, drt-deconvolution.md §7).

        Returns dict with keys:
            'freq', 'R_inf',
            'nnls_re','nnls_im'   -> from the polished DRT vector,
            'ecm_re','ecm_im'     -> from the equivalent circuit (None if absent),
            'ecm_model'           -> 'zarc' (R-CPE) or 'rc' (Gaussian RC) or None.
        Both real parts already include the shared R_inf.
        """
        if self.polished_drt is None:
            raise ValueError(
                "No polished DRT available. Call polish_spectrum() before reconstruct_impedance().")

        if freq is None:
            freq = self.freq
        freq = np.asarray(freq, dtype=float)

        # Single shared ohmic resistance for BOTH curves.
        # Use the same R_inf that was used in polish_spectrum (min of the
        # preprocessed/extrapolated spectrum) so the NNLS reconstruction is
        # internally consistent with the polished DRT.  Falls back to the
        # measured minimum only if polish has not been run yet.
        R_inf = (max(0.0, self.R_inf) if self.R_inf is not None
                 else max(0.0, float(np.min(self.target_data[:, 1]))))

        # Shared design matrices on the common grid using the model tau grid.
        W   = (2 * np.pi * freq)[:, np.newaxis]
        T   = self.tau[np.newaxis, :]
        den = 1.0 + (W * T) ** 2
        A_re = 1.0 / den
        A_im = -(W * T) / den

        # --- NNLS curve: from the polished DRT vector ---
        nnls_re = R_inf + A_re @ self.polished_drt
        nnls_im = A_im @ self.polished_drt

        # Joint mode (task #56): the polished DRT was solved together with a
        # Warburg-open column, so the calculated-DRT curve must include that
        # branch to be consistent with the fit it came from.
        if self.joint_warburg is not None:
            zwo = self.joint_warburg['R_W'] * self._warburg_open(
                2 * np.pi * freq, 1.0, self.joint_warburg['Tau_W'])
            nnls_re = nnls_re + zwo.real
            nnls_im = nnls_im + zwo.imag

        out = {'freq': freq, 'R_inf': R_inf,
               'nnls_re': nnls_re, 'nnls_im': nnls_im,
               'ecm_re': None, 'ecm_im': None, 'ecm_model': None}

        # --- ECM curve ---
        # Prefer the ZARC (R-CPE) model: each element carries a CPE exponent phi
        # so broad/depressed arcs are reproduced directly. Fall back to the
        # Gaussian-RC synthesis only if no ZARC fit is available.
        zarc = getattr(self, 'zarc_params', None)
        if zarc and zarc.get('Elements'):
            w   = 2 * np.pi * freq
            params = []
            for e in zarc['Elements']:
                params += [e['R'], np.log10(e['Tau']), e['Phi']]
            Z = self._zarc_sum(w, R_inf, params)
            # Add the fitted diffusion element so the ECM curve reproduces the
            # low-frequency branch the tail-cut excluded from relaxation
            # analysis. The reported element drives the curve regardless of
            # provenance: the post-hoc fit (section 2.4) or, in joint mode
            # where the post-hoc stage is skipped, the jointly fitted branch
            # (drt-core 5.5, task #60).
            wb = getattr(self, 'warburg_params', None) or self.joint_warburg
            if wb and include_warburg:
                Z = Z + self._warburg_open(w, wb['R_W'], wb['Tau_W'])
                # Composite record (task #89): the slow-ZARC branch is part of
                # the reported diffusion model, so the ECM curve carries it.
                sz = wb.get('slow_zarc')
                if sz:
                    Z = Z + sz['R'] / (1.0 + (1j * w * sz['Tau']) ** sz['Phi'])
                out['ecm_model'] = 'zarc+warburg'
            else:
                out['ecm_model'] = 'zarc'
            out['ecm_re'] = Z.real
            out['ecm_im'] = Z.imag
            return out

        # NOTE: this Gaussian-RC synthesis fallback is defensive only — it is
        # effectively unreachable, because fit_zarc_elements always populates
        # zarc_params whenever refined peaks exist (falling back to phi = 1 on
        # fit failure). It uses the equivalent Amplitude/Sigma stored for each
        # peak, so it also works for ZARC-basis peaks (approximate shape).
        ref = getattr(self, 'refined_ecm_params', None)
        if ref and ref.get('Peaks'):
            G = np.zeros_like(self.tau)
            for p in ref['Peaks']:
                G += p['Amplitude'] * np.exp(
                    -((self.log_tau - p['Mu']) ** 2) / (2 * p['Sigma'] ** 2))
            out['ecm_re'] = R_inf + A_re @ G
            out['ecm_im'] = A_im @ G
            out['ecm_model'] = 'rc'
        return out

    def optimize_gaussian_drt(self, *args, **kwargs) -> dict:
        """Deprecated alias for :meth:`deconvolve_peaks` (task #20).

        The historical name implied a Gaussian basis, but the default is the
        ZARC kernel. Retained so existing callers and tests keep working; new
        code should call ``deconvolve_peaks``.
        """
        return self.deconvolve_peaks(*args, **kwargs)

    def run_full_optimization(self, lambda_reg: float = 0.01) -> dict:
        """Run the full pipeline in sequence: auto_tune -> polish_spectrum -> deconvolve_peaks."""
        print("\n=== DRT Full Automation Pipeline Start ===")
        self.auto_tune()
        self.polish_spectrum(lambda_reg)
        self.deconvolve_peaks()
        print("=== Pipeline complete ===\n")
        return {
            'best_noise':   self.best_noise,
            'polished_drt': self.polished_drt,
            'ecm_params':   getattr(self, 'refined_ecm_params', None)
        }

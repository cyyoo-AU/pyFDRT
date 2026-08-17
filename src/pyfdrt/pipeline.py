import os
import numpy as np
from typing import Callable, Dict, Any, Optional

from pyfdrt.io.eis_loader import (load_eis_file, filter_positive_imag,
                                  trim_overcorrected_hf, SPIKE_TRIM_FACTOR)
from pyfdrt.io.eis_utils import ensure_ascending_2d
from pyfdrt.core.EIScorrectionKK import EIS_correction_kk
from pyfdrt.core.dataExtrapolationKK import dataExtrapolationKK, ARC_CLOSURE_TOL
from pyfdrt.core.interpolateEIS import interpolateEIS
from pyfdrt.core.DRTOptimizer import DRTOptimizer, WARBURG_MIN_CONTRIBUTION
from pyfdrt.core.linKK import kk_gate
from pyfdrt.pipeline_config import (
    PipelineConfig, LCorrMode, LambdaMode, TailCutMode, RInfMode, ExtRangeMode,
    KKGateMode, WarburgMode,
)

# Guard threshold for ext_range_mode = AUTO: the reduced-range result is
# adopted only when every refined peak's R and Tau stay within this relative
# drift of the reference run (drt-pipeline.md 2.4, task #43).
AUTO_RANGE_DRIFT_TOL = 0.05

class DRTAnalyzer:
    """
    GUI-independent EIS data processing and DRT (Distribution of Relaxation Times) analysis pipeline.
    """

    def __init__(self, logger: Optional[Callable[[str], None]] = None):
        """
        :param logger: callback that handles log messages (defaults to console print)
        """
        self.logger = logger or print
        self.raw_data = None
        self.optimizer = None
        self.results = None

    def log(self, message: str):
        self.logger(message)

    def load_data(self, file_path: str) -> np.ndarray:
        """
        Load an EIS data file (.txt, .z, .csv, .dat) without dropping points.
        """
        self.raw_data = load_eis_file(file_path)
        self.log(f"-> [OK] Data loaded: {len(self.raw_data)} points")
        return self.raw_data

    def run(self, config=None) -> Dict[str, Any]:
        """
        Run the preprocessing and DRT optimization pipeline.

        ``config`` may be a :class:`~pyfdrt.pipeline_config.PipelineConfig`, a
        label dict in the GUI/CLI style (e.g. ``{'l_corr_mode': 'Auto (Smart
        Detection)', 'ext_decades': 2, ...}``), or ``None`` for the default
        configuration. A label dict is validated and translated by
        :meth:`PipelineConfig.from_dict`, which rejects unknown keys instead of
        silently ignoring them. The 20-key schema, its defaults and the
        label-matching rules are specified in docs/specs/drt-pipeline.md
        section 3. Defaults are the golden-corpus configuration.

        With ``ext_range_mode = AUTO`` the pipeline may run twice: once at the
        configured decade counts and once with closed arc edges reduced to
        zero, adopting the reduced result only when the guard of
        drt-pipeline.md 2.4 accepts it.
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Please call load_data(file_path) first.")

        cfg = PipelineConfig.from_dict(config)

        # The low side may be overridden independently (task #41); the high side
        # always follows ext_decades - see drt-preprocessing.md 6.2 for why.
        dec_low = cfg.ext_decades_low if cfg.ext_decades_low is not None else cfg.ext_decades
        dec_high = cfg.ext_decades

        out = self._run_pipeline(cfg, dec_low, dec_high)
        if cfg.ext_range_mode is ExtRangeMode.AUTO:
            out = self._auto_range(cfg, out, dec_low, dec_high)

        rm = out['ext_dbg'].get('range_mode')
        dec_used = (tuple(rm['reduced_decades'])
                    if rm and rm.get('adopted') else (dec_low, dec_high))

        # The task #59 stability-guarded lambda escalation ran here until task
        # #66 removed it: its probe quantity (perturbation drift of the refined
        # peaks) is platform-chaotic on exactly the ill-conditioned solutions
        # it was meant to catch, so its escalate/keep verdict flipped between
        # platforms and lambda_reg - a pinned quantity - differed 2x (ubuntu
        # CI red). The lambda pick is the plain L-curve selection (drt-core
        # 5.4); the 7.1-corner consequence on LIB_data is handled by the
        # conformance refined-value exclusion instead (drt-conformance 3).

        # Detect-then-joint upgrade (task #60): AUTO means "post-hoc, unless
        # the default pass detects a diffusion tail-cut - then the diffusion
        # branch belongs INSIDE the NNLS fit, so re-run with the joint polish
        # (drt-core 5.5) and adopt it". Unconditional JOINT was reviewed and
        # rejected as a default: it asserts a Warburg on non-diffusion
        # spectra (false positive on synthetic negami) and rewrites their
        # ECMs. The joint re-run inherits the lambda of the default pass as a
        # MANUAL value - no L-curve inside the upgrade pass, so the adopted
        # lambda is exactly the deterministic default-pass one (task #61).
        if (cfg.warburg_mode is WarburgMode.AUTO
                and getattr(self.optimizer, 'deconv_tail_cut_tau', None) is not None):
            from dataclasses import replace
            cfg_joint = replace(cfg, warburg_mode=WarburgMode.JOINT,
                                lambda_mode=LambdaMode.MANUAL,
                                lambda_val=float(out['lambda_reg']))
            self.log("-> [Warburg Auto] Diffusion tail-cut fired in the default pass; "
                     f"re-running with the joint Warburg polish at the adopted "
                     f"lambda {out['lambda_reg']:.4g} (drt-core 5.5)...")
            prev_dbg = out['ext_dbg']
            out = self._run_pipeline(cfg_joint, *dec_used)
            if 'range_mode' in prev_dbg:
                out['ext_dbg']['range_mode'] = prev_dbg['range_mode']
            out['ext_dbg']['warburg_upgrade'] = {
                'from': 'auto', 'reason': 'diffusion tail-cut active in the default pass',
                'lambda_inherited': cfg_joint.lambda_val}
        return out

    @staticmethod
    def _reconstruction_misfit(optimizer) -> float:
        """Relative L2 misfit of the NNLS reconstruction on the measured grid."""
        rec = optimizer.reconstruct_impedance(freq=optimizer.target_data[:, 0])
        z_meas = optimizer.target_data[:, 1] + 1j * optimizer.target_data[:, 2]
        z_rec = rec['nnls_re'] + 1j * rec['nnls_im']
        return float(np.linalg.norm(z_rec - z_meas) / np.linalg.norm(z_meas))

    def _auto_range(self, cfg: PipelineConfig, reference: Dict[str, Any],
                    dec_low: int, dec_high: int) -> Dict[str, Any]:
        """Guarded auto extrapolation range (drt-pipeline.md 2.4, task #43).

        Evaluates the arc-closure diagnostic on the corrected measured data,
        reduces CLOSED edges to zero decades, re-runs the pipeline at the
        reduced counts, and adopts the reduced result only if the extracted
        ECM is stable (identical peak count, per-peak R/Tau drift under
        ``AUTO_RANGE_DRIFT_TOL``) and the reconstruction misfit is not worse.
        The decision is recorded in ``ext_dbg['range_mode']`` (unpinned).
        """
        opt_ref = reference['optimizer']
        asc, _ = ensure_ascending_2d(opt_ref.target_data)
        z_im = asc[:, 2]
        max_abs_im = float(np.max(np.abs(z_im)))
        # Same rule dataExtrapolationKK uses: an edge is OPEN when its |Z_im|
        # fraction exceeds ARC_CLOSURE_TOL. Ascending rows: index 0 = lowest f.
        low_closed = abs(z_im[0]) / max_abs_im <= ARC_CLOSURE_TOL
        high_closed = abs(z_im[-1]) / max_abs_im <= ARC_CLOSURE_TOL
        red_low = 0 if low_closed else dec_low
        red_high = 0 if high_closed else dec_high

        info = {'mode': 'auto', 'reference_decades': (dec_low, dec_high),
                'reduced_decades': (red_low, red_high),
                'low_closed': low_closed, 'high_closed': high_closed}
        if (red_low, red_high) == (dec_low, dec_high):
            info.update(adopted=False, reason='no closed edge; nothing to reduce')
            reference['ext_dbg']['range_mode'] = info
            self.log("-> [Auto Range] Both arc edges open; keeping "
                     f"{dec_low} low / {dec_high} high decades (single pass)")
            return reference

        self.log(f"-> [Auto Range] Closed edge(s) detected (low={'closed' if low_closed else 'open'}, "
                 f"high={'closed' if high_closed else 'open'}); trying "
                 f"{red_low} low / {red_high} high decades...")
        reduced = self._run_pipeline(cfg, red_low, red_high)

        peaks_ref = reference['results'].get('Peaks', []) if reference['results'] else []
        peaks_red = reduced['results'].get('Peaks', []) if reduced['results'] else []
        mis_ref = self._reconstruction_misfit(reference['optimizer'])
        mis_red = self._reconstruction_misfit(reduced['optimizer'])
        info.update(misfit_reference=mis_ref, misfit_reduced=mis_red,
                    n_peaks_reference=len(peaks_ref), n_peaks_reduced=len(peaks_red))

        if len(peaks_red) != len(peaks_ref):
            adopted, reason = False, (f'peak count changed '
                                      f'({len(peaks_ref)} -> {len(peaks_red)})')
        else:
            drift = 0.0
            for p_ref, p_red in zip(peaks_ref, peaks_red):
                for key in ('R', 'Tau'):
                    denom = max(abs(p_ref[key]), 1e-12)
                    drift = max(drift, abs(p_red[key] - p_ref[key]) / denom)
            info['max_peak_drift'] = drift
            if drift >= AUTO_RANGE_DRIFT_TOL:
                adopted, reason = False, (f'peak R/Tau drift {drift:.1%} exceeds '
                                          f'{AUTO_RANGE_DRIFT_TOL:.0%}')
            elif mis_red > mis_ref:
                adopted, reason = False, (f'reconstruction misfit worsened '
                                          f'({mis_ref:.2%} -> {mis_red:.2%})')
            else:
                adopted, reason = True, (f'stable ECM (drift {drift:.1%}), misfit '
                                         f'{mis_ref:.2%} -> {mis_red:.2%}')
        info.update(adopted=adopted, reason=reason)

        if adopted:
            self.log(f"-> [Auto Range] ADOPTED reduced range: {reason}")
            reduced['ext_dbg']['range_mode'] = info
            return reduced
        # Fall back: restore the analyzer state to the reference pass, which
        # the second _run_pipeline call overwrote.
        self.optimizer = reference['optimizer']
        self.results = reference['results']
        self.log(f"-> [Auto Range] REJECTED reduced range ({reason}); "
                 f"keeping {dec_low} low / {dec_high} high decades")
        reference['ext_dbg']['range_mode'] = info
        return reference

    def _run_pipeline(self, cfg: PipelineConfig, dec_low: int,
                      dec_high: int) -> Dict[str, Any]:
        """Single pass of the preprocessing and DRT optimization pipeline at
        explicit decade counts. ``run`` is the orchestrating entry point."""
        self.log("\n=================================")
        self.log("Starting pipeline...")

        # Step 1: Inductance correction and filtering
        if cfg.l_corr_mode is LCorrMode.FORCE_OFF:
            # Skip KK correction; immediately filter out any positive imaginary points
            EIS_corrected, n_dropped = filter_positive_imag(self.raw_data)
            self.log("-> L Correction: Force OFF (Keep Original)")
            if n_dropped:
                self.log(f"   - Excluded {n_dropped} point(s) with positive imaginary part (Z_imag > 0)")
            L_val = 0.0
        elif cfg.l_corr_mode is LCorrMode.FORCE_ON:
            # Apply KK correction to unfiltered raw data. force=True bypasses the
            # 0.5 % significance threshold so a small but real inductance is still
            # corrected; the physical-validity check (L <= 0) still applies, since
            # forcing it would fabricate an inductive artefact.
            # fit_points is deliberately NOT overridden (task #70): the
            # estimation is mode-independent; force affects gate B only.
            corrected_raw, L_val = EIS_correction_kk(
                self.raw_data, logger=self.log, force=True)
            self.log(f"-> L Correction: Force ON (L = {L_val:.3e} H)")
            # Clean up any remaining positive imaginary points after correction
            EIS_corrected, n_dropped = filter_positive_imag(corrected_raw)
            if n_dropped:
                self.log(f"   - Excluded {n_dropped} point(s) with positive imaginary part after L correction")
        else:
            # Apply KK correction to unfiltered raw data
            corrected_raw, L_val = EIS_correction_kk(
                self.raw_data, logger=self.log)
            self.log(f"-> L Correction: Auto (L = {L_val:.3e} H)")
            # Clean up any remaining positive imaginary points after correction
            EIS_corrected, n_dropped = filter_positive_imag(corrected_raw)
            if n_dropped:
                self.log(f"   - Excluded {n_dropped} point(s) with positive imaginary part after L correction")
        # Over-correction spike trim (drt-preprocessing.md 2.4.1, task #91):
        # negative residuals of the L subtraction survive the sign filter as
        # spurious capacitive spikes at the HF edge. Only meaningful when a
        # correction was actually applied.
        if L_val > 0:
            EIS_corrected, _spike_fs = trim_overcorrected_hf(
                EIS_corrected, self.raw_data)
            if _spike_fs:
                self.log(f"   - Trimmed {len(_spike_fs)} over-corrected HF "
                         f"point(s) (|Z_im| > {SPIKE_TRIM_FACTOR:.0f}x the "
                         f"neighbour median): "
                         + ", ".join(f"{f:.4g} Hz" for f in _spike_fs))
        # Step 1.5: lin-KK measurement-validity gate (drt-preprocessing.md 2.6,
        # task #45). Warn-only: it never rejects the run and never modifies the
        # data. Under ext_range_mode=AUTO both passes evaluate it on the same
        # corrected data; the repeat is cheap and bit-identical.
        if cfg.kk_gate_mode is KKGateMode.WARN:
            try:
                gate_rec = kk_gate(EIS_corrected)
            except Exception as exc:   # a diagnostic must never kill the run
                gate_rec = {'skipped': f'{type(exc).__name__}: {exc}'}
            if 'skipped' in gate_rec:
                self.log(f"-> [KK Gate] Skipped ({gate_rec['skipped']})")
            elif gate_rec['ok']:
                self.log(f"-> [KK Gate] PASS (residual RMS {gate_rec['rms']:.2%}, "
                         f"max {gate_rec['max']:.2%} of |Z|)")
            else:
                self.log(f"-> [KK Gate] WARNING: {gate_rec['reason']}. The measurement "
                         "may violate Kramers-Kronig (drift, nonlinearity, or glitches); "
                         "downstream results may be unreliable.")
        else:
            gate_rec = {'skipped': 'kk_gate_mode = Off'}

        # Step 2: Extrapolation
        # ext_method enum values are exactly 'linear' / 'rc' / 'auto', the
        # strings dataExtrapolationKK expects. The label-vs-mode translation
        # (strict whole-value matching, task #95) lives in ExtMethod.from_label.
        ext_method = cfg.ext_method.value

        # R_inf estimation mode: 'Auto' corrects the min-Z_re estimate when the
        # high-frequency arc is open (see dataExtrapolationKK); 'Min Z_re' is
        # the legacy behaviour; 'Manual' uses cfg.r_inf_value.
        if cfg.r_inf_mode is RInfMode.MANUAL:
            r_mode, r_val = 'manual', cfg.r_inf_value
        elif cfg.r_inf_mode is RInfMode.MIN_ZRE:
            r_mode, r_val = 'min_zre', None
        else:
            r_mode, r_val = 'auto', None

        EIS_extrapolated, ext_dbg, R_inf_ext = dataExtrapolationKK(
            EIS_corrected, decades_low=dec_low, decades_high=dec_high,
            method=ext_method, r_inf_mode=r_mode, r_inf_value=r_val,
            logger=self.log)
        ext_dbg['kk_gate'] = gate_rec   # unpinned diagnostic record
        ext_desc = (f"{dec_low} decades on each side" if dec_low == dec_high
                    else f"{dec_low} low / {dec_high} high decades")
        self.log(f"-> Extrapolation complete (extended by {ext_desc}, "
                 f"method = {ext_dbg['method']}, R_inf = {R_inf_ext:.4f} "
                 f"[source: {ext_dbg.get('r_inf_source', 'min Z_re')}])")
        if 'rc' in ext_dbg.get('method', '') and 'rc_n_elements' in ext_dbg:
            self.log(f"   - RC extrapolation model: {ext_dbg['rc_n_elements']} RC element(s), "
                     f"fit misfit = {ext_dbg.get('rc_misfit', float('nan')):.3f}")
        
        if ext_method == 'auto':
            chosen = ext_dbg['method'].upper()
            self.log(f"   - Auto decision: {ext_dbg['reason']} "
                     f"[linear R_inf={ext_dbg['R_inf_linear']:.4f}, "
                     f"RC R_inf={ext_dbg['R_inf_rc']:.4f}, RC misfit={ext_dbg['rc_misfit']:.3f}]")

        # Step 3: Interpolation onto uniform log grid
        N_total = len(EIS_extrapolated) * 2
        EIS_ready = interpolateEIS(EIS_extrapolated, N=N_total)
        self.log(f"-> Interpolation complete (uniform log spacing: {N_total} points)")

        # Step 4: Initialize optimizer
        # Pass the L-corrected data as EIS_raw so the NNLS and ZARC fitting target is the physical impedance
        self.optimizer = DRTOptimizer(EIS_preprocessed=EIS_ready, EIS_raw=EIS_corrected, R_inf=R_inf_ext)
        self.optimizer.solver_method = cfg.solver_method
        self.optimizer.auto_tune()

        # Step 5: NNLS polishing
        lambda_prior_val = cfg.lambda_prior

        if cfg.lambda_mode is LambdaMode.MANUAL:
            lam_reg = cfg.lambda_val
            self.log(f"-> [Step 2] NNLS Polishing (Manual Lambda: {lam_reg}, lambda_prior: {lambda_prior_val})")
        else:  # Auto (L-Curve)
            # Couple the prior into the lambda search as well
            lam_reg = self.optimizer.optimize_lambda(method='L-Curve', prior=self.optimizer.base_drt, lambda_prior=lambda_prior_val)
            self.log(f"-> [Step 2] NNLS Polishing (Auto L-Curve Lambda: {lam_reg:.4e}, lambda_prior: {lambda_prior_val})")

        if cfg.warburg_mode is WarburgMode.JOINT:
            # Joint mode (task #56, drt-core 5.5): both passes solve the RC
            # basis together with one Warburg-open column on a measured-only
            # target, so the diffusion branch is part of the NNLS fit rather
            # than a post-hoc patch. Same lambda machinery as the default path.
            self.optimizer.polish_spectrum_joint(lambda_reg=lam_reg,
                                                 lambda_prior=lambda_prior_val)
            jw = self.optimizer.joint_warburg
            if jw['C_lim'] is None:
                # Degenerate fit (drt-core 5.5, task #98): the column collapsed
                # onto a search boundary, so C_lim and Freq are withheld rather
                # than printed as a plausible finite capacitance.
                self.log(f"-> [Step 2] Joint polish complete (R_W = {jw['R_W']:.4g} Ohm, "
                         f"tau_W = {jw['Tau_W']:.3g} s; DEGENERATE: diffusion column "
                         f"collapsed, no knee identified and C_lim withheld)")
            else:
                self.log(f"-> [Step 2] Joint polish complete (R_W = {jw['R_W']:.4g} Ohm, "
                         f"tau_W = {jw['Tau_W']:.3g} s, C_lim = {jw['C_lim']:.3g} F"
                         + ("" if jw['knee_resolved']
                            else "; knee beyond window: only C_lim identified") + ")")
        else:
            # First polish: FFT base_drt as the prior guide
            first_drt = self.optimizer.polish_spectrum(lambda_reg=lam_reg, prior=self.optimizer.base_drt, lambda_prior=lambda_prior_val)
            self.log("-> [Step 2a] First polish complete (prior = FFT base_drt)")

            # Second polish: first-pass result (first_drt) as the prior guide
            self.optimizer.polish_spectrum(lambda_reg=lam_reg, prior=first_drt, lambda_prior=lambda_prior_val)
            self.log("-> [Step 2b] Second polish complete (prior = first pass)")

        # Step 6: Peak deconvolution
        max_p = cfg.max_peaks
        crit = cfg.criterion.value

        # Peak deconvolution method. Default is the analytic ZARC-kernel basis
        # with BIC model selection (window-clamped, diffusion tail-cut): unlike
        # the symmetric Gaussian basis it represents broad/depressed arcs with
        # a single kernel, so wide arcs are not over-split.
        # Validated 2026-07 on 14 datasets (zarc/negami/RRQRQ synthetic truths
        # all recovered; LIB battery data needs the tail-cut, which triggers
        # only on diffusion-type spectra).
        dec_method = cfg.deconvolution_method.value
        lam_lasso = cfg.lambda_lasso
        channels = cfg.lasso_channels

        # Diffusion tail-cut control (ZARC method only):
        #   Auto   -> detect a diffusion rise at the low-frequency window edge
        #   Manual -> cut at cfg.tail_cut_tau seconds
        #   Off    -> no cut
        if cfg.tail_cut_mode is TailCutMode.MANUAL:
            tail_cut = cfg.tail_cut_tau
        elif cfg.tail_cut_mode is TailCutMode.OFF:
            tail_cut = 'off'
        else:
            tail_cut = 'auto'

        # JOINT already fitted the diffusion branch inside the NNLS polish, so
        # the post-hoc fit is skipped and the joint record is injected below.
        warburg_arg = ('off' if cfg.warburg_mode is WarburgMode.JOINT
                       else cfg.warburg_mode.value)

        self.log(f"-> [Step 3] Peak deconvolution (method = {dec_method}, max {max_p} peaks, "
                 f"tail_cut = {tail_cut}, warburg = {cfg.warburg_mode.value})...")

        self.results = self.optimizer.deconvolve_peaks(
            max_peaks=max_p, criterion=crit,
            method=dec_method, lambda_lasso=lam_lasso, lasso_channels=channels,
            tail_cut=tail_cut, warburg=warburg_arg
        )

        # Joint mode: report a Warburg element unless the joint branch is
        # negligible (below WARBURG_MIN_CONTRIBUTION of |Z| on the measured
        # window - a clean non-diffusion spectrum drives R_W to ~0 and
        # reporting it would be noise). This gate is the exact complement of
        # the degeneracy test in polish_spectrum_joint, which withholds C_lim
        # and Freq on the same amplitude (drt-core.md 5.5), so both read the
        # one constant: retuning it must move them together or neither.
        # The joint column shares the diffusion mass with the free
        # slow RC bins beyond the cut, so it under-carries the branch in the
        # reported circuit (task #86, drt-deconvolution.md 2.4.1): refit the
        # reported element on the post-refinement residual of the excluded
        # region when a cut is active, falling back to the joint record when
        # the ordinary section-2.4 guards decline the fit. The NNLS
        # reconstruction keeps the joint branch either way.
        if (cfg.warburg_mode is WarburgMode.JOINT
                and self.optimizer.joint_warburg is not None
                and self.optimizer.joint_warburg['contribution'] > WARBURG_MIN_CONTRIBUTION):
            wb_refit = self.optimizer.fit_warburg_tail()
            if wb_refit is not None:
                # Composite refit (task #89, drt-deconvolution.md 2.4.2): a
                # slow relaxation can hide in the excluded region; adopt the
                # slow-ZARC + Warburg composite only under its guards.
                wb_comp = self.optimizer.fit_diffusion_composite()
                if wb_comp is not None:
                    wb_refit = wb_comp
            self.results['Warburg'] = (wb_refit if wb_refit is not None
                                       else self.optimizer.joint_warburg)

        # Log extracted parameters
        self.log("\n[Final ECM Parameters Extracted]")
        if not self.results or 'Peaks' not in self.results:
            self.log("No peaks found.")
        else:
            self.log(f"R_inf (High-freq resistance): {self.results['R_inf']:.4f} Ω")
            for param in self.results['Peaks']:
                phi = param.get('Phi', 1.0)
                C_eff = param.get('C_eff', param['Tau'] / param['R'] if param['R'] > 0 else 0.0)
                Q_val = param.get('CPE_Q', float('nan'))
                self.log(
                    f"Peak {param['Peak']}: "
                    f"R={param['R']:.4f} Ω, "
                    f"phi(CPE)={phi:.3f}, "
                    f"Q={Q_val:.2e} S·s^phi, "
                    f"Tau={param['Tau']:.2e} s, "
                    f"C_eff={C_eff:.2e} F"
                )
            wb = self.results.get('Warburg')
            if wb:
                joint = wb.get('model') == 'warburg_open_joint'
                composite = 'slow_zarc' in wb
                how = (f"joint NNLS fit on {wb['n_points']} measured points"
                       if joint else
                       f"composite slow-ZARC + Warburg refit on "
                       f"{wb['n_points']} measured points"
                       if composite else
                       f"fitted on {wb['n_points']} excluded LF points")
                c_lim = (f"C_lim={wb['C_lim']:.3g} F "
                         if wb.get('C_lim') is not None
                         else "C_lim withheld (degenerate fit) ")
                self.log(
                    f"Warburg (open, diffusion): "
                    f"R_W={wb['R_W']:.4g} Ω, "
                    f"Tau_W={wb['Tau_W']:.3g} s, "
                    f"{c_lim}"
                    f"({how}, misfit={wb['misfit']:.3f})"
                )
                if composite:
                    sz = wb['slow_zarc']
                    self.log(
                        f"   + slow diffusion-branch element: "
                        f"R={sz['R']:.4g} Ω, Tau={sz['Tau']:.3g} s, "
                        f"phi={sz['Phi']:.3f}, C_eff={sz['C_eff']:.3g} F "
                        f"(reported inside the Warburg record - R is mostly "
                        f"beyond-window extrapolation)")
                if joint and not wb['knee_resolved']:
                    if wb.get('C_lim') is None:
                        # Degenerate fit (drt-core 5.5, task #98): nothing was
                        # identified here, so do not point the reader at C_lim.
                        self.log("   - Diffusion column collapsed onto the search "
                                 "floor: no knee identified and C_lim withheld; "
                                 "treat R_W/Tau_W as unsupported.")
                    else:
                        self.log("   - Diffusion knee beyond the measured window: "
                                 "C_lim is identified, R_W/Tau_W individually are not.")

        self.log("=================================\n-> [Done] Optimization complete!")

        return {
            'optimizer': self.optimizer,
            'results': self.results,
            'ext_dbg': ext_dbg,
            'L_val': L_val,
            'lambda_reg': lam_reg
        }

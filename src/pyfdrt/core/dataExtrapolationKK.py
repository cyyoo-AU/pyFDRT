import numpy as np
from scipy.optimize import lsq_linear
from ..io.eis_utils import ensure_ascending_2d, restore_order


def _polarization_resistance_logscale(omega, Z_im):
    """
    Eq. (10): polarization resistance R_pol from the log-frequency-scale discretisation
    of the Kramers-Kronig integral, Eq. (9):

        R_pol = (2/pi) * 1/(n-1) * ln(omega_min/omega_max) * sum(Z_im)

    Z_im is the *raw* measured imaginary part (negative for capacitive data),
    so ln(omega_min/omega_max) < 0 and sum(Z_im) < 0 yield R_pol > 0.
    """
    n = len(omega)
    omega_min = omega.min()
    omega_max = omega.max()
    return (2.0 / np.pi) * (1.0 / (n - 1)) * np.log(omega_min / omega_max) * np.sum(Z_im)


def _ohmic_highfreq(Z_re):
    """
    Physical ohmic resistance estimate = high-frequency real-axis limit.

    At omega -> inf every RC arc collapses (Z_re -> R_inf), so the smallest
    measured real part is the data-driven ohmic intercept.  This matches the
    convention used elsewhere in the codebase (DRTOptimizer uses min(Z_re)).
    Clamped to be non-negative.
    """
    return max(0.0, float(np.min(Z_re)))


def _edge_slope(x, y, n_edge, side):
    """Least-squares slope/intercept of y vs x on the lowest/highest n_edge points."""
    if side == 'low':
        xs, ys = x[:n_edge], y[:n_edge]
    else:  # 'high'
        xs, ys = x[-n_edge:], y[-n_edge:]
    slope, intercept = np.polyfit(xs, ys, 1)
    return slope, intercept


def _linear_extrapolation(f_meas, Z_re, Z_im, R_inf,
                          decades_low, decades_high, pts_per_decade, n_edge):
    """
    Linear data extrapolation on a logarithmic frequency scale (Eqs. 8-10).

    High-frequency end relaxes toward the final values:
        Z_re -> R_inf (ohmic limit),   Z_im -> 0  (Eq. 8 leading limit).
    Low-frequency end uses a straight-line continuation (in log f) of the
    lowest measured points, since Eqs. (8)/(10) only fix the high-f limit.
    """
    log_f = np.log10(f_meas)
    log_f_min, log_f_max = log_f[0], log_f[-1]

    f_ext_low = 10 ** np.linspace(log_f_min - decades_low, log_f_min,
                                  decades_low * pts_per_decade, endpoint=False)
    f_ext_high = 10 ** np.linspace(log_f_max, log_f_max + decades_high,
                                   decades_high * pts_per_decade + 1)[1:]

    # --- High-frequency extension: linear (in log f) toward the eq. (8)/(10) limits ---
    # decades_high = 0 means "no extension on this side" (the documented meaning
    # of 0 in drt-pipeline.md 3.1): the extension is empty and must not be
    # indexed (task #39). The low side already tolerates an empty extension.
    log_hi = np.log10(f_ext_high)
    if log_hi.size:
        log_hi_end = log_hi[-1]
        span_hi = (log_hi - log_f_max) / (log_hi_end - log_f_max) if log_hi_end > log_f_max else np.zeros_like(log_hi)
        Z_re_hi = Z_re[-1] + (R_inf - Z_re[-1]) * span_hi   # -> R_inf  (Eq. 10)
        Z_im_hi = Z_im[-1] + (0.0   - Z_im[-1]) * span_hi   # -> 0      (Eq. 8)
    else:
        Z_re_hi = np.empty(0)
        Z_im_hi = np.empty(0)

    # --- Low-frequency extension: straight-line continuation of the lowest points ---
    s_re_lo, b_re_lo = _edge_slope(log_f, Z_re, n_edge, 'low')
    s_im_lo, b_im_lo = _edge_slope(log_f, Z_im, n_edge, 'low')
    log_lo = np.log10(f_ext_low)
    Z_re_lo = s_re_lo * log_lo + b_re_lo
    Z_im_lo = s_im_lo * log_lo + b_im_lo

    f_total    = np.concatenate((f_ext_low, f_meas, f_ext_high))
    Z_re_total = np.concatenate((Z_re_lo, Z_re, Z_re_hi))
    Z_im_total = np.concatenate((Z_im_lo, Z_im, Z_im_hi))
    return np.column_stack((f_total, Z_re_total, Z_im_total))


def _rc_series_extrapolation(f_meas, Z_re, Z_im,
                             decades_low, decades_high, pts_per_decade):
    """
    Physical RC-series model fit (Eqs. 11-12) sharing the measured tau grid,
    solved by *plain* modulus-weighted non-negative least squares (no Tikhonov
    regularisation / no lambda).

        Z_re(w) = R_inf + sum_m R_m / (1 + (w*tau_m)^2)                (11)
        Z_im(w) = w*L   - sum_m R_m*w*tau_m / (1 + (w*tau_m)^2)        (12)

    Returns (EIS_extrapolated, x_opt, R_inf_fit, misfit) where misfit is the
    modulus-weighted relative RMSE of the fit over the measured window.

    The fitted RC-arc model is used **only to synthesise the low/high frequency
    extrapolation tails**.  The measured window is returned verbatim, so the DRT
    is not idealised by replacing real data with a sum of perfect Debye terms
    (which collapses the DRT into artificially sharp spikes).
    """
    omega = 2 * np.pi * f_meas
    tau   = 1.0 / omega
    M     = len(tau)

    W     = omega[:, np.newaxis]
    T     = tau[np.newaxis, :]
    denom = 1.0 + (W * T) ** 2

    A_re = np.zeros((len(f_meas), M + 2))
    A_im = np.zeros((len(f_meas), M + 2))
    A_re[:, 0]  = 1.0          # R_inf column (Eq. 11)
    A_im[:, 1]  = omega        # L column     (Eq. 12)
    A_re[:, 2:] = 1.0 / denom
    A_im[:, 2:] = -(W * T) / denom

    A = np.vstack((A_re, A_im))
    b = np.concatenate((Z_re, Z_im))

    # Modulus weighting (data weighting, NOT regularisation): keeps small high-f
    # arcs from being swamped by large low-f arcs.
    Z_mod   = np.sqrt(Z_re ** 2 + Z_im ** 2)
    w_diag  = np.concatenate((1.0 / (Z_mod + 1e-8),) * 2)
    A_w     = A * w_diag[:, np.newaxis]
    b_w     = b * w_diag

    lower = [0.0, -np.inf] + [0.0] * M     # R_inf >= 0, L free, R_m >= 0
    x_opt = lsq_linear(A_w, b_w, bounds=(lower, np.inf), method='bvls').x

    # Data-misfit quality metric over the measured window.
    misfit = np.linalg.norm(A_w @ x_opt - b_w) / (np.linalg.norm(b_w) + 1e-30)

    R_inf_fit, L_fit, R_m = x_opt[0], x_opt[1], x_opt[2:]

    log_f_min = np.log10(f_meas[0])
    log_f_max = np.log10(f_meas[-1])
    f_ext_low = 10 ** np.linspace(log_f_min - decades_low, log_f_min,
                                  decades_low * pts_per_decade, endpoint=False)
    f_ext_high = 10 ** np.linspace(log_f_max, log_f_max + decades_high,
                                   decades_high * pts_per_decade + 1)[1:]

    # Evaluate the fitted RC arcs at arbitrary frequencies.  The series
    # inductance term (w*L) is used solely to absorb the measured high-frequency
    # tail during fitting; propagating w*L into the extrapolated window makes Z_im
    # diverge linearly (unphysical), so it is excluded here.
    def _rc_model(f):
        W_f = (2 * np.pi * f)[:, np.newaxis]
        d_f = 1.0 + (W_f * T) ** 2
        zr  = R_inf_fit + np.sum(R_m / d_f, axis=1)
        zi  = -np.sum(R_m * (W_f * T) / d_f, axis=1)
        return zr, zi

    # Synthesize ONLY the extrapolation tails from the RC model; the measured
    # window is preserved verbatim (see assembly below).  Each tail is anchored
    # to the adjacent measured edge so it joins the data continuously and the
    # measured spectrum's physical ohmic resistance is retained.
    zr_edge, zi_edge = _rc_model(np.array([f_meas[0], f_meas[-1]]))
    zr_lo, zi_lo = _rc_model(f_ext_low)
    zr_hi, zi_hi = _rc_model(f_ext_high)
    zr_lo += Z_re[0]  - zr_edge[0];  zi_lo += Z_im[0]  - zi_edge[0]
    zr_hi += Z_re[-1] - zr_edge[1];  zi_hi += Z_im[-1] - zi_edge[1]

    f_total    = np.concatenate((f_ext_low, f_meas, f_ext_high))
    Z_re_total = np.concatenate((zr_lo, Z_re, zr_hi))
    Z_im_total = np.concatenate((zi_lo, Z_im, zi_hi))
    return np.column_stack((f_total, Z_re_total, Z_im_total)), x_opt, R_inf_fit, misfit


# Edge |Z_im| fraction above which an arc is deemed OPEN at that edge. Shared
# with the pipeline's guarded auto-range mode (drt-pipeline.md 2.4) so both
# call sites agree on what "closed" means.
ARC_CLOSURE_TOL = 0.15

# Low-frequency |Z_im| fraction above which the LF arc is open ENOUGH to warrant
# an RC continuation of the low-frequency tail (drt-preprocessing.md 3.4). It is
# deliberately higher than ARC_CLOSURE_TOL: a marginally-open LF arc is well
# continued by a linear tail, and forcing RC there restructures the DRT (peaks
# merge or split) rather than merely extending it. Only a strongly-open arc — a
# mass-transport / diffusion tail still far from the real axis — carries the
# multi-relaxation structure that an RC ladder is needed to continue.
LF_RC_OPEN_TOL = 0.30


def dataExtrapolationKK(EIS: np.ndarray, decades_low=2, decades_high=2,
                        pts_per_decade=10, method='auto',
                        arc_closure_tol=ARC_CLOSURE_TOL, rc_misfit_tol=0.10,
                        n_edge=5, correct_ohmic=True,
                        r_inf_mode='auto', r_inf_value=None,
                        lf_rc_open_tol=LF_RC_OPEN_TOL, logger=print):
    """
    Extend the EIS frequency window so the Fourier-transform DRT has a
    sufficiently wide range, using only the methodology of Eqs. (8)-(12).
    No Tikhonov regularisation (no lambda) is used.

    method :
        'linear' -> linear extrapolation in log-frequency (Eqs. 8-10).
        'rc'     -> physical RC-series model fit (Eqs. 11-12), plain NNLS.
        'auto'   -> run BOTH, then select a method PER EDGE (drt-preprocessing 3.4):
                      HF tail + R_inf:
                        * arcs closed at both edges            -> linear (sufficient)
                        * high-frequency arc open             -> linear (anchored R_inf)
                        * low-frequency arc open, RC fits well -> rc
                        * low-frequency arc open, RC unreliable -> linear
                      LF tail (independent):
                        * LF arc strongly open (>= lf_rc_open_tol) & RC fits well -> rc
                        * otherwise                                              -> linear
                    Reported as 'linear'/'rc' when the edges agree, else a
                    per-end token ('rc-LF/linear-HF' or 'linear-LF/rc-HF').
    correct_ohmic : if True (default), align the extrapolation to the physical
                    ohmic resistance = high-frequency Z_re limit (~min Z_re):
                      - linear: high-f target R_inf is the physical ohmic value;
                      - rc    : the measured window is preserved as-is (so it
                                already carries the physical ohmic); this flag
                                only selects which R_inf value is reported.
                    if False, use Eq.(10) for linear and the RC-fitted R_inf.
    r_inf_mode : how the ohmic resistance is estimated (when correct_ohmic):
        'auto'    -> min(Z_re) when the high-frequency arc is closed; when it
                     is OPEN, min(Z_re) still contains unresolved arc real
                     part and overestimates R_inf — the Re/Im are KK-linked,
                     so a too-high fixed R_inf makes the NNLS DRT unable to
                     fit the low-frequency tail.  Use the RC-series-fitted
                     R_inf instead (fallback: Eq. 10), clamped to min(Z_re).
        'min_zre' -> legacy behaviour: always min(Z_re).
        'manual'  -> use r_inf_value as given.
    arc_closure_tol : edge |Z_im| fraction above which an arc is deemed open.
    rc_misfit_tol   : max modulus-weighted relative RMSE for RC to be trusted.
    n_edge          : number of edge points used for the low-f linear slope.
    logger          : sink for diagnostic messages (R_inf ladder decision,
                      extrapolation method reasons). Defaults to ``print``;
                      the pipeline passes its log method so the lines reach
                      the GUI System Log (task #78; EIS_correction_kk
                      pattern). Wording is unpinned.
    lf_rc_open_tol  : low-frequency |Z_im| fraction above which the LF arc is
                      open enough (a diffusion tail) to warrant an RC low-
                      frequency tail. Higher than arc_closure_tol: RC on a
                      closed or marginally-open LF arc restructures the DRT
                      instead of extending it (drt-preprocessing 3.4).

    Returns (EIS_extrapolated, debug_dict, R_inf).
    """
    EIS_sorted, is_reversed = ensure_ascending_2d(EIS)
    f_meas    = EIS_sorted[:, 0]
    Z_re_meas = EIS_sorted[:, 1]
    Z_im_meas = EIS_sorted[:, 2]
    omega     = 2 * np.pi * f_meas

    # Two ohmic references.
    R_pol_kk   = _polarization_resistance_logscale(omega, Z_im_meas)  # Eq. (10) polarization resistance
    R_inf_eq10 = max(0.0, Z_re_meas[0] - R_pol_kk)                    # ohmic resistance from the KK integral
    R_inf_phys = _ohmic_highfreq(Z_re_meas)                           # high-freq Z_re limit

    method = str(method).strip().lower()
    if method not in ('auto', 'linear', 'rc'):
        raise ValueError(f"method must be 'auto', 'linear' or 'rc', got {method!r}")

    # Arc-closure diagnostic: an open arc at either edge means linear is unsafe.
    max_abs_im = np.max(np.abs(Z_im_meas)) + 1e-30
    lf_im_ratio = abs(Z_im_meas[0]) / max_abs_im
    low_open   = lf_im_ratio > arc_closure_tol
    high_open  = abs(Z_im_meas[-1]) / max_abs_im > arc_closure_tol
    # Strongly-open LF arc = a diffusion/mass-transport tail worth an RC tail.
    lf_strongly_open = lf_im_ratio >= lf_rc_open_tol

    # RC-series model fit: needed for the rc/auto extrapolation AND for the
    # open-arc R_inf estimate. Skipped when neither needs it (forced linear
    # with a closed HF arc, or manual/min_zre R_inf) to save the NNLS solve.
    r_mode = str(r_inf_mode).strip().lower()
    if r_mode == 'manual' and r_inf_value is None:
        logger("-> [R_inf] Warning: r_inf_mode='manual' but no r_inf_value "
              "given; falling back to automatic estimation.")
        r_mode = 'auto'
    need_rc = (method in ('rc', 'auto')) or (
        correct_ohmic and high_open and r_mode == 'auto')
    if need_rc:
        rc_ext_full, rc_x_opt, rc_R_inf_fit, rc_fit_misfit = _rc_series_extrapolation(
            f_meas, Z_re_meas, Z_im_meas, decades_low, decades_high, pts_per_decade)
    else:
        rc_ext_full, rc_x_opt = None, None
        rc_R_inf_fit, rc_fit_misfit = float('nan'), float('inf')

    # Target ohmic used to anchor the extrapolation and injected into the DRT
    # inversion. When the HIGH-frequency arc is open, min(Z_re) still contains
    # unresolved arc real part and overestimates R_inf; because Re and Im are
    # KK-linked, a too-high fixed R_inf makes the NNLS unable to fit the
    # low-frequency tail. Prefer the RC-model-fitted R_inf in that case.
    if r_mode == 'manual' and r_inf_value is not None:
        R_inf_target, r_inf_source = max(0.0, float(r_inf_value)), 'manual'
    elif not correct_ohmic:
        R_inf_target, r_inf_source = R_inf_eq10, 'Eq. 10 (correct_ohmic=False)'
    elif r_mode == 'min_zre' or not high_open:
        R_inf_target, r_inf_source = R_inf_phys, 'min Z_re'
    elif rc_fit_misfit <= rc_misfit_tol:
        R_inf_target = min(rc_R_inf_fit, R_inf_phys)
        r_inf_source = 'RC-model fit (HF arc open)'
    else:
        R_inf_target = min(R_inf_eq10, R_inf_phys)
        r_inf_source = 'Eq. 10 (HF arc open, RC fit unreliable)'
    if 'HF arc open' in r_inf_source:
        logger(f"-> [R_inf] high-frequency arc is open: min Z_re ({R_inf_phys:.4g}) "
              f"overestimates the ohmic resistance; using {r_inf_source} "
              f"R_inf = {R_inf_target:.4g}")

    def _run_linear():
        return _linear_extrapolation(f_meas, Z_re_meas, Z_im_meas, R_inf_target,
                                     decades_low, decades_high, pts_per_decade, n_edge)

    def _run_rc():
        # The measured window is preserved verbatim and the extrapolation tails
        # are anchored to it, so the spectrum already carries the data's physical
        # ohmic resistance — no global real-axis offset is applied (that would
        # shift the measured data).  R_inf is reported for reference only.
        R_inf_eff = R_inf_target if correct_ohmic else rc_R_inf_fit
        # Number of RC elements the NNLS actually used: R_m contributing more
        # than 0.1% of the total polarization resistance (NNLS zeroes the rest).
        R_m = rc_x_opt[2:]
        n_rc = int(np.sum(R_m > 1e-3 * (R_m.sum() + 1e-30)))
        return rc_ext_full, rc_x_opt, rc_R_inf_fit, R_inf_eff, rc_fit_misfit, n_rc

    debug = {'R_inf_eq10': R_inf_eq10, 'R_inf_phys': R_inf_phys,
             'R_inf_rc_fit': rc_R_inf_fit, 'r_inf_source': r_inf_source,
             'correct_ohmic': bool(correct_ohmic),
             'arc_open_low': bool(low_open), 'arc_open_high': bool(high_open)}

    if method == 'linear':
        EIS_ext = _run_linear()
        R_inf_out = R_inf_target
        debug.update(method='linear', R_inf=R_inf_out, reason='forced linear')
        logger(f"-> [Extrapolation] linear (Eqs. 8-10); R_inf = {R_inf_out:.4f}")

    elif method == 'rc':
        EIS_ext, x_opt, R_inf_fit, R_inf_eff, misfit, n_rc = _run_rc()
        R_inf_out = R_inf_eff
        debug.update(method='rc', fitted_model_params=x_opt, R_inf=R_inf_out,
                     R_inf_rc_fit=R_inf_fit, rc_misfit=misfit,
                     rc_n_elements=n_rc, reason='forced rc')
        logger(f"-> [Extrapolation] RC fit (Eqs. 11-12, plain NNLS); "
              f"R_inf = {R_inf_out:.4f}, misfit = {misfit:.3f}, "
              f"{n_rc} RC element(s)")

    else:  # auto -> run both, then select a method independently per edge
        lin_ext = _run_linear()
        rc_ext, rc_x, rc_fit, rc_eff, rc_misfit, rc_n = _run_rc()
        both_closed = not (low_open or high_open)

        # --- High-frequency tail + R_inf (unchanged decision) ---
        if both_closed:
            hf_method, hf_reason = 'linear', 'arcs closed at both edges (linear sufficient)'
            R_inf_out = R_inf_target
        elif high_open:
            # With the open-arc-corrected R_inf (above), the linear tail is
            # anchored to the right ohmic limit; an RC tail would impose
            # discrete-RC model structure on the FFT prior exactly at the
            # unresolved high-frequency edge, over-smoothing real HF peaks.
            hf_method = 'linear'
            hf_reason = 'high-frequency arc open: linear tail anchored to corrected R_inf'
            R_inf_out = R_inf_target
        elif rc_misfit <= rc_misfit_tol:
            hf_method, hf_reason = 'rc', f'low-frequency arc open & RC fits well (misfit={rc_misfit:.3f})'
            R_inf_out = rc_eff
        else:
            hf_method, hf_reason = 'linear', f'arc open but RC misfit high ({rc_misfit:.3f}); linear safer'
            # R_inf_target already reflects the open-arc-aware estimate above.
            R_inf_out = R_inf_target

        # --- Low-frequency tail (chosen independently of the HF choice) ---
        # Use the RC tail for the low-frequency end only when the RC model is
        # trustworthy AND the low-frequency arc is STRONGLY open (a diffusion /
        # mass-transport tail; lf_rc_open_tol). The openness precondition is
        # load-bearing: an RC tail continues an unclosed diffusive arc
        # physically, but on a closed — or only marginally open — low-frequency
        # arc there is nothing to continue, and the RC ladder's multi-element
        # structure restructures the DRT (a synthetic single-ZARC splits into
        # many peaks; a marginally-open arc merges two resolved peaks) rather
        # than merely extending it. A linear tail gently continues the settled
        # value and is correct there. Decided on the low-frequency arc alone,
        # not the HF arc, so a closed LF arc under an open HF arc stays linear.
        lf_method = ('rc' if rc_misfit <= rc_misfit_tol and lf_strongly_open
                     else 'linear')

        # Assemble: LF tail from lf_method; measured window + HF tail from
        # hf_method. Both extrapolations share the frequency grid, so the first
        # decades_low*pts_per_decade rows are exactly the low-frequency tail.
        base = rc_ext if hf_method == 'rc' else lin_ext
        lf_src = rc_ext if lf_method == 'rc' else lin_ext
        n_lo = decades_low * pts_per_decade
        EIS_ext = np.vstack((lf_src[:n_lo], base[n_lo:]))

        # Method token: the plain edge name when both tails agree, else an
        # explicit per-end token ('rc-LF/linear-HF' or 'linear-LF/rc-HF').
        chosen = lf_method if lf_method == hf_method else f'{lf_method}-LF/{hf_method}-HF'
        reason = f'{hf_reason}; LF tail -> {lf_method}'
        if 'rc' in (lf_method, hf_method):
            debug.update(fitted_model_params=rc_x)
            reason += f' [{rc_n} RC element(s)]'

        debug.update(method=chosen, lf_method=lf_method, hf_method=hf_method,
                     R_inf=R_inf_out, R_inf_linear=R_inf_target, R_inf_rc=rc_eff,
                     rc_misfit=rc_misfit, rc_n_elements=rc_n, reason=reason)
        logger(f"-> [Extrapolation] AUTO -> {chosen} ({reason}); "
              f"R_inf = {R_inf_out:.4f}  [linear R_inf={R_inf_target:.4f}, "
              f"RC R_inf={rc_eff:.4f}, RC misfit={rc_misfit:.3f}]")

    EIS_final = restore_order(EIS_ext, is_reversed)
    return EIS_final, debug, R_inf_out

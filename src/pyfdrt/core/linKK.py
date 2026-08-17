"""lin-KK measurement-model validity gate (drt-preprocessing.md 2.6, task #45).

Answers "is this measurement consistent with a causal, linear, time-invariant
system?" by fitting a Boukamp measurement model - ohmic offset, inductance,
and M sign-unconstrained RC elements on the measured tau window - and warning
when the modulus-normalized residuals exceed calibrated bands. The basis is
KK-compliant by construction, so finite-window truncation cancels and the
residual measures genuine KK violation (drift, nonlinearity, glitches), not
the window.

The Schoenleber mu-criterion is deliberately NOT used to choose M: on sparse
spectra (few sharp relaxations) mu oscillates with the parity of M because
tau-grid alignment with the true time constant flips adjacent R_k signs
(observed 1.00 -> -3.78 -> 0.02 -> 0.97 on RRC_simulation + 1 % noise). M is
instead grown by the discrepancy principle against a data-derived noise floor.
"""
import numpy as np

from ..io.eis_utils import ensure_ascending_2d

# Calibration (2026-07-23, 15-dataset corpus + injected drift/spike controls;
# evidence table in drt-preprocessing.md 2.6). Corpus maxima on valid data:
# rms 0.394 %, max 1.20 % -> both thresholds hold a >= 1.25x margin with zero
# false alarms, while 5 % drift (rms 0.76-0.81 %) and 3x5 % spikes
# (max 2.4-3.6 %) are caught.
RMS_WARN = 0.005          # warn when residual RMS exceeds 0.5 % of |Z|
MAX_WARN = 0.015          # warn when any point residual exceeds 1.5 % of |Z|
NOISE_FLOOR_MIN = 1e-4    # sigma floor: below this the discrepancy stop is noise-free
ELEMENTS_PER_DECADE = 7   # RC element density of the measurement model
M_MIN, M_MAX = 10, 50     # density cap bounds
MIN_POINTS = 8            # below this the fit is under-determined; gate skips


def estimate_noise_floor(re_n: np.ndarray, im_n: np.ndarray) -> float:
    """MAD-robust point-noise estimate from second differences.

    Inputs are the modulus-normalized spectrum components in ascending
    frequency order. For a smooth signal with iid point noise the second
    difference has variance 6*sigma^2; the median absolute deviation makes
    the estimate robust to spikes. Correlated (slow) error is invisible to
    second differences by design - it inflates the fit residual instead,
    which is exactly what the gate tests.
    """
    d2 = np.concatenate([np.diff(re_n, 2), np.diff(im_n, 2)])
    sigma = np.median(np.abs(d2 - np.median(d2))) / 0.6745 / np.sqrt(6.0)
    return max(float(sigma), NOISE_FLOOR_MIN)


def linkk_inductance(freq: np.ndarray, re: np.ndarray, im: np.ndarray) -> float:
    """Series-inductance estimate from the lin-KK measurement model.

    One modulus-weighted least-squares fit of R_ohm + j*w*L + sign-free RC
    elements at the gate's density cap (no discrepancy loop - the L column is
    read off, not the residual). The RC elements absorb the arc, so L is
    separated from arc curvature in a well-posed joint fit - unlike any
    fixed-window intercept fit on the KK residual, whose constant-baseline
    assumption arc curvature violates (task #57: window-narrowing alone left
    LIB_data's L at 1.16e-6 H vs ~0.7e-6 H actual).

    The returned L is sign-free and on clean spectra reflects noise (e.g.
    -2.6e-7 H on RRC_experimental), so callers MUST keep their own physical
    guard; EIS_correction_kk only consults this under its positive-Im-run
    guard (drt-preprocessing.md 2.1 step 4).
    """
    w = 2 * np.pi * np.asarray(freq, dtype=float)
    zmod = np.sqrt(re**2 + im**2)
    lo, hi = np.log10(1 / w.max()), np.log10(1 / w.min())
    M = int(np.clip(round(ELEMENTS_PER_DECADE * (hi - lo)), M_MIN, M_MAX))
    tau = np.logspace(lo, hi, M)
    den = 1 + (w[:, None] * tau[None, :])**2
    A = np.vstack([
        np.hstack([np.ones((len(w), 1)), np.zeros((len(w), 1)), 1 / den]),
        np.hstack([np.zeros((len(w), 1)), w[:, None],
                   -(w[:, None] * tau[None, :]) / den]),
    ])
    weight = np.concatenate([1 / zmod, 1 / zmod])
    x, *_ = np.linalg.lstsq(A * weight[:, None],
                            np.concatenate([re, im]) * weight, rcond=None)
    return float(x[1])


def kk_gate(EIS: np.ndarray) -> dict:
    """Run the warn-only lin-KK validity gate on a corrected spectrum.

    Parameters
    ----------
    EIS : (N, 3) array of [freq, Z_re, Z_im] in either sweep direction.

    Returns
    -------
    dict - either ``{'skipped': reason}`` or::

        {'ok': bool, 'reason': str or None, 'rms': float, 'max': float,
         'sigma': float, 'ratio': float, 'M': int, 'cap_hit': bool}

    ``rms``/``max`` are modulus-normalized residual statistics; ``ratio`` =
    rms/sigma is recorded for expert inspection but NOT gated (its valid-data
    baseline spans 0.4-19.8 across the corpus because the second-difference
    estimator misses correlated noise). The gate never modifies the data.
    """
    asc, _ = ensure_ascending_2d(np.asarray(EIS, dtype=float))
    if len(asc) < MIN_POINTS:
        return {'skipped': f'too few points (N={len(asc)} < {MIN_POINTS})'}

    freq, re, im = asc[:, 0], asc[:, 1], asc[:, 2]
    w = 2 * np.pi * freq
    zmod = np.sqrt(re**2 + im**2)
    sigma = estimate_noise_floor(re / zmod, im / zmod)

    lo, hi = np.log10(1 / w.max()), np.log10(1 / w.min())
    cap = int(np.clip(round(ELEMENTS_PER_DECADE * (hi - lo)), M_MIN, M_MAX))
    weight = np.concatenate([1 / zmod, 1 / zmod])
    y = np.concatenate([re, im])
    norm = np.concatenate([zmod, zmod])

    rms = rmax = M = None
    for M in range(3, cap + 1):
        tau = np.logspace(lo, hi, M)
        den = 1 + (w[:, None] * tau[None, :])**2
        A = np.vstack([
            np.hstack([np.ones((len(w), 1)), np.zeros((len(w), 1)), 1 / den]),
            np.hstack([np.zeros((len(w), 1)), w[:, None],
                       -(w[:, None] * tau[None, :]) / den]),
        ])
        x, *_ = np.linalg.lstsq(A * weight[:, None], y * weight, rcond=None)
        r = (y - A @ x) / norm
        rms = float(np.sqrt(np.mean(r**2)))
        rmax = float(np.max(np.abs(r)))
        if rms <= sigma:   # discrepancy principle: model has reached the noise
            break

    problems = []
    if rms > RMS_WARN:
        problems.append(f"residual RMS {rms:.2%} exceeds {RMS_WARN:.1%} of |Z|")
    if rmax > MAX_WARN:
        problems.append(f"max point residual {rmax:.2%} exceeds {MAX_WARN:.1%} of |Z|")

    return {
        'ok': not problems,
        'reason': ' and '.join(problems) if problems else None,
        'rms': rms, 'max': rmax, 'sigma': sigma, 'ratio': rms / sigma,
        'M': M, 'cap_hit': M == cap,
    }

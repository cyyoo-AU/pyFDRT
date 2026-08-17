import numpy as np
from scipy.integrate import trapezoid
from ..io.eis_utils import ensure_ascending_1d, restore_order


def kkim2re(freq: np.ndarray, data_im: np.ndarray):
    """Compute Z_re from Z_im using the Kramers-Kronig transform."""
    # Sort to ascending frequency for stable integration
    freq_sorted, (im_sorted,), is_reversed = ensure_ascending_1d(freq, data_im)

    omega = 2 * np.pi * freq_sorted

    # Broadcasting: W is target frequency (N x 1), X is integration variable (1 x N)
    W = omega[:, np.newaxis]
    X = omega[np.newaxis, :]

    Z_im_W = im_sorted[:, np.newaxis]
    Z_im_X = im_sorted[np.newaxis, :]

    numerator   = X * Z_im_X - W * Z_im_W   # x*Z_im(x) - omega*Z_im(omega)
    denominator = X**2 - W**2                # x^2 - omega^2

    # Avoid division by zero at poles (denominator == 0)
    integrand = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=integrand, where=(denominator != 0))

    # Cauchy principal value correction on the diagonal (x == omega)
    N = len(freq_sorted)
    if N > 2:
        for i in range(1, N - 1):
            integrand[i, i] = (integrand[i, i-1] + integrand[i, i+1]) / 2
        integrand[0, 0]   = integrand[0, 1]
        integrand[-1, -1] = integrand[-1, -2]

    # Log-scale integration: integral f(x) dx = integral f(x)*x d(ln x)
    integral  = trapezoid(integrand * X, x=np.log(omega), axis=1)
    result_re = (2 / np.pi) * integral

    return restore_order(result_re, is_reversed)


def kkre2im(freq: np.ndarray, data_re: np.ndarray):
    """Compute Z_im from Z_re using the Kramers-Kronig transform."""
    freq_sorted, (re_sorted,), is_reversed = ensure_ascending_1d(freq, data_re)

    omega = 2 * np.pi * freq_sorted

    W = omega[:, np.newaxis]
    X = omega[np.newaxis, :]

    Z_re_W = re_sorted[:, np.newaxis]
    Z_re_X = re_sorted[np.newaxis, :]

    numerator   = Z_re_X - Z_re_W   # Z_re(x) - Z_re(omega)
    denominator = X**2 - W**2

    integrand = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=integrand, where=(denominator != 0))

    N = len(freq_sorted)
    if N > 2:
        for i in range(1, N - 1):
            integrand[i, i] = (integrand[i, i-1] + integrand[i, i+1]) / 2
        integrand[0, 0]   = integrand[0, 1]
        integrand[-1, -1] = integrand[-1, -2]

    integral  = trapezoid(integrand * X, x=np.log(omega), axis=1)
    result_im = -(2 * W.flatten() / np.pi) * integral

    return restore_order(result_im, is_reversed)


def kk_correction(freq: np.ndarray, data_re: np.ndarray, data_im: np.ndarray):
    """Public KK-consistency validation utility (re-exported as
    ``pyfdrt.validation.kk_correction``; drt-core.md 2.4, task #46).

    Compare measured EIS data with KK predictions, scale-fit the KK result,
    and return corrected Re and Im free of inductive artefacts. Intended for
    MANUAL data-quality inspection: the gap between the returned corrected
    components and the measurement is a KK-consistency residual a user can
    examine directly.

    This function is deliberately NOT wired into the pipeline. Its direct-
    integration residual is unsuitable as an automated validity gate because
    finite-window truncation dominates it (task #45); the pipeline's automated
    gate is :func:`pyfdrt.validation.kk_gate`, which uses a truncation-immune
    lin-KK measurement model instead.
    """
    # Im -> Re via KK, then linear fit to measured Re
    re_from_im_by_kk   = kkim2re(freq, data_im)
    cpv, R_inf         = np.polyfit(re_from_im_by_kk, data_re, 1)
    re_corrected       = re_from_im_by_kk * cpv + R_inf

    # Re -> Im via KK, then linear fit to measured Im
    im_from_re_by_kk = kkre2im(freq, data_re)
    p                = np.polyfit(im_from_re_by_kk, data_im, 1)
    im_corrected     = im_from_re_by_kk * p[0] + p[1]

    return re_corrected, im_corrected, R_inf

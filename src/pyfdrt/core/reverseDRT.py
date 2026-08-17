import numpy as np


def reverse_drt(freq: np.ndarray, drt: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Reconstruct simulated EIS from a DRT spectrum via numerical integration.

    weights : optional per-point integration weights (not used in standard NNLS mode).
    """
    F_col = freq[:, np.newaxis]
    F_row = freq[np.newaxis, :]

    drt_eff       = drt * weights if weights is not None else drt
    Z_calc_matrix = drt_eff / (1 + 1j * F_col / F_row)
    Z_calc        = np.sum(Z_calc_matrix, axis=1)

    return np.column_stack((freq, np.real(Z_calc), np.imag(Z_calc), drt))

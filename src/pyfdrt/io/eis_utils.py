import numpy as np


def ensure_ascending_2d(EIS: np.ndarray) -> tuple:
    """
    Ensure the EIS 2D array is sorted in ascending frequency order.

    Returns:
        EIS_sorted (np.ndarray): sorted array
        is_reversed (bool): True if the original was descending (needed for restore_order)
    """
    is_reversed = EIS[0, 0] > EIS[-1, 0]
    EIS_sorted  = EIS[::-1, :].copy() if is_reversed else EIS.copy()
    return EIS_sorted, is_reversed


def ensure_ascending_1d(freq: np.ndarray, *data_arrays: np.ndarray) -> tuple:
    """
    Sort a 1-D frequency array and any number of companion data arrays to
    ascending order (used by kk_transform for stable 1-D operations).
    """
    is_reversed = freq[0] > freq[-1]
    if is_reversed:
        freq_sorted  = freq[::-1].copy()
        data_sorted  = tuple(arr[::-1].copy() for arr in data_arrays)
    else:
        freq_sorted  = freq.copy()
        data_sorted  = tuple(arr.copy() for arr in data_arrays)
    return freq_sorted, data_sorted, is_reversed


def restore_order(data_array: np.ndarray, is_reversed: bool) -> np.ndarray:
    """Reverse the array back to its original sweep direction if needed."""
    return data_array[::-1] if is_reversed else data_array


def validate_log_spacing(freq: np.ndarray, rtol: float = 1e-3):
    """
    Verify that the log-frequency spacing is uniform (required before FFT).
    Raises ValueError if the spacing is non-uniform.
    """
    d_logf = np.diff(np.log10(freq))
    if not np.allclose(d_logf, d_logf[0], rtol=rtol):
        raise ValueError(
            "Non-uniform log-frequency spacing detected. "
            "Run interpolateEIS before calling calcFDRT."
        )


def lcurve_corner(lambdas: np.ndarray,
                  residuals: np.ndarray,
                  roughnesses: np.ndarray) -> float:
    """
    Pick the L-curve corner: the lambda whose (log residual, log roughness)
    point lies farthest from the chord joining the two endpoints of the curve.

    Shared by DRTOptimizer.optimize_lambda (method='L-Curve') and
    dataExtrapolationKK so both stages use one consistent selection rule.

    Parameters
    ----------
    lambdas      : 1-D array of scanned regularisation strengths.
    residuals    : data-misfit norms ||A x - b||^2 for each lambda.
    roughnesses  : penalty-term norms (e.g. ||L x||^2 or ||R_m||^2) for each lambda.

    Returns
    -------
    float : the lambda located at the corner of the L-curve.
    """
    lambdas     = np.asarray(lambdas,     dtype=float)
    residuals   = np.asarray(residuals,   dtype=float)
    roughnesses = np.asarray(roughnesses, dtype=float)

    eps     = 1e-30  # guard against log(0)
    log_res = np.log(np.maximum(residuals,   eps))
    log_reg = np.log(np.maximum(roughnesses, eps))

    p1 = np.array([log_res[0],  log_reg[0]])
    p2 = np.array([log_res[-1], log_reg[-1]])
    chord = np.linalg.norm(p2 - p1)
    if chord == 0 or not np.isfinite(chord):
        # Degenerate curve: fall back to the mid-range lambda.
        return float(lambdas[len(lambdas) // 2])

    # Perpendicular distance of each point to the endpoint chord. The 2-D cross
    # product is the scalar determinant d[0]*v[1] - d[1]*v[0]; it is written out
    # explicitly because NumPy 2.0 deprecates the 2-D cross-product helper and
    # will remove it. The operand order matches that helper exactly (a[0]*b[1] -
    # a[1]*b[0]), so lambda_reg (a pinned golden quantity) stays bit-identical.
    # See task #26.
    d = p2 - p1
    distances = []
    for i in range(len(lambdas)):
        v = p1 - np.array([log_res[i], log_reg[i]])
        cross_z = d[0] * v[1] - d[1] * v[0]
        distances.append(np.abs(cross_z) / chord)
    return float(lambdas[int(np.argmax(distances))])

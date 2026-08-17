import numpy as np
from scipy.interpolate import interp1d


def interpolateEIS(EIS: np.ndarray, N: int) -> np.ndarray:
    """
    Interpolate EIS data onto N uniformly spaced points in log-frequency.
    Preserves the original sweep direction (high-to-low or low-to-high).
    """
    f_raw    = EIS[:, 0]
    log_f_raw = np.log10(f_raw)

    # Temporarily flip to ascending order for stable interpolation
    is_descending = log_f_raw[0] > log_f_raw[-1]
    if is_descending:
        x_fit = log_f_raw[::-1]
        y_fit = EIS[::-1, 1:3]
    else:
        x_fit = log_f_raw
        y_fit = EIS[:, 1:3]

    # Build interpolator (handles Re and Im simultaneously)
    interpolator = interp1d(x_fit, y_fit, axis=0, fill_value="extrapolate")

    # New x-axis keeps the original sweep direction
    new_log_f = np.linspace(log_f_raw[0], log_f_raw[-1], N)
    new_f     = 10 ** new_log_f
    new_z     = interpolator(new_log_f)

    return np.column_stack((new_f, new_z))

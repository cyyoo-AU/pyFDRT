import numpy as np

def find_freq_cutoff(flog: np.ndarray, Fz_log_mag: np.ndarray) -> tuple:
    """
    Extract the signal amplitude and the noise floor directly from the FFT data
    using simple statistics instead of a forced curve fit.
    (Prevents the collapse previously seen on multi-arc data.)
    """
    # 1. Smooth to stabilise spiky noise
    width = 15
    w = np.ones(width) / width
    avg = np.convolve(Fz_log_mag, w, mode='same')

    # Remove convolution boundary effects at both ends
    avg[:width] = avg[width]
    avg[-width:] = avg[-width-1]

    # 2. Extract the DC peak and the high-frequency noise floor directly
    peak_log = np.max(avg)
    floor_log = np.min(avg)

    amp_estimate = 10 ** peak_log

    # Return array: [amplitude, width (dummy), log10 noise level]
    param_final = np.array([amp_estimate, 1.0, floor_log])

    return param_final, {'smoothed_data': avg}

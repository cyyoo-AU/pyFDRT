import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift
from .findFreqCutoff import find_freq_cutoff
from ..io.eis_utils import ensure_ascending_2d, restore_order, validate_log_spacing


def sech(x):
    return 1 / np.cosh(x)


def execute_hammingfull(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.linspace(1, 0, N))


WINDOW_FUNCTIONS = {
    'hammingfull': execute_hammingfull,
}


def calcFDRT(EIS: np.ndarray, **kwargs) -> tuple:
    # Parameter initialisation
    params = {'windowfunction2': 'wiener', 'window2': 1, 'addjustnoise': 0}
    params.update(kwargs)

    # Safety checks: sort ascending and validate log spacing
    EIS_sorted, is_reversed = ensure_ascending_2d(EIS)
    validate_log_spacing(EIS_sorted[:, 0])

    logf = np.linspace(
        np.min(np.log10(EIS_sorted[:, 0])),
        np.max(np.log10(EIS_sorted[:, 0])),
        EIS_sorted.shape[0]
    )

    # Zero-padding for leakage reduction
    Z_im_raw = -EIS_sorted[:, 2]
    N_raw    = len(Z_im_raw)
    pad_len  = N_raw // 2
    N_pad    = N_raw + 2 * pad_len

    Z_pad    = np.pad(Z_im_raw, (pad_len, pad_len), mode='constant', constant_values=0)
    norm_pad = 1 / np.sqrt(N_pad)
    Fz       = norm_pad * fft(Z_pad)

    # Phase-aligned sech kernel centred at zero
    df       = np.mean(np.diff(np.log10(EIS_sorted[:, 0])))
    x_linear = (np.arange(N_pad) - N_pad // 2) * df * np.log(10)
    s_pad    = 0.5 * sech(x_linear)

    # ifftshift moves the peak to index 0, eliminating phase wrap
    s_pad_shifted = ifftshift(s_pad)
    Fs            = norm_pad * fft(s_pad_shifted)

    # Wiener filter deconvolution
    if params['windowfunction2'] == 'wiener':
        logf_min    = np.log10(EIS_sorted[0, 0])
        f_pad_log   = np.linspace(logf_min - pad_len * df,
                                  logf_min + (N_raw + pad_len - 1) * df,
                                  N_pad)
        param, _    = find_freq_cutoff(f_pad_log, np.log10(np.abs(Fz)))

        noise_level = 10 ** (param[2] + params.get('addjustnoise', 0))
        yestimate   = Fs * param[0] / Fs[0]
        wienfilter  = np.abs(yestimate)**2 / (np.abs(yestimate)**2 + noise_level**2)

        Fzfil       = wienfilter * Fz
        win_func    = WINDOW_FUNCTIONS['hammingfull']
        window_array = win_func(N_pad)

        Fs_safe  = np.where(np.abs(Fs) < 1e-12, 1e-12, Fs)
        ratiofil = ifftshift(window_array * fftshift(Fzfil / Fs_safe))
    else:
        Fs_safe  = np.where(np.abs(Fs) < 1e-12, 1e-12, Fs)
        ratiofil = Fz / Fs_safe

    # IFFT and extract the valid central region (strip padding)
    drtfil_pad = ifft(ratiofil)
    drtfil     = drtfil_pad[pad_len : pad_len + N_raw]

    # Assemble output and restore original sweep direction
    EISg_sorted = np.column_stack((EIS_sorted, np.real(drtfil).T))
    EISg        = restore_order(EISg_sorted, is_reversed)

    debug = {
        'drt_raw':      np.real(drtfil),
        'padded_length': N_pad
    }

    return EISg, debug

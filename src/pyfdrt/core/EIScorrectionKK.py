import numpy as np
from .kk_transform import kkre2im
from .linKK import linkk_inductance
from ..io.eis_utils import ensure_ascending_2d, restore_order


def EIS_correction_kk(EIS: np.ndarray, fit_points: int = 10, logger=print,
                      force: bool = False) -> tuple:
    """
    Automatically detect and remove inductive artefacts from EIS data.

    The algorithm estimates the inductive coefficient L from the high-frequency
    tail.  Correction is applied only when L is physically meaningful and its
    contribution exceeds 0.5 % of the impedance modulus at the highest frequency.

    Parameters
    ----------
    logger : callable, optional
        Sink for diagnostic messages (skip reason / applied correction).
        Defaults to ``print`` so CLI/notebook behaviour is unchanged; the GUI
        passes its own log method so the reason is shown in the System Log panel.
    force : bool, optional
        Bypass the 0.5 % significance threshold (gate B), applying the fitted
        inductance even when its contribution is negligible.  This is what the
        GUI's "Force ON" mode selects.

        The flag changes ONLY that admission decision (task #70): the
        estimation chain — the primary origin-constrained fit and the
        short-tail recovery — runs identically in both modes, and the
        recovery trigger always uses the Auto admission test so that forcing
        cannot short-circuit it.

        The physical-validity check (gate A, ``L <= 0``) is NOT bypassed, and
        deliberately so: subtracting ``j*omega*L`` with a negative L *adds* an
        inductive term, fabricating an artefact that was never measured.  A
        "force" switch may override a significance heuristic, but it must not
        override physics — so a non-positive L still skips the correction.
    """
    EIS_sorted, is_reversed = ensure_ascending_2d(EIS)
    freq  = EIS_sorted[:, 0]
    re    = EIS_sorted[:, 1]
    im    = EIS_sorted[:, 2]
    omega = 2 * np.pi * freq

    # Estimate L from the residual between measured Im and the KK prediction over
    # the highest `fit_points`, with the historical origin-constrained fit. This
    # stays the primary estimate: it already handles spectra with a clear inductive
    # tail (e.g. SCN, SOFC), and the intercept recovery below would over-correct
    # their longer tails into KK-inconsistency.
    im_kk     = kkre2im(freq, re)
    omega_fit = omega[-fit_points:]
    im_diff   = im[-fit_points:] - im_kk[-fit_points:]
    L_coeff   = np.sum(omega_fit * im_diff) / np.sum(omega_fit ** 2)

    # Recovery (task #54): the origin-constrained fit mis-signs a SHORT inductive
    # tail sitting on a capacitive baseline — it folds the baseline into a spurious
    # slope — so it misses a real inductance the data plainly shows as a run of
    # positive Im at the high-frequency end (a purely capacitive spectrum has
    # Im < 0 throughout). When the origin-constrained estimate would NOT be
    # admitted (non-positive, or negligible under the same test as gate B) yet
    # there are at least MIN_INDUCTIVE_POINTS consecutive positive-Im points, re-fit
    # with an intercept so the KK baseline is absorbed by the constant term instead
    # of the slope. The positive-Im run is an essential guard: without it the
    # intercept fabricates inductance on clean capacitive spectra.
    MIN_INDUCTIVE_POINTS = 3
    _max_Z_mod  = np.sqrt(re[-1] ** 2 + im[-1] ** 2)
    # The recovery trigger deliberately uses the AUTO admission test regardless
    # of `force` (task #70): forcing must not short-circuit the recovery. With
    # `force` in this test, a weak positive primary estimate was admitted
    # directly and the recovery never ran, so Force ON delivered a 10-14x
    # smaller L than Auto on the LIB pair - the mode meant to force a
    # correction forced a worse one. `force` affects gate B only, below.
    _admissible = L_coeff > 0 and L_coeff * omega[-1] >= 0.005 * _max_Z_mod
    if not _admissible:
        n_positive = 0
        for v in im[::-1]:
            if v > 0.0:
                n_positive += 1
            else:
                break
        if n_positive >= MIN_INDUCTIVE_POINTS:
            # Recovery estimate from the lin-KK measurement model (task #57).
            # Any fixed-window intercept fit on the KK residual assumes a
            # constant baseline, which arc curvature violates even inside the
            # positive-Im run itself: the original max(n_positive, fit_points)
            # window over-estimated LIB's L by ~40% (1.2e-6 H vs ~0.7e-6 H),
            # narrowing to the run alone still gave 1.16e-6 H, and the
            # over-subtracted residue was the root cause of the high-frequency
            # NNLS mismatch. The measurement-model fit carries sign-free RC
            # elements that absorb the arc, so L is separated from curvature;
            # the positive-Im run above remains the essential physical guard
            # (the sign-free fit reads noise as L on clean spectra).
            L_recover = linkk_inductance(freq, re, im)
            if L_recover > 0:
                L_coeff = L_recover

    # Gate A: negative inductance is unphysical — skip correction.
    # NOT overridable by force: applying it would add a fabricated artefact.
    if L_coeff <= 0:
        tag = "Force" if force else "Auto"
        logger(f"-> [{tag}] No inductive tail detected (L = {L_coeff:.3e} H ≤ 0). "
               "Correction skipped"
               + (" (force does not override the physical-validity check)."
                  if force else "."))
        return EIS.copy(), 0.0  # a copy: mutating the result must not touch the input

    # Gate B: ignore L if its contribution is below 0.5 % of |Z| at f_max.
    # Overridden by force=True, which applies even a negligible correction.
    max_L_impedance = L_coeff * omega[-1]
    max_Z_modulus   = np.sqrt(re[-1]**2 + im[-1]**2)
    if max_L_impedance < max_Z_modulus * 0.005:
        if not force:
            logger(f"-> [Auto] Inductive contribution ({max_L_impedance:.4f} Ω) is negligible "
                   f"(< 0.5 % of |Z|); L = {L_coeff:.3e} H. Correction skipped.")
            return EIS.copy(), 0.0  # a copy: mutating the result must not touch the input
        logger(f"-> [Force] Inductive contribution ({max_L_impedance:.4f} Ω) is negligible "
               f"(< 0.5 % of |Z|), but force=True; applying anyway. L = {L_coeff:.3e} H.")

    # Apply correction: subtract inductive term from imaginary part
    logger(f"-> [{'Force' if force else 'Auto'}] Significant inductive tail detected "
           f"(L = {L_coeff:.4e} H). Applying correction.")
    EIS_corrected       = EIS_sorted.copy()
    EIS_corrected[:, 2] = im - (L_coeff * omega)

    return restore_order(EIS_corrected, is_reversed), L_coeff

"""The extracted circuit as a Boukamp Circuit Description Code (drt-io.md 5.4).

One line of text describing the circuit the analysis reported, in the notation
of B. A. Boukamp, Solid State Ionics 20 (1986) 31: series elements are
concatenated, and parentheses enclose a parallel combination.

It lives in the package rather than in a front-end because each front-end
assembles its own run summary. Three copies of this derivation would let the
desktop window, the browser workbench and the standard browser surface describe
one circuit three different ways, which is the failure task #120 removed for the
shared drawing.

Nothing here computes: every code is read off ``results``.
"""

# Element codes. Boukamp's letters, so a string labelled "Boukamp CDC" is one.
CODE_RESISTOR = 'R'
CODE_CPE = 'Q'
CODE_INDUCTOR = 'L'

# The finite-length Warburg with a REFLECTIVE (open) boundary. Checked against
# DRTOptimizer._warburg_open's formula rather than its name: it computes
# R_W*coth(sqrt(j w tau))/sqrt(j w tau), whose low-frequency limit is
# R_W/3 + 1/(j w C_lim) - the blocking element, which is Boukamp's 'O'. 'W' is
# the semi-infinite element and 'T' the transmissive tanh form; neither is
# implemented here. Some current packages spell this 'Wo' (with 'Ws' for the
# short form); that is a different convention, and the constant exists so the
# choice is made in one place rather than scattered through format strings.
CODE_WARBURG_OPEN = 'O'


def circuit_cdc(results, inductance: float = 0.0) -> str:
    """The reported circuit as a CDC string, e.g. ``LR(RQ)(RQ)O``.

    ``results`` is the dict returned by ``deconvolve_peaks`` / the pipeline.
    ``inductance`` is the series inductance the preprocessing adopted
    (``L_val``); pass it so the code describes the measurement rather than the
    corrected spectrum. It is **not** read from ``results``: the golden
    snapshots serialise that dict wholesale, so adding a key there would move
    every pinned file to carry a number `L_val` already pins separately.

    Returns ``R`` alone when no element was reported - a spectrum that
    resolved nothing still has its series resistance, and an empty string
    would read as a failure rather than as a result.
    """
    if not results:
        return CODE_RESISTOR

    parallel = CODE_RESISTOR + CODE_CPE          # the 'RQ' inside every block
    parts = []
    # Only an adopted inductance is written. A zero means the preprocessing
    # gates rejected it - unphysical, or below 0.5 % of |Z| at f_max
    # (drt-preprocessing.md 2) - and an `L` in the code would assert an element
    # the analysis declined to accept.
    if inductance and inductance > 0:
        parts.append(CODE_INDUCTOR)
    parts.append(CODE_RESISTOR)                  # R_inf, always in series

    # Every block is (RQ), never (RC). A refined Phi of exactly 1.0 is the
    # refinement on its upper bound, not a measured ideality, and
    # drt-deconvolution.md 2.4.2 forbids reading it as one. Four corpus spectra
    # sit on that ceiling; naming them ideal capacitors here would put the
    # forbidden reading into the string most likely to be quoted.
    parts += [f'({parallel})' for _ in results.get('Peaks', []) or []]

    warburg = results.get('Warburg')
    if warburg:
        # The composite (drt-deconvolution.md 2.4.2) carries a slow relaxation
        # in series with the diffusion branch. It is not a peak, so it is
        # absent from Peaks above and must be added here or the circuit the
        # string describes is not the circuit that was fitted.
        if 'slow_zarc' in warburg:
            parts.append(f'({parallel})')
        parts.append(CODE_WARBURG_OPEN)

    return ''.join(parts)

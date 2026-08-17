"""The built-in reference spectrum (drt-io.md 2.5, task #122).

A synthetic ``R1(R2*CPE1)(R3*CPE2)`` spectrum, so a front-end has something to
analyze for a user who arrives without a file of their own.

It is generated from its parameters rather than shipped as a data file. A
stored file would carry its description in a header comment that nothing keeps
true: edit the numbers and the comment still reads the same, and the example
starts lying about what it is. Here the constants below are both the
documentation and the source of the data, so the two cannot disagree.
"""

import numpy as np

# Where the parameter set comes from. Recorded because an example spectrum
# with no provenance invites the reader to assume it was chosen to flatter the
# software; this one was not chosen here at all.
SOURCE = "B. A. Boukamp, J. Phys. Energy 2020, 2, 042001"

# Series (ohmic) resistance, ohm.
R1 = 5.0e3

# First R-CPE element: resistance (ohm), CPE magnitude (S*s^phi), exponent.
R2, Q1, PHI1 = 10.0e3, 1.0e-6, 0.75

# Second R-CPE element, same units.
R3, Q2, PHI2 = 20.0e3, 2.0e-8, 0.85

# Measured window and density. 10 points per decade over 6 decades.
F_MAX, F_MIN, POINTS_PER_DECADE = 1.0e5, 1.0e-1, 10

# The two characteristic times land 1.33 decades apart, which is the reason
# this particular parameter set is the built-in one. Wider and the example
# stops exercising what the ZARC basis is for; narrower and it would sit under
# the >= 1-decade resolution limit of drt-deconvolution.md and demonstrate a
# documented failure instead of ordinary operation.


def reference_spectrum() -> np.ndarray:
    """The reference spectrum as the ``(N, 3)`` array of drt-io.md 1.

    Column 2 is the actual imaginary part, negative for capacitive points,
    matching every loader.
    """
    n = int(round(np.log10(F_MAX / F_MIN) * POINTS_PER_DECADE)) + 1
    freq = np.logspace(np.log10(F_MAX), np.log10(F_MIN), n)
    jw = 1j * 2.0 * np.pi * freq
    z = (R1
         + R2 / (1.0 + R2 * Q1 * jw ** PHI1)
         + R3 / (1.0 + R3 * Q2 * jw ** PHI2))
    return np.column_stack([freq, z.real, z.imag])


def reference_spectrum_text() -> str:
    """The same data as loader-compatible headed text (drt-io.md 2.2).

    Front-ends load this rather than passing the array in directly, so the
    example goes through the same parse, monotonicity check and sign handling
    as a user's own file instead of succeeding by a private route.
    """
    data = reference_spectrum()
    lines = ["Freq(Hz)\tZ_re(ohm)\tZ_im(ohm)"]
    lines += [f"{f:.10g}\t{re:.10g}\t{im:.10g}" for f, re, im in data]
    return "\n".join(lines) + "\n"


def reference_spectrum_summary() -> str:
    """One line naming the circuit, for a front-end to show beside the button.

    Deliberately limited to construction parameters. What the pipeline recovers
    from this spectrum is not stated anywhere in the package: no golden pins
    it, so any such claim would go stale without failing (drt-io.md 2.5).
    """
    return (f"R1(R2·CPE1)(R3·CPE2)  ·  R1 = {R1/1e3:.0f} kΩ  ·  "
            f"R2 = {R2/1e3:.0f} kΩ, Q1 = {Q1:g} S·s^φ, φ1 = {PHI1}  ·  "
            f"R3 = {R3/1e3:.0f} kΩ, Q2 = {Q2:g} S·s^φ, φ2 = {PHI2}  ·  "
            f"{F_MAX:g} to {F_MIN:g} Hz  ·  {SOURCE}")

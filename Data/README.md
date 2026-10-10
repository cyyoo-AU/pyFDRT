# Data provenance

## `EIS_RRQRQ_boukamp2016.txt` — the worked example

A **synthetic** spectrum, computed from a published circuit and parameter set
rather than measured or copied. The parameters are those of

> B. A. Boukamp, *J. Phys. Energy* **2020**, 2, 042001.

for the circuit `R1(R2·CPE1)(R3·CPE2)`:

| Element | Value |
|---|---|
| `R1` | 5 kΩ |
| `R2`, `Q1`, `φ1` | 10 kΩ, 1×10⁻⁶ S·sᶲ, 0.75 |
| `R3`, `Q2`, `φ2` | 20 kΩ, 2×10⁻⁸ S·sᶲ, 0.85 |

Sampled from 100 kHz to 0.1 Hz at 101 points, with about 0.1 % noise added, in
the three-column format the loaders expect (`f (Hz)`, `Z' (Ω)`, `Z'' (Ω)`).
Recomputing the impedance from the parameters above reproduces the file to
**0.11 % rms** and 0.33 % at its worst point, and its limits are exact: 5170 Ω
at 100 kHz and 35 kΩ at 0.1 Hz, the latter being `R1 + R2 + R3`.

Parameters published in a paper are facts about a circuit, and the numbers in
this file were produced here by evaluating them. Nothing in it is reproduced
from the paper.

**The `2016` in the filename is not the year of that paper.** It is carried
over from an earlier name and is kept only because the regression goldens,
the specifications and the manual all refer to the file by it; renaming it
would rewrite a pinned corpus entry to fix a word. Read the citation above,
not the filename.

The same parameter set is also built into the package as the
`Load Example Spectrum` button's data (`pyfdrt.io.reference_spectrum`), which
generates it at 10 points per decade instead of this file's 16.7 and adds no
noise.

## The other files here

Everything else in this directory is the development corpus: measured spectra
and simulations used by the regression suite, which lives in the development
repository. It is not part of the distribution, and the benchmark collections
under `Benchmark/` carry their own provenance and licence notes
(`Benchmark/Ciucci/README-tutorial-data.md` for the pyDRTtools tutorial data,
`Benchmark/EISART/README.md` and `LICENSE-EISART.txt` for the EISART samples).

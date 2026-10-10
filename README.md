<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/pyFDRT_logo_dark.png">
  <img src="assets/pyFDRT_logo.png" alt="pyFDRT" width="420">
</picture>

**An open-source Python tool for Fourier-transform analysis of the
distribution of relaxation times in electrochemical impedance spectra.**

A desktop application and a batch command line for turning an impedance
spectrum into a distribution of relaxation times and an equivalent circuit —
with the checks that tell you when not to trust the result.

---

## What it does

Inductance correction, frequency-window extrapolation and ohmic-resistance
estimation; an FFT/Wiener DRT estimate polished by a regularised non-negative
inversion; peak deconvolution into R-CPE elements with information-criterion
model selection; and a diffusion branch fitted when the spectrum shows one.

Every run also reports a **Kramers-Kronig validity check** and an **R-budget
consistency check**. These do not stop a run — they tell you whether the
measurement satisfied the theory's premises, and whether the reported circuit
describes what the measured window actually contains.

## Install

Python 3.10 or newer.

```bash
python -m pip install -e ".[gui]"
```

Omit `[gui]` if you only want the batch command line.

## Run it

**Desktop application**

```bash
python scripts/main_app.py
```

Click **🧪 Load Example Spectrum** to see a complete run without supplying a
file. The built-in spectrum is generated from its parameters rather than read
from disk, so what the program analyses and what it claims to be cannot drift
apart. Its parameter set is from B. A. Boukamp, *J. Phys. Energy* **2020**, 2,
042001.

**Batch command line**

```bash
python scripts/run_batch.py --input_dir Data --output_dir Results
```

Each input file produces an Excel workbook and a plot. A file that fails does
not stop the batch; the closing summary names every failure.

## Input format

Three columns — frequency in Hz, real impedance, imaginary impedance — as
whitespace- or comma-delimited text, with or without a header, in `.txt`,
`.csv` or `.dat`; ZView `.z` files are also read. Frequencies must be
monotonic.

`Data/EIS_RRQRQ_boukamp2016.txt` is included as a worked example. It is a
**synthetic** spectrum computed from the `R1(R2·CPE1)(R3·CPE2)` parameter set
of B. A. Boukamp, *J. Phys. Energy* **2020**, 2, 042001 — the `2016` in the
filename is not the year of that paper and is kept only because the
documentation and the regression goldens refer to the file by that name.
[`Data/README.md`](Data/README.md) carries the parameters and the agreement
with them.

## Documentation

**[docs/manual.md](docs/manual.md)** — the user manual: every control, the two
click-editing modes, how to read the diagnostics, what the exported workbook
contains, and the batch arguments.

## Specifications and the regression corpus

This repository holds the shipped product. The design specifications, the
golden regression corpus that pins the numerical behaviour, and the test suite
are maintained in a separate development repository — they are not omitted
because they do not exist. Write to the address below if you need them, for
review or for reproducing published numbers.

**Contributing a change?** Read [`DISTRIBUTE.md`](DISTRIBUTE.md) first. Edits
made here are overwritten by the next release and never reach the development
repository, so a fix has to start there instead.

## Citing

The accompanying paper is in press at the *Bulletin of the Korean Chemical
Society*.

```bibtex
@article{pyFDRT2026,
  author  = {Yoo, Chung-Yul and Park, Jaeku and Kim, Jeongjin},
  title   = {pyFDRT: an open-source Python tool for Fourier-transform
             analysis of the distribution of relaxation times in
             electrochemical impedance spectra},
  journal = {Bulletin of the Korean Chemical Society},
  year    = {2026},
  doi     = {10.1002/bkcs.70232},
  note    = {in press}
}
```

Add the volume and pages to the entry once the issue is assigned.

## Authors and contact

Chung-Yul Yoo · Jaeku Park · Jeongjin Kim

Please write to <cyyoo@ajou.ac.kr> for academic and industrial collaboration.

## License

MIT. See [LICENSE](LICENSE).

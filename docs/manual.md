# pyFDRT — User Manual

Distribution of Relaxation Times (DRT) analysis for electrochemical impedance
spectroscopy. This manual covers the two ways you run it: the **desktop
application** and the **batch command line**.

It describes what the software does. Where a number or a label appears here it
was read from the running program, not assumed — the control labels in
particular are matched whole-value by the configuration layer, which rejects
anything else with an error rather than falling back to a default.

---

## 1. Installing

You need Python 3.10 or newer.

```bash
python -m pip install -e ".[gui]"
```

That installs the `pyfdrt` package with the desktop application's Qt
dependency. Omit `[gui]` if you only intend to use the batch command line.

The analysis itself needs numpy, scipy, pandas, openpyxl and matplotlib; pip
pulls them in. The desktop application additionally needs PyQt6.

> **A note on numerical reproducibility.** The dependency declarations carry no
> upper bounds, so a fresh install takes whatever numpy and scipy are current.
> Results move slightly between BLAS builds. If you need to reproduce published
> numbers exactly, pin the versions your reference results were produced with.

---

## 2. Starting the desktop application

```bash
python scripts/main_app.py
```

The window is titled **pyFDRT v1.0**. It has a control panel on one side and
two plot panels on the other, with a System Log beneath them.

### 2.1 Starting without a file

Click **🧪 Load Example Spectrum**.

This loads a built-in synthetic spectrum — an `R1(R2·CPE1)(R3·CPE2)` circuit,
10 points per decade over six decades from 100 kHz to 0.1 Hz. It is generated
from its parameters at load time rather than read from a file, so what the
program analyses and what the documentation says it is cannot drift apart.

Its parameter set is taken from B. A. Boukamp, *J. Phys. Energy* **2020**, 2,
042001. The two characteristic times sit 1.33 decades apart, which is far
enough that the analysis is doing ordinary work rather than demonstrating a
documented resolution limit.

Use it to see a complete run before committing your own data to the process.

### 2.2 Loading your own spectrum

Click **📁 Load Data**. The file dialog offers:

| Filter | Extensions |
|---|---|
| Data Files | `.txt`, `.csv`, `.dat`, `.z` |
| ZView Files | `.z` |
| All Files | everything |

The loader accepts three columns — frequency in Hz, real impedance, imaginary
impedance — as whitespace- or comma-delimited text, with or without a header
line, and it reads ZView `.z` files. Frequencies must be monotonic; a file
whose frequency column wanders is rejected by name rather than analysed.

---

## 3. The control panel

Two groups, in the order the panel presents them. Every control has a tooltip;
this section says what the settings mean and which are safe to leave alone.

**The short answer: leave them alone.** The defaults below are the configuration
the reference results are generated with. Change one when you have a reason you
could state, and read §5 to see whether the run agreed with you.

### 3.1 ① Preprocessing (Inductance, Extrapolation, R∞)

| Control | Options | Default |
|---|---|---|
| **Inductance (L)** | `Auto (Smart Detection)`, `Force ON (Force Correction)`, `Force OFF (Keep Original)` | Auto |
| **Ext. range (dec)** | 0 – 5 | 2 |
| **Ext. method** | `Auto (Linear → RC fallback)`, `Linear`, `RC fit` | Auto |
| **R∞ estimate** | `Auto (open-arc corrected)`, `Min Z_re (legacy)`, `Manual` | Auto |

**Inductance** removes the series inductance of the leads and cell hardware.
Auto detects whether there is an inductive tail at all; when there is none the
log says so and skips the correction rather than fitting noise.

**Ext. range** extends the measured window before inversion, in decades, on
both ends. `0` means no extension and is supported. Extension exists because
the transform needs data beyond the measured edges; without it the DRT near
the window boundaries is unreliable.

**Ext. method** chooses how that extension is built. Auto decides per end —
linear where the arcs are closed, an RC fit where the low-frequency arc is
still open — and the log states which it chose and why.

**R∞ estimate** is the high-frequency ohmic resistance. `Manual` enables the
numeric field beside it, in ohms.

### 3.2 ② Fitting (λ, Deconvolution, Warburg)

| Control | Options | Default |
|---|---|---|
| **λ mode** | `Auto (L-Curve)`, `Manual` | Auto |
| **λ value** | 1e-6 – 1000, log slider | 0.01 (Manual only) |
| **Criterion** | `Conservative (BIC)`, `Flexible (AICc)` | Conservative (BIC) |
| **Deconvolution** | `ZARC + BIC`, `LASSO + Curve-Fit`, `CWT + BIC` | ZARC + BIC |
| **Max peaks** | 1 – 15 | 10 |
| **Warburg** | `Auto (With Tail-Cut)`, `Joint (NNLS)`, `Off` | Auto (With Tail-Cut) |

**λ** is the smoothing strength. Too little and the DRT breaks into spikes; too
much and neighbouring processes merge into one. Auto finds the corner of the
L-curve. The value it adopted appears in the log and in the run summary — worth
reading even when you did not choose it.

**Criterion** decides how readily model selection accepts another peak.
Conservative (BIC) penalises extra peaks harder than Flexible (AICc).

**Deconvolution** picks the basis fitted to the DRT. `ZARC + BIC` is the
validated default. The other two are cross-check bases: when a feature survives
all three, it is not an artefact of one basis.

**Max peaks** bounds the model-selection loop. Raising it does not force more
peaks; the criterion still has to justify each one.

**Warburg** controls the diffusion element. `Auto (With Tail-Cut)` fits one only
when a diffusion tail is actually detected — see §5.4 for what that costs.
`Joint (NNLS)` always fits one, which on a spectrum without diffusion invents an
element that is not there. `Off` never fits one.

### 3.3 Resetting

**↺ Reset to Validated Defaults** returns every control to the table values
above. Use it after exploring, before a run whose numbers you intend to keep.

---

## 4. Running

Click **🚀 Run DRT Optimization**.

The System Log narrates the run in stages: preprocessing decisions, the noise
filter auto-tuning, the λ search, the NNLS polish, then peak deconvolution and
the extracted circuit parameters. A one-line run summary closes it.

Two plot panels update: the Nyquist view with the measured points and the
reconstruction, and the DRT spectrum itself with the deconvolved peaks.

### 4.1 The two click modes

Both are toggles, off by default. **Both can be on at the same time** — they do
not conflict, because each listens to a different panel:

| Mode | Panel it listens to |
|---|---|
| ✍️ Manual Peak Edit | the **DRT** plot |
| ✂️ Tail-Cut | the **Nyquist** plot |

Both require a completed run first; clicking before one tells you so and does
nothing.

#### ✍️ Manual Peak Edit Mode — on the DRT plot

**Left-click** adds a peak at the clicked relaxation time. **Right-click**
removes the nearest one, but only if it is within 0.5 in log(τ) of the click —
a right-click further away is ignored silently, so a stray click cannot delete
a peak you meant to keep.

Each edit re-fits immediately and logs what changed:

```
-> [Manual Edit] Peak added: log(tau) = -3.10
-> [Manual Edit] Peak removed: log(tau) = -3.10
```

Note that adding a peak marks a location for the fit; it does not guarantee the
reported peak count rises, because the refit still has to place an element
there.

#### ✂️ Tail-Cut Mode — on the Nyquist plot

**Left-click a measured point** to exclude every frequency below it. The click
snaps to the nearest measured point within 20 pixels, so it is the point you
see that is used, not the exact pixel you hit. Click further away than that and
nothing is cut:

```
-> [Tail-Cut] No data point near the click; click closer to a measured point.
```

**Right-click** discards the manual cut and restores automatic detection. The
button toggle does not clear it — right-click is the way back.

The cut applies to both the peak deconvolution and the impedance-domain ZARC
refinement, and it re-deconvolves at once, reusing the preprocessing and the
NNLS polish so the feedback is immediate. On the built-in example spectrum,
cutting below 3.16 kHz took the run from two peaks to one, and the right-click
restored both:

```
-> [Tail-Cut] Manual cut set: excluding f < 3.16e+03 Hz (tau > 5.03e-05 s)
-> [Tail-Cut] Re-deconvolution complete: 1 peak(s)
-> [Tail-Cut] Manual cut cleared; automatic detection restored.
-> [Tail-Cut] Re-deconvolution complete: 2 peak(s)
```

A cut that would leave too few points is rejected and automatic detection is
restored, rather than the run proceeding on an unusable remainder.

### 4.2 The run summary, token by token

The summary line that closes a run is the densest thing the program shows:

```
✓ 2 peaks · R(RQ)(RQ) · L=0 · λ=1.301 (auto) · ext=LINEAR · Warburg: none · Rb: 100% · KK: PASS (0.02%)
```

Most spectra only ever show one form of each token, so the alternatives below
are easy to meet for the first time on the one measurement that matters. Every
form the program can print is listed here. In the browser build each token also
explains itself on hover, and the **?** button opens the same text as a list.

| Token | Form | Meaning, and when you see it |
|---|---|---|
| **peaks** | `2 peaks` | Relaxation processes resolved inside the measured window. The count comes from the model-selection criterion (§3.2), not from a threshold. Singular for one. |
| **circuit** | `R(RQ)(RQ)` | The extracted circuit in Boukamp circuit-description code. Series elements are written in sequence; parentheses enclose a parallel pair. `R` is the series resistance, and each `(RQ)` is one relaxation — a resistor in parallel with a constant-phase element. |
| | `LR(RQ)(RQ)` | A leading `L` is the series inductance, present only when one was adopted — it is the same value the `L` token reports. Where the token reads `L=0` the code has no `L`, because the estimate was rejected as unphysical or as too small to matter. |
| | `R(RQ)(RQ)(RQ)(RQ)O` | A trailing `O` is the finite-length Warburg with a reflective boundary, the diffusion branch. Some other packages write this element `Wo`; it is the same element. Where the diffusion record is the composite of §3.6, its slow relaxation is the `(RQ)` immediately before the `O` — it is part of the circuit but is not a peak, so it is not in the parameter table. |
| | | Every block is written `(RQ)`, never `(RC)`, even when the reported φ is exactly 1. A φ of 1.000 is the refinement resting on its upper bound rather than a measured ideal capacitor, and the notation does not make a claim the fit did not make. Read φ in the table, with the caveats of §3.5. |
| | | The code describes what was extracted. **No circuit was searched for or selected** — the topology is fixed by the method and only the number of elements is chosen (§3.2). If you know this notation from a fitting program, that difference matters. |
| **L** | `L=0` | No series inductance was removed — either none was detected, or its contribution was below 0.5 % of \|Z\| at the highest frequency. |
| | `L=6.46e-07 H` | The inductance that was removed before the analysis. This is wiring, not the cell. |
| **λ** | `λ=1.301 (auto)` | Smoothing strength of the regularised inversion, chosen at the L-curve corner of a 50-point scan. |
| | `λ=0.5 (manual)` | The value you set in the panel was used instead. Larger smooths the DRT; smaller splits peaks. |
| **ext** | `ext=LINEAR` | The spectrum was continued beyond the measured window by extending the tail in log-frequency, at both ends. |
| | `ext=RC` | Continued by fitting a physical RC series, at both ends. |
| | `ext=RC-LF/LINEAR-HF` | The two ends were continued differently. The token names the low-frequency choice first; the pattern is `<lf>-LF/<hf>-HF`. |
| **Warburg** | `Warburg: none` | No diffusion element was fitted, because no diffusion tail was detected. |
| | `Warburg: post-hoc` | A Warburg-open element was fitted to the excluded low-frequency region **after** the relaxation analysis. |
| | `Warburg: joint` | The diffusion branch was fitted **together with** the relaxation basis inside the inversion, so the calculated DRT curve reproduces the low-frequency rise. |
| | `Warburg: composite` | The excluded region needed a Warburg element **plus** a slow near-RC relaxation. Both are reported inside the Warburg record rather than as an extra peak. |
| **Rb** | `Rb: 100%` | R-budget (§5.2): the reported elements carry this share of the DRT's resistance inside the fitting window. Inside 85–115 %, so the circuit accounts for the DRT it came from. |
| | `Rb: ⚠ 117%` | Outside 85–115 %: the circuit and the DRT disagree about how much impedance the window holds — typically an element reaching past the measured window, or a shoulder the model does not carry. |
| | `Rb: —` | Not available, because no peaks were reported for this spectrum. |
| **KK** | `KK: PASS (0.02%)` | Kramers-Kronig gate (§5.1) passed; the figure is the residual RMS. The limits are 0.5 % RMS and 1.5 % peak, both as a fraction of \|Z\|. |
| | `KK: ⚠ WARNING` | One or both limits were exceeded. See below for what the accompanying reason can say. |
| | `KK: off` | The gate did not run. |

**Two of these carry a variable explanation.**

A **KK warning** names what failed, as one clause or both joined by `and`:

```
residual RMS 0.72% exceeds 0.5% of |Z|
max point residual 5.07% exceeds 1.5% of |Z|
```

A **KK: off** names why the gate did not run: either `kk_gate_mode = Off`
because you turned it off, or a reason the gate could not proceed — most often
`too few points (N=6 < 8)`, below which the fit is under-determined.

Percentages in these messages are of two kinds, and it is worth telling them
apart: **0.5 %, 1.5 % and 85–115 % are fixed limits** built into the program,
while the residuals, the ratio and the resistances are what your run produced.

The peak count has a companion reading on the Nyquist panel, where each
element is drawn as its own arc — see §4.3.

### 4.3 Reading the element arcs on the Nyquist panel

Alongside the two whole-spectrum curves, the Nyquist panel draws **one arc per
extracted element**, in the colour that element's peak has in the DRT panel.
The colour is the only link between the two panels, so an arc and a shaded
peak of the same colour are the same element.

Each arc is placed at the running total of resistance: the first starts at
`R∞`, the next where it ends, and so on, so the chain spans `R∞` to the
circuit's total resistance. That makes two things readable directly:

*   the **width** of an arc is that element's `R`
*   how **flattened** it is, is that element's `φ` — a perfect semicircle is
    `φ = 1`, an ideal RC

The legend adds `— M% overlap` to each. That figure answers one question:
**may I read this arc on its own?** Near `0 %`, yes. At tens of percent, read
its width and shape but not its position, because neighbouring elements are
what put it where it is.

Across the reference spectra the figure tracks how far apart the time
constants are, which is what it is measuring:

| Spectrum | Time constants apart | Overlap reported |
|---|---|---|
| `PEMFC` | 2.87 decades | 0 %, 0 % |
| built-in example | 1.31 decades | 6 %, 13 % |
| `SOFC` | 1.24 decades | 20 %, 21 %, 5 % |
| `Ciucci/LIB_data_2` | 0.77 decades | 49 %, 25 %, 3 % |

The built-in example sits deliberately in the middle. Its two time constants
are 1.33 decades apart by construction: wider and it would stop showing what
the ZARC basis is for, narrower and it would fall under the one-decade
resolution limit and demonstrate a documented failure instead of ordinary
operation.

The last element of a chain always reports near zero. Nothing slower follows
it, and everything faster has already contributed its full resistance — so
its placement has nothing left to be wrong about.

#### Why the arcs do not add up to the total curve

They are not meant to. **The arcs decompose the circuit; the total curve is
what the circuit predicts.** Those differ, for two reasons.

**The total is a sum at each frequency; the chain is not.** Each arc sits at a
fixed resistance offset, which assumes the faster elements have already given
all their resistance and the slower ones none of theirs. That holds only when
the time constants are far apart — and the overlap percentage is exactly how
far from holding it is.

**Each arc peaks at its own frequency, so heights do not add.** On
`LIB_data_2` the three arcs reach `0.00136`, `0.00193` and `0.00528` — summing
to `0.00858`, while the actual curve never exceeds `0.00567`. No single
frequency has all three at their maximum, so adding the apexes overstates the
curve by half.

#### Why an arc can touch the axis where the total does not

An arc meets the real axis at its own two limits, `ω → ∞` on the left and
`ω → 0` on the right. Its imaginary part is zero there — **its own**, not the
others'. At that same frequency the remaining elements still have imaginary
parts, so the total is elsewhere.

The junction between two arcs makes this concrete. On the built-in example,
arc 1 ends and arc 2 begins at the same point, `x = 24,490 Ω`:

| At that junction | The total at the same frequency |
|---|---|
| arc 1's right end is `ω → 0` | `x = 35,006 Ω`, `−Z_im = 3.3` |
| arc 2's left end is `ω → ∞` | `x = 5,171 Ω`, `−Z_im = 2.0` |

**The junction is not one frequency at all** — it is the low-frequency limit of
one element and the high-frequency limit of the next, opposite ends of the
sweep. The chain is joined in space, not in frequency, and its x-axis is
accumulated resistance rather than a frequency coordinate.

The two outer ends are the exception, and the only places the arcs and the
total agree by construction: at `R∞` every element is at `ω → ∞`, and at
`R∞ + ΣR` every element is at `ω → 0`. Those are the only frequencies where
all of them are at a limit together.

---

## 5. Reading the diagnostics

These are the messages you will actually meet, in the prose the program writes
to the log. For the compressed form of the same information — the one-line run
summary and every token it can print — see §4.2.

None of them stops a run — the
program finishes and tells you what to distrust.

### 5.1 The Kramers-Kronig gate

```
[KK Gate] WARNING: residual RMS 0.72% exceeds 0.5% of |Z| and max point
residual 5.07% exceeds 1.5% of |Z|. The measurement may violate
Kramers-Kronig (drift, nonlinearity, or glitches); downstream results may
be unreliable.
```

This is about the **measurement**, not the analysis. It says the spectrum is
not consistent with a causal, linear, time-invariant system — the cell drifted
during acquisition, the amplitude was outside the linear regime, or there are
glitches. A DRT computed from such a spectrum is arithmetic performed on data
that does not satisfy the theory's premises. Re-measure if you can.

### 5.2 The R-budget check

```
R-budget ratio 4.13 is outside [0.85, 1.15]: the reported circuit and the
windowed NNLS mass disagree - typically a bound-limited element absorbing
impedance beyond the measured window, or an unmodelled shoulder.
```

The reported equivalent circuit should carry about as much resistance as the
DRT holds inside the measured window. A ratio far from 1 means the circuit is
describing something the window does not contain. Treat the element values as
indicative, not measured.

### 5.3 Bound-limited elements

```
Warning: refined element 2 is bound-limited (tau at the +1.0-decade bound
from its seed): the reported value sits on the fit bound, not a
data-determined optimum.
```

The fit ran into its own constraint. The number is where the optimiser was
stopped, not where the data put it. Usually it accompanies an R-budget
warning, and for the same reason.

A related one:

```
Warning: the slow element's phi saturates the Debye ceiling (1.000): phi is
weakly identified there, so read it as 'not distinguishable from ideal RC',
not as a measured shape.
```

### 5.4 The Warburg re-run

```
[Warburg Auto] Diffusion tail-cut fired in the default pass; re-running with
the joint Warburg polish at the adopted lambda 2.888
```

Under the default Warburg mode a spectrum with a detected diffusion tail is
analysed **twice**: once normally, then again with the diffusion element solved
jointly, at the λ the first pass settled on. The second pass is the result.

This matters when reading the log: peak counts and parameters printed before
this line belong to the discarded first pass.

---

## 6. Exporting

**💾 Export Results to Excel** writes a workbook with three sheets:

| Sheet | Contents |
|---|---|
| `Equivalent Circuit Parameters` | `R_inf`, the circuit description code, and the inductance when one was adopted, all in row 1; the Warburg and slow-element scalars in row 2; then one row per peak: `R`, `Phi`, `Phi_deconv`, `CPE_Q`, `Tau`, `C_eff` — and, two rows under the table, the circuit drawn as a schematic |
| `DRT_Spectrum` | `f_Hz`, `omega_rad_s`, `Tau_s`, `Log10_Tau`, `DRT_G_tau`, one `Peak_N_G_tau` column per deconvolved peak, and `Peak_fit_sum_G_tau` — their sum, which is what you compare against `DRT_G_tau` to judge the fit |
| `Simulated_EIS` | `f_Hz`, `omega_rad_s`, measured `Z_re`/`Z_im`, the NNLS reconstruction, and the equivalent-circuit curve |

The schematic under the parameter table carries **element names only**. An
`L` box appears at the left when an inductance was adopted, matching the
leading `L` of the code; where none was, neither shows it. `R1` and
`CPE1` are the `R` and `CPE_Q` of the `Peak 1` row directly above it, so the
picture is an index into the table rather than a second copy of it. Where the
diffusion record is the composite of §5.4, its slow element appears as
`R_slow`/`CPE_slow` — it is part of the circuit but is not a peak, so it has no
number and no table row of its own; its values are the `Warburg slow ZARC`
cells in row 2. The name is spelled out because `Rs` is normally read as the
*series* resistance, which in this diagram is the `R∞` box beside it. A line
under the drawing says where that element came from:

> `R_slow, CPE_slow: a slow relaxation inside the low-frequency region the
> diffusion tail-cut excluded`

That is the one thing the picture cannot show. The block sits in the span the
tail-cut removed, so without the line it reads as a relaxation measured
alongside the numbered ones.

**Every box in the picture has its value on the same sheet.** `L` is in `F1`,
`R∞` in `B1`, each numbered element is a table row, `R_slow` is the
`Warburg slow ZARC` cells of row 2, and `Wo` is `Warburg R_W` / `Tau_W` /
`C_lim`. The picture indexes the sheet; nothing drawn is unaccounted for.

The diffusion box is labelled `Wo`, the element name a circuit editor uses. The
code in row 1 writes the same element as `O`, Boukamp's letter. They are one
element in two notations.

If the line **`diffusion knee not resolved`** appears under that box, the fit
could not locate the knee of the diffusion branch: `R_W` and `τ_W` are what the
search returned, but the limiting capacitance derived from them would be a ratio
of two meaningless numbers, so it is not reported. The spectrum did not
determine it; see §5.4.

The plot is saved alongside as a PNG, and the schematic as a second PNG.

---

## 7. The batch command line

For many files at once, without the GUI:

```bash
python scripts/run_batch.py --input_dir Data --output_dir Results
```

| Argument | Default | Meaning |
|---|---|---|
| `--input_dir` | `Data` | Folder to read. Not recursive — subfolders are ignored. |
| `--output_dir` | `Results` | Folder to write. Created if absent. |
| `--file_pattern` | `*` | `*` expands to `*.txt`, `*.z`, `*.csv`, `*.dat` |
| `--deconv-method` | `zarc` | `zarc`, `lasso`, or `cwt` |

Each file produces `<name>_results.xlsx` and `<name>_plot.png`. All other
settings are the validated defaults; the batch runner does not expose them.

**A failing file does not stop the batch.** The error is printed with its
traceback, the run continues, and a summary at the end names every failure:

```
Batch analysis complete: 11 succeeded, 0 failed (of 11).
```

Read that line. A batch that "finished" is not a batch that succeeded, and the
per-file warnings of §5 still apply to every file that did.

---

## 8. Troubleshooting

**The window does not appear, `ModuleNotFoundError: No module named 'PyQt6'`** —
install the GUI extra: `python -m pip install -e ".[gui]"`.

**`ModuleNotFoundError: No module named 'pyfdrt'`** — the package is not
installed in the interpreter you are running. Install it, or set
`PYTHONPATH` to the `src` directory.

**`ModuleNotFoundError: No module named 'drt_pipeline'`** — run the scripts by
path from the repository root, as shown above, so their own directory is on the
import path.

**A `ValueError` naming a configuration key and listing accepted values** — a
label was supplied that the schema does not accept. This is deliberate: the
alternative is silently using a default and returning a plausible wrong answer.
Use one of the spellings it lists.

**The results changed slightly after upgrading numpy or scipy** — expected, and
see the note in §1.

---

## 9. Citing pyFDRT

The accompanying paper is in press at the *Bulletin of the Korean Chemical
Society*:

> Yoo, C.-Y., Park, J., & Kim, J. pyFDRT: an open-source Python tool for
> Fourier-transform analysis of the distribution of relaxation times in
> electrochemical impedance spectra. *Bulletin of the Korean Chemical Society*
> (2026), in press. doi:10.1002/bkcs.70232

```bibtex
@article{pyfdrt,
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

The volume and pages follow once the issue is assigned.

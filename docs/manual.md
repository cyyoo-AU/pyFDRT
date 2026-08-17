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

---

## 5. Reading the diagnostics

These are the messages you will actually meet. None of them stops a run — the
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
| `Equivalent Circuit Parameters` | `R_inf`, then one row per peak: `R`, `Phi`, `Phi_deconv`, `CPE_Q`, `Tau`, `C_eff` |
| `DRT_Spectrum` | `f_Hz`, `omega_rad_s`, `Tau_s`, `Log10_Tau`, `DRT_G_tau`, one `Peak_N_G_tau` column per deconvolved peak, and `Peak_fit_sum_G_tau` — their sum, which is what you compare against `DRT_G_tau` to judge the fit |
| `Simulated_EIS` | `f_Hz`, `omega_rad_s`, measured `Z_re`/`Z_im`, the NNLS reconstruction, and the equivalent-circuit curve |

The plot is saved alongside as a PNG.

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

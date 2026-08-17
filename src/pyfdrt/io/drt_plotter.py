"""Figure export for DRT analysis results.

``save_drt_plots`` renders the three-panel Nyquist / Bode / DRT-spectrum figure
that the batch CLI writes next to each Excel workbook. It lives in the package
(task #34) so the installed distribution ships the plot export and every entry
point can share one rendering implementation, instead of the function sitting
inside ``scripts/run_batch.py`` where the GUI could not reuse it.
"""

import numpy as np
import matplotlib.pyplot as plt

# Colour of the ECM curve's beyond-tail-cut segment, the part carried by
# the Warburg element alone (drt-io.md 6, task #113). Deliberately not
# the circuit curve's blue and not the NNLS red.
WARBURG_SEGMENT_COLOR = '#008B8B'

# Nyquist aspect (tasks #102, #119). Equal axes are the point: an arc's shape
# and its depression are read geometrically, so one ohm must span the same
# number of pixels on both axes.
#
# The box is pinned square rather than left to the layout, because with a free
# box `axis('equal')` is decided by the window instead of the data. Matplotlib
# satisfies it by expanding the data limits to fill whatever box tight_layout
# produced, and the resulting fill fraction is exactly
#
#     fill = min(r, A) / max(r, A)      r = data W/H,  A = box W/H
#
# so a box far from the data's own ratio strands the curve as a sliver. The
# corpus spans r = 0.30-5.45: at A ~ 0.22 the fill runs 0.04-0.14 and NOT ONE
# spectrum keeps equal axes, while at A = 1 it runs 0.18-0.63 and all but one
# do. Task #102 calibrated a threshold against the window geometry of the day
# (measuring 0.18-0.52) and tasks #104/#106/#107/#109-#112 then moved that
# geometry out from under it, with no test able to notice. Pinning A removes
# the window from the relation, which is why this is a pin and not a new
# calibration.
NYQUIST_BOX_ASPECT = 1.0

# Retained as a guard against layouts a square box cannot rescue, no longer as
# the ordinary path: with the box pinned the corpus floor is 0.18
# (550C_BZYPtFC, r = 5.45), so this sits below the band rather than inside it.
# Below it the panel is auto-scaled and callers mark it not-to-scale.
NYQUIST_MIN_FILL = 0.15

# Per-peak fill colours, cycled by peak index. Shared by the interactive
# panels and the exported figure so peak 2 is the same colour in the window
# and in the file a reader is handed.
PEAK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def apply_nyquist_aspect(ax) -> bool:
    """Choose the Nyquist aspect for ``ax`` from what has been plotted on it.

    Call AFTER every series is drawn and after the figure layout is settled:
    the decision reads ``ax.dataLim`` and the equal-aspect limits resolved for
    the pinned box. Returns True when the axes were left equal (arc geometry
    faithful), False when they were auto-scaled to use the panel.
    """
    span_x = float(ax.dataLim.width)
    span_y = float(ax.dataLim.height)
    if not np.isfinite(span_x) or not np.isfinite(span_y) or span_x <= 0 or span_y <= 0:
        # Degenerate or empty axes: nothing to distort, and 'equal' on a
        # zero-width bbox makes matplotlib pick arbitrary limits. The box is
        # still pinned so an empty panel opens square.
        ax.set_box_aspect(NYQUIST_BOX_ASPECT)
        ax.set_aspect('auto')
        return True

    # Pin the box first, then scale the data into it. 'datalim' is required:
    # with the default adjustable='box' matplotlib would satisfy equal aspect
    # by reshaping the box we just pinned, and the pin would do nothing.
    ax.set_box_aspect(NYQUIST_BOX_ASPECT)
    ax.set_aspect('equal', adjustable='datalim')
    # Resolve what equal aspect actually does in that box, rather than
    # predicting it: apply_aspect writes back the limits matplotlib would use.
    ax.apply_aspect()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    range_x = abs(x1 - x0)
    range_y = abs(y1 - y0)
    fill = min(span_x / range_x if range_x > 0 else 0.0,
               span_y / range_y if range_y > 0 else 0.0)
    if fill >= NYQUIST_MIN_FILL:
        return True

    # Release only the data scaling, not the box: the panel keeps its square
    # outline so the two axis lines stay the same physical length, and the
    # caller's '(axes not to scale)' title is what says the units no longer
    # match. Unpinning here would also let the box drift back with the window.
    ax.set_aspect('auto')
    ax.relim()
    ax.autoscale_view()
    return False


def draw_analysis_panels(fig, ax_nyquist, ax_drt, optimizer, results,
                         raw_data) -> bool:
    """Draw the interactive Nyquist and DRT panel pair onto caller-supplied axes.

    This is the figure a front-end shows while the user works, as distinct from
    the three-panel export figure of :func:`save_drt_plots`. It clears both
    axes, draws every series, lays the figure out and settles the Nyquist
    aspect. Returns what :func:`apply_nyquist_aspect` returned, which the
    titles already reflect; callers need it only if they mirror the state
    elsewhere.

    ``fig`` is required rather than derived from the axes because the layout
    is part of the contract, not an optional extra: ``apply_nyquist_aspect``
    reads the geometry that ``tight_layout`` resolved, so laying out after the
    aspect decision — or not at all — produces a different figure while
    type-checking, running and looking plausible. Owning both steps here is
    what stops a caller recomposing them in the wrong order (the interface
    design standards, rule 4: domain ordering stays behind its abstraction).

    Every front-end must call this rather than reimplement it. Two copies of
    this drawing let two front-ends describe the same run differently, which
    is the failure task #120 removed.
    """
    if optimizer is None or results is None:
        raise ValueError(
            "draw_analysis_panels requires a completed run: pass the optimizer "
            "and results returned by DRTAnalyzer.run().")
    if raw_data is None:
        raise ValueError(
            "draw_analysis_panels requires the raw data the run was loaded "
            "from; it is drawn whenever the inductance correction moved the "
            "spectrum.")

    meas_data = optimizer.target_data
    # Two model curves labelled by role (tasks #50/#51, drt-io.md 6):
    #   NNLS = calculated DRT reconstruction (DRT self-consistency)
    #   ECM  = extracted physical-circuit fit, gated on the REPORTED Warburg
    #          record (task #88). Without one, the ZARC elements only, drawn
    #          over the region they were fitted - up to the tail-cut. With one,
    #          the full-window ZARC + Warburg curve, the beyond-cut segment in
    #          its own hue (task #113).
    has_warburg = bool(results and results.get('Warburg'))
    rec = optimizer.reconstruct_impedance(freq=meas_data[:, 0],
                                          include_warburg=has_warburg)
    sim_re, sim_im = rec['nnls_re'], rec['nnls_im']
    ecm_re, ecm_im = rec['ecm_re'], rec['ecm_im']

    cut_tau = getattr(optimizer, 'deconv_tail_cut_tau', None)
    arc = (meas_data[:, 0] >= 1.0 / (2 * np.pi * cut_tau)
           if cut_tau else np.ones(meas_data.shape[0], dtype=bool))
    ecm_show = np.ones(meas_data.shape[0], dtype=bool) if has_warburg else arc
    ecm_label = 'ECM fit (ZARC + Warburg)' if has_warburg else 'ECM fit (ZARC)'
    if ecm_re is not None:
        z_max = float(np.hypot(meas_data[:, 1], meas_data[:, 2]).max())
        ecm_rms = 100.0 * np.sqrt(np.mean(
            (ecm_re[ecm_show] - meas_data[ecm_show, 1]) ** 2
            + (ecm_im[ecm_show] - meas_data[ecm_show, 2]) ** 2)) / z_max
        ecm_label = f"{ecm_label} — {ecm_rms:.2g}% misfit"

    ax_nyquist.clear()
    ax_drt.clear()

    # If the L correction changed the data, show both raw and corrected so the
    # inductive region that was removed stays visible. atol is scaled to the
    # data: near-zero Z_im points must not read as a correction when none was
    # effectively applied, and rtol alone is too strict around zero.
    im_scale = float(np.max(np.abs(meas_data[:, 2]))) + 1e-30
    # filter_positive_imag drops Z_im > 0 points after the correction, so the
    # two grids need not share a length and an elementwise comparison would
    # raise. A differing length already means the data was modified.
    if raw_data.shape[0] != meas_data.shape[0]:
        l_corrected = True
    else:
        l_corrected = not np.allclose(raw_data[:, 2], meas_data[:, 2],
                                      rtol=1e-4, atol=1e-6 * im_scale)
    if l_corrected:
        ax_nyquist.plot(raw_data[:, 1], -raw_data[:, 2], '+',
                        color='#bbb', markersize=7, label='Raw Data')
        ax_nyquist.plot(meas_data[:, 1], -meas_data[:, 2], '+',
                        color='gray', markersize=8,
                        label='Measured (L-corrected)')
    else:
        ax_nyquist.plot(meas_data[:, 1], -meas_data[:, 2], '+',
                        color='gray', markersize=8, label='Measured Data')
    ax_nyquist.plot(sim_re, -sim_im, 'r-', linewidth=2.5,
                    label='Calculated DRT (NNLS)')
    if ecm_re is not None:
        ax_nyquist.plot(ecm_re[arc], -ecm_im[arc], 'b--', linewidth=2,
                        label=ecm_label)
        if has_warburg and not arc.all():
            ax_nyquist.plot(ecm_re[~arc], -ecm_im[~arc],
                            color=WARBURG_SEGMENT_COLOR, linestyle='-.',
                            linewidth=2.8,
                            label='Warburg region (beyond tail-cut)')

    # Points excluded by the diffusion tail-cut (auto or manual). Open rings,
    # not a competing marker series: the excluded points are a SUBSET of the
    # measured ones, so ringing them reads as "these specific ones", keeps a
    # second red out of a panel that already has the red NNLS curve, and
    # survives greyscale print.
    if cut_tau:
        f_cut = 1.0 / (2 * np.pi * cut_tau)
        excluded = meas_data[:, 0] < f_cut
        if excluded.any():
            ax_nyquist.plot(meas_data[excluded, 1], -meas_data[excluded, 2],
                            'o', markerfacecolor='none',
                            markeredgecolor='#333333', markersize=12,
                            markeredgewidth=1.6,
                            label=f'Excluded (f < {f_cut:.2g} Hz)')
    ax_nyquist.set_xlabel(r"$Z_{re}\ (\Omega)$")
    ax_nyquist.set_ylabel(r"$-Z_{im}\ (\Omega)$")
    ax_nyquist.legend(loc='best', framealpha=0.75)
    ax_nyquist.grid(True, linestyle='--', alpha=0.6)
    ax_nyquist.set_title("Nyquist Plot", fontweight='bold')

    # DRT over the measured tau range tau = 1/(2*pi*f), matching the
    # frequency-domain panels (task #52). The tail-cut is marked on the
    # Nyquist panel only (task #53): hiding the cut region here would remove a
    # diffusion feature the DRT placed near the low-frequency edge. Tau beyond
    # the measured window is extrapolation and is excluded.
    meas_f = meas_data[:, 0]
    tau_lo = 1.0 / (2.0 * np.pi * meas_f.max())
    tau_hi = 1.0 / (2.0 * np.pi * meas_f.min())
    drt_mask = ((optimizer.log_tau >= np.log10(tau_lo))
                & (optimizer.log_tau <= np.log10(tau_hi)))
    x_drt = optimizer.log_tau[drt_mask]

    ax_drt.plot(x_drt, optimizer.polished_drt[drt_mask], 'k-', linewidth=2,
                label='DRT results')
    n_legend = 1
    refined = getattr(optimizer, 'refined_ecm_params', None)
    if refined and 'Peaks' in refined:
        for idx, p in enumerate(refined['Peaks']):
            color = PEAK_COLORS[idx % len(PEAK_COLORS)]
            R_val, tau_val = p['R'], p['Tau']
            phi_val = p.get('Phi', 1.0)
            c_eff = p.get('C_eff', tau_val / R_val if R_val > 0 else 0.0)
            peak_curve = optimizer.peak_curve(optimizer.log_tau, p)[drt_mask]
            # Wrapped onto two lines so the legend stays inside the axes.
            ax_drt.fill_between(
                x_drt, 0, peak_curve, color=color, alpha=0.4,
                label=(f"Peak {p['Peak']}: R={R_val:.4g} " + r"$\Omega$, "
                       + r"$\phi$" + f"={phi_val:.3f}," + "\n"
                       + f"C_eff={c_eff:.2e} F, " + r"$\tau$"
                       + f"={tau_val:.2e} s"))
            ax_drt.plot(x_drt, peak_curve, color=color, linestyle='--')
            n_legend += 1
    ax_drt.set_xlabel(r"$\log_{10}(\tau)\ \mathrm{(s)}$")
    ax_drt.set_ylabel(r"$G(\tau)$")
    ax_drt.set_title("DRT Plot", fontweight='bold')
    # Headroom so the legend sits above the peaks rather than over them.
    ymin, ymax = ax_drt.get_ylim()
    if ymax > ymin:
        ax_drt.set_ylim(ymin, ymax * (1.0 + 0.16 * n_legend))
    ax_drt.legend(loc='best', framealpha=0.75)
    ax_drt.grid(True, linestyle='--', alpha=0.6)

    # Layout first, then the aspect decision - see the docstring. On this
    # figure the DRT legend can push the Nyquist box narrow, which is exactly
    # the case the aspect check must see resolved rather than predicted.
    fig.tight_layout()
    faithful = apply_nyquist_aspect(ax_nyquist)
    if not faithful:
        ax_nyquist.set_title("Nyquist Plot (axes not to scale)",
                             fontweight='bold')
    return faithful


def draw_raw_data_panels(fig, ax_nyquist, ax_drt, raw_data) -> bool:
    """Draw the loaded spectrum alone, before any analysis has run.

    Same layout-then-aspect contract as :func:`draw_analysis_panels`.
    """
    if raw_data is None:
        raise ValueError("draw_raw_data_panels requires loaded data.")
    ax_nyquist.clear()
    ax_drt.clear()
    ax_nyquist.plot(raw_data[:, 1], -raw_data[:, 2], 'k+-', label='Raw Data')
    ax_nyquist.set_xlabel(r"$Z_{re}\ (\Omega)$")
    ax_nyquist.set_ylabel(r"$-Z_{im}\ (\Omega)$")
    ax_nyquist.grid(True, linestyle='--', alpha=0.6)
    ax_nyquist.set_title("Nyquist Plot")
    fig.tight_layout()
    faithful = apply_nyquist_aspect(ax_nyquist)
    if not faithful:
        ax_nyquist.set_title("Nyquist Plot (axes not to scale)")
    return faithful


def save_drt_plots(file_path: str, optimizer, results, original_raw_data=None):
    """
    Save the DRT analysis results as a three-panel Nyquist / Bode / DRT-spectrum figure.
    """
    fig = plt.figure(figsize=(18, 5))

    freq = optimizer.freq
    Z_re = optimizer.Z_re
    Z_im = optimizer.Z_im
    log_tau = optimizer.log_tau
    polished_drt = optimizer.polished_drt

    # reconstruct_impedance rebuilds the impedance without shape mismatch
    rec = optimizer.reconstruct_impedance(freq=freq)
    sim_re = rec['nnls_re']
    sim_im = rec['nnls_im']

    # Restrict the frequency-domain panels to the measured window. The
    # 'Preprocessed EIS' and 'Calculated DRT (NNLS)' series live on the extended grid
    # (ext_decades padding added for the FFT-DRT). In that padding there is no
    # measurement, and the band-limited DRT reconstruction rolls to zero while
    # the extrapolated data does not, which reads as a fit mismatch at the plot
    # edges. Showing only the measured span [f_min, f_max] keeps the display to
    # real data; the physics and all reported quantities are unaffected. The
    # ECM/ZARC curve is already evaluated on the measured grid (target_data).
    meas_f = optimizer.target_data[:, 0]
    show = (freq >= meas_f.min()) & (freq <= meas_f.max())

    colors = PEAK_COLORS

    # 1. Nyquist Plot
    ax1 = fig.add_subplot(131)
    if original_raw_data is not None:
        ax1.plot(original_raw_data[:, 1], -original_raw_data[:, 2], '+', color='lightgray', markersize=8, label='Raw Data')

    ax1.plot(Z_re[show], -Z_im[show], 'k.-', label='Preprocessed EIS', alpha=0.8)
    ax1.plot(sim_re[show], -sim_im[show], 'r-', linewidth=2.5, label='Calculated DRT (NNLS)')

    # Two model curves, labelled by role (task #50, drt-io.md §6):
    #   NNLS = calculated DRT reconstruction (DRT self-consistency)
    #   ECM  = extracted physical-circuit fit (validation of the reported R/Tau/
    #          Phi). The ECM label carries its reconstruction misfit so a poor
    #          physical fit is legible rather than mistaken for a rendering fault.
    # Gated on the REPORTED Warburg record (task #88): without one, the curve is
    # the ZARC elements only, drawn over the region they were fitted - up to the
    # tail-cut (task #51). With one, the diffusion region is explicitly modelled,
    # so the full-window ZARC + Warburg curve is drawn, the beyond-cut segment
    # dotted to mark it as carried by the Warburg element alone.
    _has_wb = bool(results and results.get('Warburg'))
    rec = optimizer.reconstruct_impedance(freq=optimizer.target_data[:, 0],
                                          include_warburg=_has_wb)
    ecm_f = optimizer.target_data[:, 0]
    _cut = getattr(optimizer, 'deconv_tail_cut_tau', None)
    _arc = ecm_f >= 1.0 / (2.0 * np.pi * _cut) if _cut else np.ones(len(ecm_f), dtype=bool)
    ecm_show = np.ones(len(ecm_f), dtype=bool) if _has_wb else _arc
    ecm_label = ('ECM fit (ZARC + Warburg)' if _has_wb
                 else 'ECM fit (ZARC)')
    if rec['ecm_re'] is not None:
        _zmax = np.hypot(optimizer.target_data[:, 1], optimizer.target_data[:, 2]).max()
        _ecm_rms = 100.0 * np.sqrt(np.mean((rec['ecm_re'][ecm_show] - optimizer.target_data[ecm_show, 1]) ** 2
                                           + (rec['ecm_im'][ecm_show] - optimizer.target_data[ecm_show, 2]) ** 2)) / _zmax
        ecm_label = f"{ecm_label} — {_ecm_rms:.2g}% misfit"
        ax1.plot(rec['ecm_re'][_arc], -rec['ecm_im'][_arc], 'b--', linewidth=2, label=ecm_label)
        if _has_wb and not _arc.all():
            # Its own hue, not the circuit curve's blue: the panel otherwise
            # carries two blues separated only by dash pattern (drt-io.md 6,
            # task #113). Teal is neither the NNLS red nor the circuit blue,
            # so it survives red-green colour blindness, and the dash-dot
            # pattern carries the distinction in greyscale.
            ax1.plot(rec['ecm_re'][~_arc], -rec['ecm_im'][~_arc],
                     color=WARBURG_SEGMENT_COLOR, linestyle='-.', linewidth=2.8,
                     label='Warburg region (beyond tail-cut)')

    ax1.set_xlabel(r"$Z_{re} (\Omega)$", fontsize=12)
    ax1.set_ylabel(r"$-Z_{im} (\Omega)$", fontsize=12)
    ax1.legend(loc='best', framealpha=0.75)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_title("1. Nyquist Plot Validation", fontsize=14, fontweight='bold')

    # 2. Bode Plot
    ax2 = fig.add_subplot(132)
    ax2.semilogx(freq[show], -Z_im[show], 'k.-', label='Preprocessed EIS', alpha=0.8)
    ax2.semilogx(freq[show], -sim_im[show], 'r-', linewidth=2.5, label='Calculated DRT (NNLS)')
    if rec['ecm_im'] is not None:
        ax2.semilogx(rec['freq'][_arc], -rec['ecm_im'][_arc], 'b--', linewidth=2, label=ecm_label)
        if _has_wb and not _arc.all():
            ax2.semilogx(rec['freq'][~_arc], -rec['ecm_im'][~_arc],
                         color=WARBURG_SEGMENT_COLOR, linestyle='-.', linewidth=2.8)

    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.set_ylabel(r"$-Z_{im} (\Omega)$", fontsize=12)
    ax2.set_title("2. Bode Plot (Imaginary)", fontsize=14, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.75)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. DRT Spectrum & ECM Peaks. The DRT is drawn over the measured tau range
    # tau = 1/(2*pi*f) for f in [f_min, f_max], matching the frequency-domain
    # panels (task #52, drt-io.md §6), so any diffusion feature the DRT places near
    # the low-frequency edge stays visible instead of being silently removed. The
    # tail-cut is no longer marked on this panel (task #53). Extrapolation-only
    # tau beyond the measured window is excluded.
    ax3 = fig.add_subplot(133)
    _meas_f = optimizer.target_data[:, 0]
    _tau_lo = 1.0 / (2.0 * np.pi * _meas_f.max())
    _tau_hi = 1.0 / (2.0 * np.pi * _meas_f.min())
    drt_mask = (log_tau >= np.log10(_tau_lo)) & (log_tau <= np.log10(_tau_hi))
    x_drt = log_tau[drt_mask]

    ax3.plot(x_drt, polished_drt[drt_mask], 'k-', linewidth=2, label='Polished DRT')

    if results and 'Peaks' in results:
        for idx, p in enumerate(results['Peaks']):
            color = colors[idx % len(colors)]
            if p.get('Amplitude', 0.0) > 0:
                peak_curve = optimizer.peak_curve(log_tau, p)[drt_mask]
                R_val = p['R']
                ax3.fill_between(x_drt, 0, peak_curve, color=color, alpha=0.4,
                                 label=f"Peak {p['Peak']} (R={R_val:.3g} $\\Omega$)")
                ax3.plot(x_drt, peak_curve, color=color, linestyle='--')

    ax3.set_xlabel(r"$\log_{10}(\tau)$ (s)", fontsize=12)
    ax3.set_ylabel("$G(\\tau)$", fontsize=12)
    ax3.set_title("3. DRT Spectrum & ECM Peaks", fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9, framealpha=0.75)
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    # After the layout: the aspect decision depends on the axes box.
    if not apply_nyquist_aspect(ax1):
        ax1.set_title("1. Nyquist Plot Validation (axes not to scale)",
                      fontsize=14, fontweight='bold')
    plt.savefig(file_path, dpi=150)
    plt.close()

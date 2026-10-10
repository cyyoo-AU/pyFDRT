"""Figure export for DRT analysis results.

``save_drt_plots`` renders the three-panel Nyquist / Bode / DRT-spectrum figure
that the batch CLI writes next to each Excel workbook. It lives in the package
(task #34) so the installed distribution ships the plot export and every entry
point can share one rendering implementation, instead of the function sitting
inside ``scripts/run_batch.py`` where the GUI could not reuse it.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

# The whole-circuit curve (drt-io.md 6, task #135). Black rather than the blue
# it used to be: PEAK_COLORS opens on the same blue, so once the per-element
# arcs entered the panel the total and the first peak's arc would have
# separated by dash pattern alone. Black is achromatic - unchanged under every
# form of colour vision deficiency, colliding with no categorical hue - and
# the width below keeps it apart from the thin arcs in greyscale too.
ECM_TOTAL_COLOR = '#000000'
ECM_TOTAL_DASH = (0, (7, 3))
ECM_TOTAL_WIDTH = 1.8

# The reconstruction is drawn as a band, not a line (drt-io.md 6, task #136).
# Two lines of comparable weight defeat each other wherever the fit is good:
# they coincide, and the panel shows alternating red and black instead of
# either curve. Wide and semi-transparent, it becomes the reference the thin
# circuit curve is read against - agreement is the black line running inside
# it, disagreement is the line leaving it.
NNLS_BAND_WIDTH = 5.0
NNLS_BAND_ALPHA = 0.30

# Arcs take a little transparency so a crossing with the band or the data
# shows both. This does not address crowding, which is a matter of view.
ARC_ALPHA = 0.75
ARC_WIDTH = 1.4

# Drawing order by role. The measured points sit above every model curve:
# on a well-fitting spectrum the curves would otherwise hide the very points
# a reader is checking them against.
Z_ARCS, Z_BAND, Z_CURVE, Z_DATA, Z_EXCLUDED = 2, 3, 4, 5, 6

# Decades either side of an element's own 1/tau. Wide enough that each arc
# closes on the real axis, so its diameter reads as R and its flattening as
# phi; the measured window alone would often show only part of an arc.
#
# Five, not three, because the low-frequency end approaches the axis as
# (w*tau)**phi and a depressed arc therefore closes slowly: at three decades a
# phi = 0.70 element still stops ~0.9 % short of R, which reads as a diameter
# that is not the R the table reports. Five brings that under 0.05 % across
# the corpus. The approach is asymptotic, so no finite range closes it exactly.
ARC_DECADES = 5.0
ARC_POINTS = 400


def _element_overlap(params, k):
    """How far arc ``k``'s placement sits from the total curve, over its own R.

    At the element's characteristic frequency ``w = 1/tau_k``, compare where
    the chained arc is drawn against where the total impedance actually is
    (drt-io.md 6, task #137):

        assumed = R_inf + sum(R_j, j<k) + Z_k(w)
        actual  = R_inf + sum(Z_j(w), all j)

    ``R_inf`` is common to both and cancels, so it is not needed here. The
    result is normalised by ``R_k`` so it compares between elements and
    between spectra: it answers "may I read this arc on its own?", which a
    raw impedance difference does not.
    """
    R_k, tau_k, _ = params[k]
    w = 1.0 / tau_k
    assumed = sum(p[0] for p in params[:k]) + _zarc(params[k], w)
    actual = sum(_zarc(p, w) for p in params)
    return abs(assumed - actual) / R_k


def _zarc(param, w):
    """One R-CPE element's impedance. ``param`` is ``(R, tau, phi)``."""
    R, tau, phi = param
    return R / (1.0 + (1j * w * tau) ** phi)


def draw_element_arcs(ax, optimizer, results, r_inf):
    """Draw one arc per refined ZARC element, chained along the real axis.

    Each element is drawn at the cumulative offset ``r_inf + sum(R_j, j<k)``,
    so the chain runs from the ohmic resistance to the circuit's total and
    each arc's diameter is that element's ``R`` (drt-io.md 6).

    These arcs decompose the **circuit**, not the curve: the total impedance
    is the complex sum at each frequency, so the chain coincides with the
    total curve only where the time constants are well separated. Where two
    elements overlap in tau the arcs visibly depart from it, and that
    departure is the overlap being shown rather than a fault.

    Returns the number of arcs drawn.
    """
    refined = getattr(optimizer, 'refined_ecm_params', None)
    if not refined or 'Peaks' not in refined:
        return 0

    # Only elements with an arc take part, in both the drawing and the
    # overlap: a discarded degenerate element must not shift the chain or the
    # frequency sum, or every number after it would describe a different
    # circuit from the one drawn.
    usable = [(i, p) for i, p in enumerate(refined['Peaks'])
              if float(p['R']) > 0 and float(p['Tau']) > 0]
    params = [(float(p['R']), float(p['Tau']), float(p.get('Phi', 1.0)))
              for _, p in usable]

    offset = float(r_inf)
    for k, (idx, p) in enumerate(usable):
        R, tau, phi = params[k]
        centre = np.log10(1.0 / tau)
        w = 2 * np.pi * np.logspace(centre - ARC_DECADES,
                                    centre + ARC_DECADES, ARC_POINTS)
        Z = R / (1.0 + (1j * w * tau) ** phi)
        ax.plot(offset + Z.real, -Z.imag,
                color=PEAK_COLORS[idx % len(PEAK_COLORS)],
                linewidth=ARC_WIDTH, alpha=ARC_ALPHA, zorder=Z_ARCS,
                label=f"Peak {p['Peak']} — {_element_overlap(params, k):.0%} overlap")
        offset += R
    return len(usable)


# --- panel text sizes (task #147) ------------------------------------------
#
# The interactive figure fills whatever window it is given, so a fixed point
# size shrinks against the panel as the window grows: measured on the desktop
# window, a default 10 pt tick label is 5.5 % of the panel height at a 900 px
# canvas and 1.7 % at 2400 px. The text does not get smaller - the panel gets
# bigger - which is exactly why it reads as unreadable on a maximised window.
#
# The sizes below are the export figure's (`save_drt_plots`): 14 pt bold
# titles and 12 pt axis labels over three panels of an 18 in figure, i.e. one
# panel 6.0 in wide. That figure is the calibration because it was sized for
# print and is the one nobody has complained about, so the interactive panels
# inherit its proportions rather than a new set of numbers.
_PANEL_REF_WIDTH_IN = 6.0
_PANEL_TITLE_PT = 14.0
_PANEL_LABEL_PT = 12.0
_PANEL_TICK_PT = 11.0
# The legends are the one departure from that calibration (drt-io.md 6.2):
# they sit inside the panels, and the export's 10 pt was sized for a three
# panel print row, not for nine Nyquist entries in a browser panel. At the
# browser figure size 10 pt hid 24 % of LIB_data_2's plotted points; 7 pt
# hides none.
_PANEL_LEGEND_PT = 7.0
# The DRT legend carries two-line peak entries, so it stays a step smaller.
_PANEL_DRT_LEGEND_PT = 6.5
# The floor is 1.0, not the proportion: a narrow window is where legibility is
# scarcest, and scaling text down there to preserve a ratio would trade the
# complaint for a worse one. So the ratio is held upward only, and no window
# renders text smaller than the print calibration. The ceiling stops a very
# wide canvas from growing labels until they crowd the data they annotate.
_PANEL_SCALE_MIN, _PANEL_SCALE_MAX = 1.0, 2.0


def _panel_fonts(fig, n_panels: int = 2) -> dict:
    """Point sizes for one row of ``n_panels`` panels across ``fig``.

    Scaled from the width one panel actually gets, so the text-to-panel ratio
    is held as the window resizes. Not a public interface: the front-ends own
    no font decisions - they hand over the figure, and the panel drawing is
    what sizes its own text.
    """
    per_panel = float(fig.get_size_inches()[0]) / max(n_panels, 1)
    scale = min(max(per_panel / _PANEL_REF_WIDTH_IN, _PANEL_SCALE_MIN),
                _PANEL_SCALE_MAX)
    return {'title': _PANEL_TITLE_PT * scale,
            'label': _PANEL_LABEL_PT * scale,
            'tick': _PANEL_TICK_PT * scale,
            'legend': _PANEL_LEGEND_PT * scale,
            'drt_legend': _PANEL_DRT_LEGEND_PT * scale}


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
    pt = _panel_fonts(fig)
    for _ax in (ax_nyquist, ax_drt):
        _ax.tick_params(labelsize=pt['tick'])

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
    # Order is set by zorder, not by call order (drt-io.md 6): the measured
    # points must sit above every model curve.
    if ecm_re is not None:
        draw_element_arcs(ax_nyquist, optimizer, results, rec['R_inf'])
    ax_nyquist.plot(sim_re, -sim_im, 'r-', linewidth=NNLS_BAND_WIDTH,
                    alpha=NNLS_BAND_ALPHA, solid_capstyle='round',
                    zorder=Z_BAND, label='Calculated DRT (NNLS)')
    if ecm_re is not None:
        ax_nyquist.plot(ecm_re[arc], -ecm_im[arc], color=ECM_TOTAL_COLOR,
                        linestyle=ECM_TOTAL_DASH, linewidth=ECM_TOTAL_WIDTH,
                        zorder=Z_CURVE, label=ecm_label)
        if has_warburg and not arc.all():
            ax_nyquist.plot(ecm_re[~arc], -ecm_im[~arc],
                            color=WARBURG_SEGMENT_COLOR, linestyle='-.',
                            linewidth=2.6, zorder=Z_CURVE,
                            label='Warburg region (beyond tail-cut)')
    if l_corrected:
        ax_nyquist.plot(raw_data[:, 1], -raw_data[:, 2], '+',
                        color='#bbb', markersize=7, zorder=Z_DATA,
                        label='Raw Data')
        ax_nyquist.plot(meas_data[:, 1], -meas_data[:, 2], '+',
                        color='#444444', markersize=8, zorder=Z_DATA,
                        label='Measured (L-corrected)')
    else:
        ax_nyquist.plot(meas_data[:, 1], -meas_data[:, 2], '+',
                        color='#444444', markersize=8, zorder=Z_DATA,
                        label='Measured Data')

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
                            markeredgewidth=1.6, zorder=Z_EXCLUDED,
                            label=f'Excluded (f < {f_cut:.2g} Hz)')
    ax_nyquist.set_xlabel(r"$Z_{re}\ (\Omega)$", fontsize=pt['label'])
    ax_nyquist.set_ylabel(r"$-Z_{im}\ (\Omega)$", fontsize=pt['label'])
    ax_nyquist.legend(loc='best', framealpha=0.75, fontsize=pt['legend'])
    ax_nyquist.grid(True, linestyle='--', alpha=0.6)
    ax_nyquist.set_title("Nyquist Plot", fontweight='bold',
                         fontsize=pt['title'])

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
    ax_drt.set_xlabel(r"$\log_{10}(\tau)\ \mathrm{(s)}$", fontsize=pt['label'])
    ax_drt.set_ylabel(r"$G(\tau)$", fontsize=pt['label'])
    ax_drt.set_title("DRT Plot", fontweight='bold', fontsize=pt['title'])
    # Headroom so the legend sits above the peaks rather than over them.
    ymin, ymax = ax_drt.get_ylim()
    if ymax > ymin:
        ax_drt.set_ylim(ymin, ymax * (1.0 + 0.16 * n_legend))
    ax_drt.legend(loc='best', framealpha=0.75, fontsize=pt['drt_legend'])
    ax_drt.grid(True, linestyle='--', alpha=0.6)

    # Layout first, then the aspect decision - see the docstring. On this
    # figure the DRT legend can push the Nyquist box narrow, which is exactly
    # the case the aspect check must see resolved rather than predicted.
    fig.tight_layout()
    faithful = apply_nyquist_aspect(ax_nyquist)
    if not faithful:
        ax_nyquist.set_title("Nyquist Plot (axes not to scale)",
                             fontweight='bold', fontsize=pt['title'])
    return faithful


def draw_raw_data_panels(fig, ax_nyquist, ax_drt, raw_data) -> bool:
    """Draw the loaded spectrum alone, before any analysis has run.

    Same layout-then-aspect contract as :func:`draw_analysis_panels`.
    """
    if raw_data is None:
        raise ValueError("draw_raw_data_panels requires loaded data.")
    ax_nyquist.clear()
    ax_drt.clear()
    pt = _panel_fonts(fig)
    for _ax in (ax_nyquist, ax_drt):
        _ax.tick_params(labelsize=pt['tick'])
    ax_nyquist.plot(raw_data[:, 1], -raw_data[:, 2], 'k+-', label='Raw Data')
    ax_nyquist.set_xlabel(r"$Z_{re}\ (\Omega)$", fontsize=pt['label'])
    ax_nyquist.set_ylabel(r"$-Z_{im}\ (\Omega)$", fontsize=pt['label'])
    ax_nyquist.grid(True, linestyle='--', alpha=0.6)
    ax_nyquist.set_title("Nyquist Plot", fontsize=pt['title'])
    fig.tight_layout()
    faithful = apply_nyquist_aspect(ax_nyquist)
    if not faithful:
        ax_nyquist.set_title("Nyquist Plot (axes not to scale)",
                             fontsize=pt['title'])
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
        ax1.plot(original_raw_data[:, 1], -original_raw_data[:, 2], '+', color='lightgray', markersize=8, zorder=Z_DATA, label='Raw Data')

    ax1.plot(Z_re[show], -Z_im[show], 'k.-', label='Preprocessed EIS', alpha=0.8,
             zorder=Z_DATA)
    ax1.plot(sim_re[show], -sim_im[show], 'r-', linewidth=NNLS_BAND_WIDTH,
             alpha=NNLS_BAND_ALPHA, solid_capstyle='round', zorder=Z_BAND,
             label='Calculated DRT (NNLS)')

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
        draw_element_arcs(ax1, optimizer, results, rec['R_inf'])
        ax1.plot(rec['ecm_re'][_arc], -rec['ecm_im'][_arc],
                 color=ECM_TOTAL_COLOR, linestyle=ECM_TOTAL_DASH,
                 linewidth=ECM_TOTAL_WIDTH, zorder=Z_CURVE, label=ecm_label)
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
        ax2.semilogx(rec['freq'][_arc], -rec['ecm_im'][_arc],
                     color=ECM_TOTAL_COLOR, linestyle=ECM_TOTAL_DASH,
                     linewidth=ECM_TOTAL_WIDTH, label=ecm_label)
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


# --------------------------------------------------------------------------
# The extracted circuit as a schematic (drt-io.md 6.1, tasks #140, #141)
#
# Drawn in the plain style of a circuit editor: every element is a labelled
# rectangle on a single line, parallel pairs stacked between two rails. It
# carries element NAMES and no values, because it is placed directly beneath
# the parameter table that holds them - repeating them in the picture would
# create a second copy to keep in agreement with the first.
#
# The topology is fixed by the method, so nothing here chooses a circuit and
# the drawing must not suggest that anything did.

_LINE = '#000000'
_WIRE = dict(color=_LINE, lw=1.3, solid_capstyle='round', zorder=2)

# Element box geometry, in the axes' own units.
_BOX_W, _BOX_H = 1.5, 0.62
_RAIL = 0.34            # rail-to-box lead length
_GAP = 0.55             # series gap between neighbouring parts
_ARM = 0.95             # rail half-height for a parallel pair


def _labelled_box(ax, x, y, label, width=_BOX_W, height=_BOX_H):
    ax.add_patch(Rectangle((x, y - height/2), width, height, facecolor='white',
                           edgecolor=_LINE, lw=1.3, zorder=3))
    ax.text(x + width/2, y, label, ha='center', va='center', fontsize=10,
            color=_LINE, zorder=4)


def draw_circuit_diagram(ax, results, inductance: float = 0.0) -> None:
    """Draw the reported circuit onto `ax` as a labelled schematic.

    Element names only. The names are the reader's index into the parameter
    table this picture sits under: ``R1``/``CPE1`` are the ``R`` and ``CPE_Q``
    of the table's ``Peak 1`` row. Printing the values here as well would put
    the same numbers in two places with nothing keeping them equal.

    ``inductance`` is the series inductance the preprocessing adopted
    (``L_val``). When it is positive an ``L`` box is drawn to the left of
    ``R_inf``, so the picture describes the measurement rather than the
    corrected spectrum; drt-preprocessing.md 2 treats that inductance as a
    physical cabling element and rejects a non-positive one as unphysical,
    which is why it belongs in the circuit. A zero draws nothing: the gates
    rejected it, and a box would assert an element the analysis declined.

    The Warburg box is labelled ``Wo``, the element name a circuit editor uses
    for the reflective finite-length form. The circuit description code of
    drt-io.md 5.4 writes the same element as Boukamp's ``O``; the two notations
    are element *names* and element *codes* respectively, and the manual
    records that they denote one element.
    """
    peaks = list(results.get('Peaks', []) or []) if results else []
    warburg = (results or {}).get('Warburg')

    y, x = 0.0, 0.4
    ax.plot([0.0, x], [y, y], **_WIRE)

    if inductance and inductance > 0:
        _labelled_box(ax, x, y, '$L$')
        x += _BOX_W
        ax.plot([x, x + _GAP/2], [y, y], **_WIRE)
        x += _GAP/2

    _labelled_box(ax, x, y, r'$R_\infty$')
    x += _BOX_W

    pairs = [(f'$R_{{{i+1}}}$', f'$CPE_{{{i+1}}}$', None)
             for i in range(len(peaks))]
    if warburg and 'slow_zarc' in warburg:
        # Spelled out rather than abbreviated to `Rs`. In EIS `Rs` is read as
        # the SERIES resistance, which in this very diagram is the box next to
        # it, so the short form actively misleads instead of merely being
        # terse. Not a peak number either: the slow element is co-fitted on the
        # residual and is absent from the table's numbered rows (drt-io.md 5.1).
        pairs.append((r'$R_{slow}$', r'$CPE_{slow}$', None))

    for top_label, bottom_label, caption in pairs:
        ax.plot([x, x + _GAP/2], [y, y], **_WIRE)
        left = x + _GAP/2
        top, bottom = y + _ARM, y - _ARM
        ax.plot([left, left], [bottom, top], **_WIRE)
        for rail in (top, bottom):
            ax.plot([left, left + _RAIL], [rail, rail], **_WIRE)
        _labelled_box(ax, left + _RAIL, top, top_label)
        _labelled_box(ax, left + _RAIL, bottom, bottom_label)
        right = left + _RAIL + _BOX_W
        for rail in (top, bottom):
            ax.plot([right, right + _RAIL], [rail, rail], **_WIRE)
        ax.plot([right + _RAIL, right + _RAIL], [bottom, top], **_WIRE)
        ax.plot([right + _RAIL, right + _RAIL + _GAP/2], [y, y], **_WIRE)
        if caption:
            ax.text((left + _RAIL + right) / 2, bottom - _BOX_H/2 - 0.18,
                    caption, ha='center', va='top', fontsize=8, color=_LINE)
        x = right + _RAIL + _GAP/2

    if warburg:
        ax.plot([x, x + _GAP/2], [y, y], **_WIRE)
        _labelled_box(ax, x + _GAP/2, y, r'$W_{o}$')
        centre = x + _GAP/2 + _BOX_W/2
        if warburg.get('C_lim') is None:
            # States what the spectrum failed to determine, not what the
            # program declined to print (task #140). A blank line here would
            # read as an omission rather than as a result.
            ax.text(centre, y - _BOX_H/2 - 0.18, 'diffusion knee not resolved',
                    ha='center', va='top', fontsize=8, color=_LINE)
        x += _GAP/2 + _BOX_W

    ax.plot([x, x + 0.4], [y, y], **_WIRE)

    # A note under the drawing rather than a caption beside the block. The
    # names are what a reader needs explained, and an explanation reads as a
    # legend for the figure rather than as an annotation of one element - which
    # also keeps the wiring row uncluttered as the element count grows.
    if warburg and 'slow_zarc' in warburg:
        # The second clause is not hedging. drt-deconvolution.md 2.4.2 records
        # that this element may be distributed diffusion - particle-size
        # smearing - rather than a discrete process, and that the data cannot
        # separate the two. Drawing it as an R-CPE block already asserts more
        # discreteness than the fit can support, so the legend must say so;
        # otherwise the picture is the most confident statement in the output.
        ax.text(-0.1, -1.62,
                '$R_{slow}$, $CPE_{slow}$: a slow relaxation inside the '
                'low-frequency region the diffusion tail-cut excluded.\n'
                'It may be distributed diffusion rather than a discrete '
                'process; the data cannot distinguish the two.',
                ha='left', va='top', fontsize=8.5, color=_LINE,
                linespacing=1.4)

    ax.set_xlim(-0.2, x + 0.6)
    ax.set_ylim(-2.55, 1.55)
    ax.set_aspect('equal')
    ax.axis('off')


def save_circuit_diagram(file_path: str, results, inductance: float = 0.0) -> None:
    """Write the circuit schematic to its own image file.

    Its own file rather than a fourth panel of `save_drt_plots`: the schematic
    is one wide, short row, and a quarter-width column would compress it to
    illegibility while stretching the three panels that already work.
    """
    fig, _ax = _circuit_figure(results, inductance)
    fig.savefig(file_path, facecolor='white')
    plt.close(fig)


def _circuit_figure(results, inductance: float = 0.0):
    """A figure sized to the element count, for a file or an embedded picture."""
    n = len(results.get('Peaks', []) or []) if results else 0
    warburg = (results or {}).get('Warburg')
    if warburg:
        n += 2 if 'slow_zarc' in warburg else 1
    if inductance and inductance > 0:
        n += 1
    fig = plt.figure(figsize=(1.8 + 1.55 * max(n, 1), 1.9), dpi=150)
    ax = fig.add_subplot(111)
    draw_circuit_diagram(ax, results, inductance)
    fig.tight_layout(pad=0.3)
    return fig, ax

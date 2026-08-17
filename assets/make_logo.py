"""Generate the pyFDRT logo assets.

Outputs (written next to this script, in assets/):
    pyFDRT_icon.png   square app/badge icon (transparent, 1024 px)
    pyFDRT_icon.ico   multi-size Windows icon for the PyQt6 window
    pyFDRT_logo.png   horizontal lockup (icon + wordmark, transparent)
    pyFDRT_logo.svg   vector version of the lockup
    pyFDRT_logo_dark.png  light wordmark for dark backgrounds

The mark tells the program's story: a Nyquist arc (EIS input) flowing into
the amber DRT peak spectrum (the tool's output). The curves are the real
SCNT3|GDC|SCNT3 symmetric solid oxide fuel cell measured at 600 C.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ASSETS = Path(__file__).resolve().parent

NAVY = "#0E2A47"
NAVY_LIGHT = "#46658A"
TEAL = "#2DD4BF"
AMBER = "#F59E0B"
AMBER_DARK = "#D97706"
SLATE = "#64748B"
WHITE = "#FFFFFF"

# Prefer a clean Windows UI font, fall back to matplotlib's bundled font.
_available = {f.name for f in fm.fontManager.ttflist}
for _f in ("Segoe UI", "Arial", "DejaVu Sans"):
    if _f in _available:
        FONT = _f
        break
else:
    FONT = "sans-serif"


# SCNT3|GDC|SCNT3 symmetric solid oxide fuel cell at 600 C. ZARC (R-CPE)
# parameters extracted by pyFDRT from the measured spectrum Data/SCNT3_600C.Z
# (see Results/DRT_Results_SCNT3_600C.xlsx and Section 3.2 of the manuscript).
# Four electrode processes give overlapping depressed Nyquist arcs and a
# four-peak DRT; PHI_K is the CPE exponent of each element (phi < 1 -> depressed).
R_INF = 29.68
R_K = [0.148, 2.423, 1.936, 2.482]
TAU_K = [4.07e-5, 4.23e-3, 6.97e-2, 1.91e-1]
PHI_K = [1.00, 0.571, 0.918, 1.00]


def _gauss(x, mu, sig, amp):
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def _scnt3_nyquist(freqs):
    """ZARC impedance: Z = R_inf + sum R_k / (1 + (j w tau_k)^phi_k)."""
    w = 2.0 * np.pi * np.asarray(freqs)
    z = R_INF + sum(r / (1.0 + (1j * w * t) ** p)
                    for r, t, p in zip(R_K, TAU_K, PHI_K))
    return z.real, -z.imag


def _scnt3_drt(logtau):
    """Four-peak DRT in log10(tau); peak heights scale with R_k."""
    sig = 0.16
    return sum(
        r * np.exp(-0.5 * ((logtau - np.log10(t)) / sig) ** 2)
        for r, t in zip(R_K, TAU_K)
    )


def _fit(v, lo, hi):
    """Linearly map array v onto [lo, hi] using its own min/max."""
    v = np.asarray(v, dtype=float)
    return lo + (v - v.min()) / (v.max() - v.min()) * (hi - lo)


def draw_icon(ax):
    """Draw the badge mark on an Axes using a 0..10 coordinate box."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Rounded badge.
    badge = FancyBboxPatch(
        (0.55, 0.55),
        8.9,
        8.9,
        boxstyle="round,pad=0,rounding_size=1.7",
        linewidth=0,
        facecolor=NAVY,
        zorder=1,
    )
    ax.add_patch(badge)

    x0, x1 = 1.35, 8.65

    # Nyquist: four overlapping SCNT3 R-CPE arcs (upper region) + data points.
    xr, yr = _scnt3_nyquist(np.logspace(6, -3, 600))
    ny_base, ny_amp = 4.7, 3.55
    nx = _fit(xr, x0, x1)
    ny = ny_base + yr / yr.max() * ny_amp
    ax.plot([x0 - 0.1, x1 + 0.1], [ny_base, ny_base], color=NAVY_LIGHT,
            lw=2.2, solid_capstyle="round", zorder=2)
    ax.plot(nx, ny, color=TEAL, lw=3.6, solid_capstyle="round",
            solid_joinstyle="round", zorder=3)
    # Markers spaced evenly along the curve (equal arc length, full span).
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(nx), np.diff(ny)))])
    targets = np.linspace(0.0, arc[-1], 15)
    ax.scatter(np.interp(targets, arc, nx), np.interp(targets, arc, ny),
               s=46, color=WHITE, zorder=4, edgecolors="none")

    # DRT: four peaks (lower region) sharing a baseline axis.
    lt = np.linspace(-5.2, 0.4, 500)
    g = _scnt3_drt(lt)
    dr_base, dr_amp = 1.95, 2.05
    dx = _fit(lt, x0, x1)
    dy = dr_base + g / g.max() * dr_amp
    ax.fill_between(dx, dr_base, dy, color=AMBER, edgecolor=AMBER_DARK,
                    linewidth=2.0, joinstyle="round", zorder=3)
    ax.plot([x0 - 0.1, x1 + 0.1], [dr_base, dr_base], color=NAVY_LIGHT,
            lw=2.2, solid_capstyle="round", zorder=2)


def draw_wordmark(ax, py_color=SLATE, main_color=NAVY, tagline_color=NAVY_LIGHT):
    """Two-tone 'pyFDRT' wordmark plus tagline, left-aligned at x=0."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig = ax.figure
    fig.canvas.draw()

    size = 78
    t_py = ax.text(0.0, 5.7, "py", color=py_color, fontsize=size,
                   fontfamily=FONT, fontweight="bold", va="center", ha="left")
    # Position 'FDRT' right after 'py' using the rendered extent.
    ext = t_py.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    x_end = inv.transform((ext.x1, 0))[0]
    ax.text(x_end + 0.12, 5.7, "FDRT", color=main_color, fontsize=size,
            fontfamily=FONT, fontweight="bold", va="center", ha="left")
    ax.text(0.12, 2.35, "Fourier-transform DRT analysis", color=tagline_color,
            fontsize=24, fontfamily=FONT, fontweight="normal",
            va="center", ha="left")


def make_icon(path, px=1024):
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_icon(ax)
    fig.savefig(path, transparent=True, dpi=100)
    plt.close(fig)


def make_lockup(path, dark=False, svg=False):
    fig = plt.figure(figsize=(13.0, 4.0), dpi=200)
    ax_icon = fig.add_axes([0.0, 0.0, 0.30, 1.0])
    draw_icon(ax_icon)
    ax_word = fig.add_axes([0.33, 0.0, 0.66, 1.0])
    if dark:
        draw_wordmark(ax_word, py_color="#94A3B8", main_color=WHITE,
                      tagline_color="#94A3B8")
    else:
        draw_wordmark(ax_word)
    fig.savefig(path, transparent=True, dpi=300 if not svg else None)
    plt.close(fig)


def png_to_ico(png_path, ico_path):
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow not available; skipping .ico)")
        return
    img = Image.open(png_path).convert("RGBA")
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
             (128, 128), (256, 256)]
    img.save(ico_path, format="ICO", sizes=sizes)


def main():
    ASSETS.mkdir(exist_ok=True)
    icon_png = ASSETS / "pyFDRT_icon.png"
    make_icon(icon_png)
    print(f"wrote {icon_png.name}")

    png_to_ico(icon_png, ASSETS / "pyFDRT_icon.ico")
    print("wrote pyFDRT_icon.ico")

    make_lockup(ASSETS / "pyFDRT_logo.png")
    print("wrote pyFDRT_logo.png")
    make_lockup(ASSETS / "pyFDRT_logo.svg", svg=True)
    print("wrote pyFDRT_logo.svg")
    make_lockup(ASSETS / "pyFDRT_logo_dark.png", dark=True)
    print("wrote pyFDRT_logo_dark.png")


if __name__ == "__main__":
    main()

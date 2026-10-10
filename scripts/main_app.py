import os
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timedelta

import numpy as np

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QPushButton, QLabel, QFileDialog,
                                 QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit,
                                 QGroupBox, QSlider, QScrollArea,
                                 QFormLayout, QMessageBox)
    from PyQt6.QtCore import Qt, QObject, QEvent
    from PyQt6.QtGui import QIcon, QFont, QFontMetrics
    QT_API = 'PyQt6'
except Exception:
    try:
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                     QHBoxLayout, QPushButton, QLabel, QFileDialog,
                                     QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit,
                                     QGroupBox, QSlider, QScrollArea,
                                     QFormLayout, QMessageBox)
        from PyQt5.QtCore import Qt, QObject, QEvent
        from PyQt5.QtGui import QIcon, QFont, QFontMetrics
        QT_API = 'PyQt5'
    except Exception:
        from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                       QHBoxLayout, QPushButton, QLabel, QFileDialog,
                                       QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit,
                                       QGroupBox, QSlider, QScrollArea,
                                       QFormLayout, QMessageBox)
        from PySide6.QtCore import Qt, QObject, QEvent
        from PySide6.QtGui import QIcon, QFont, QFontMetrics
        QT_API = 'PySide6'

# Matplotlib embedding in PyQt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Core pipeline & Export modules
from drt_pipeline import DRTAnalyzer
from pyfdrt.io.drt_exporter import export_results_to_excel, ExcelExportError
# Shared with the batch plotter so both renderings make the same aspect call.
from pyfdrt.io.drt_plotter import draw_analysis_panels, draw_raw_data_panels
from pyfdrt.io.circuit_cdc import circuit_cdc
from pyfdrt.io.reference_spectrum import (reference_spectrum_summary,
                                          reference_spectrum_text)
from pyfdrt.pipeline_config import (
    PipelineConfig, LambdaMode, RInfMode, ExtMethod, Criterion,
    DeconvolutionMethod,
)

# Project base directory (folder containing this script). Used to resolve the
# default data-load and result-save locations.
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'Data')
RESULTS_DIR = os.path.join(BASE_DIR, 'Results')
# Icon lives in the repo-root assets folder; fall back to a local assets dir.
ASSETS_DIR  = os.path.join(BASE_DIR, 'assets')
if not os.path.isdir(ASSETS_DIR):
    ASSETS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'assets')
ICON_PATH   = os.path.join(ASSETS_DIR, 'pyFDRT_icon.ico')


def _is_release_build() -> bool:
    """True when this is a packaged build rather than a source checkout.

    A frozen bundle (PyInstaller and friends) is the distribution case. What
    it still changes, since task #133 restored the log pane everywhere: a
    failure also raises a modal, and the window may shrink to fit its controls
    rather than only grow. PYFDRT_RELEASE overrides the detection either way,
    so the release layout can be exercised from a source checkout (=1) and a
    bundle can be run as a source build would (=0).
    """
    env = os.environ.get('PYFDRT_RELEASE')
    if env is not None:
        return env.strip().lower() not in ('', '0', 'false', 'no', 'off')
    return bool(getattr(sys, 'frozen', False))


def _log_dir() -> str:
    """Per-user writable log directory.

    Never beside the executable: an installed app commonly lives somewhere
    the user cannot write (Program Files), and a log that fails to open is a
    support case with no evidence. PYFDRT_LOG_DIR overrides for site setups.
    """
    base = os.environ.get('PYFDRT_LOG_DIR')
    if base:
        return base
    if sys.platform == 'win32':
        root = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    elif sys.platform == 'darwin':
        root = os.path.join(os.path.expanduser('~'), 'Library', 'Logs')
    else:
        root = (os.environ.get('XDG_STATE_HOME')
                or os.path.join(os.path.expanduser('~'), '.local', 'state'))
    return os.path.join(root, 'pyFDRT', 'logs')


class _FileLog:
    """Append every logged line to a per-day file.

    One file per calendar day (pyFDRT_YYYY-MM-DD.log), appended across runs
    and sessions with a header per session, so a day's work reads as one
    transcript. Each line is flushed: a log that only exists in a buffer is
    worthless after the crash it was meant to explain. Every failure path is
    swallowed - logging must never be the reason the app stops working.
    """

    RETENTION_DAYS = 30

    def __init__(self):
        self.path = None
        self._fh = None
        try:
            directory = _log_dir()
            os.makedirs(directory, exist_ok=True)
            self.path = os.path.join(
                directory, f"pyFDRT_{datetime.now():%Y-%m-%d}.log")
            self._fh = open(self.path, 'a', encoding='utf-8')
            self._fh.write(f"\n===== session start "
                           f"{datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
            self._fh.flush()
            self._prune(directory)
        except Exception:
            self._fh = None

    def write(self, message):
        if self._fh is None:
            return
        try:
            self._fh.write(f"{datetime.now():%H:%M:%S} {message}\n")
            self._fh.flush()
        except Exception:
            self._fh = None

    def _prune(self, directory):
        """Drop files older than RETENTION_DAYS so the folder cannot grow
        without bound on a machine that runs the app daily."""
        cutoff = datetime.now() - timedelta(days=self.RETENTION_DAYS)
        for name in os.listdir(directory):
            if not (name.startswith('pyFDRT_') and name.endswith('.log')):
                continue
            try:
                stamp = datetime.strptime(name[len('pyFDRT_'):-len('.log')],
                                          '%Y-%m-%d')
            except ValueError:
                continue
            if stamp < cutoff:
                try:
                    os.remove(os.path.join(directory, name))
                except OSError:
                    pass


class _StdoutToLog:
    """File-like adapter routing pipeline print() output into the GUI log.

    DRTAnalyzer speaks through the injected logger, which reaches the pyFDRT
    Log pane - but DRTOptimizer's diagnostics (the task #83 bound-limited
    warnings, the #87 R-budget line, the #99 Debye-ceiling warning, the #89
    composite-adoption line) are print()-based, and without capture they land
    on the launching console, which does not exist when the app is started by
    double-click. Scoped to the run call by redirect_stdout, so prints made
    elsewhere in the process are untouched (task #108).
    """

    def __init__(self, log_fn):
        self._log = log_fn
        self._buf = ''

    def write(self, text):
        self._buf += text
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if line.strip():
                self._log(line)
        return len(text)

    def flush(self):
        if self._buf.strip():
            self._log(self._buf)
        self._buf = ''


class _WheelGuard(QObject):
    """Stop the mouse wheel from silently changing an unfocused control.

    Qt lets the wheel cycle a QComboBox and step a QSpinBox/QSlider under the
    cursor even when it has no focus. That was harmless while the control
    panel could not scroll; once it could (task #101), scrolling the panel
    with the pointer over a control changes that control instead - measured on
    every one of them, so a run could silently use a configuration the user
    never chose. Here the wheel scrolls the panel and only a focused control
    (clicked into deliberately) responds to it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.area = None            # set once the scroll area exists

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            if self.area is not None:
                bar = self.area.verticalScrollBar()
                bar.setValue(bar.value() - event.angleDelta().y())
            return True             # never reaches the control
        return False

    def protect(self, *widgets):
        for w in widgets:
            # StrongFocus so a click still gives the control the keyboard, and
            # with it the wheel, when the user actually means to use it.
            w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            w.installEventFilter(self)


class DRTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyFDRT v1.0")
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))
        # Open at a size derived from the screen rather than a fixed pixel box:
        # a hard-coded geometry overflows a laptop panel and wastes a large
        # monitor. The control panel is scrollable (see below), so the window
        # is free to be shorter than the stacked controls; the clamps keep it
        # usable at both extremes, and the user can resize from here.
        avail = QApplication.primaryScreen().availableGeometry()
        # 58% (was 49%): tall enough that the log pane under the compacted
        # panel is visible without scrolling on a 1080p-class screen.
        win_h = max(520, min(880, int(avail.height() * 0.58)))
        win_h = min(win_h, avail.height())
        # Width follows height (see _width_for_height); the final height is
        # settled after initUI by _fit_to_export.
        self._avail = avail
        win_w = self._width_for_height(win_h, avail)
        self.resize(win_w, win_h)
        self.move(avail.left() + (avail.width() - win_w) // 2,
                  avail.top() + max(0, (avail.height() - win_h) // 4))
        # Below this the plot panel stops being readable; the controls stay
        # reachable at any size because they scroll.
        self.setMinimumSize(820, 460)

        # Logging is set up before initUI so any construction-time message
        # already has somewhere to go.
        self._release  = _is_release_build()
        self._file_log = _FileLog()
        self.log_text  = None          # the pane; built in initUI

        self.raw_data  = None
        self.optimizer = None
        self.results   = None
        self._running  = False   # re-entrancy guard for run_optimization
        self._fitted   = False   # _fit_to_export runs once, on the first show
        self.analyzer  = DRTAnalyzer(logger=self.log)
        # Manual diffusion tail-cut (tau in seconds) set by clicking a data
        # point in tail-cut mode; None -> automatic detection.
        self.manual_tail_cut_tau = None
        self._last_L_val = 0.0

        self.initUI()
        if self._file_log.path:
            self.log(f"-> [Log] {self._file_log.path}")
        # The fit runs on the first showEvent, not here: before the window is
        # shown the scroll viewport reports a placeholder height (476 px
        # against a real 978 on this layout), which made the derived chrome
        # ~500 px too large and the window grow instead of settle (#112).

    @staticmethod
    def _width_for_height(win_h, avail):
        """Initial width that renders the empty Nyquist axes square (task #107).

        Measured on this layout (offscreen, Fusion style): the axes box is
        axes_w = 0.45*win_w - 173 and axes_h = 0.884*win_h - 72, so equal
        boxes need win_w = 1.96*win_h + 223. The relation drifts slightly with
        fonts/DPI, and axis('equal') takes over as soon as data is plotted -
        this only makes the empty panels open square-ish.
        """
        return min(max(900, int(1.96 * win_h + 223)), avail.width())

    def showEvent(self, event):
        """Fit the window the first time it is actually shown.

        Only then does the scroll area report its true viewport height, which
        the fit needs to separate window chrome from panel content.
        """
        super().showEvent(event)
        if not self._fitted:
            self._fitted = True
            self._fit_to_export()

    def _fit_to_export(self):
        """Size the window against what the controls need down to Export.

        The screen-relative height of task #101 can leave the panel's lower
        half - Run, the two mode buttons, Export - below the fold, so the
        workflow's last step had to be scrolled into view (task #110).

        Which way the height may move depends on what sits below Export
        (task #112). A source build has the log pane there, and it uses
        whatever height is going, so the window may only GROW. A release
        build has nothing below Export, so any surplus is dead space and the
        window is sized to FIT - shrinking as readily as growing. Either way
        the width is re-derived so task #107's square relation survives, which
        is what also scales the plot panel with the window.

        The screen still wins: where the panel cannot fit, the window stays
        within the available height and the panel scrolls as before.
        """
        panel, scroll = self._control_panel, self._control_scroll
        panel.layout().activate()
        self.centralWidget().layout().activate()
        viewport_h = scroll.viewport().height()
        if viewport_h <= 0:                      # not laid out yet; keep as is
            return
        # Everything above the log pane must be reachable; the log itself may
        # stay below the fold, which is what the panel's scrolling is for.
        need_panel = (self.btn_export.geometry().bottom()
                      + panel.layout().contentsMargins().bottom())
        chrome = self.height() - viewport_h      # margins + scroll frame
        need = need_panel + chrome
        floor = self.height() if not self._release else self.minimumHeight()
        win_h = min(self._avail.height(), max(floor, need))
        if win_h == self.height():
            return
        win_w = self._width_for_height(win_h, self._avail)
        self.resize(win_w, win_h)
        self.move(self._avail.left() + (self._avail.width() - win_w) // 2,
                  self._avail.top() + max(0, (self._avail.height() - win_h) // 4))

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ==========================================
        # Left control panel
        # ==========================================
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        # Panel width follows the font instead of a pixel literal (task #109).
        # 320 px was calibrated against the default 9 pt font; Windows' font
        # size setting enlarges text without touching Qt's logical pixels, so
        # a hard-coded width clips the form rows on laptops running 125-150 %
        # text scale. The scale is measured as the ratio of the effective
        # font's line height to the same family at 9 pt, taken on this screen
        # - DPI cancels between numerator and denominator, so at the default
        # font the ratio is exactly 1 and the panel stays 320 on every
        # platform, while an enlarged font widens it in proportion.
        _ref_font = QFont(self.font())
        _ref_font.setPointSizeF(9.0)
        _font_scale = max(1.0, self.fontMetrics().height()
                          / max(1, QFontMetrics(_ref_font).height()))
        _panel_w = round(320 * _font_scale)
        control_panel.setFixedWidth(_panel_w)
        self._control_panel = control_panel

        def _shrinkable_combo():
            # A QComboBox claims the full width of its longest item text -
            # 'Auto (Smart Detection)' alone demands more than the panel - so
            # one form row of task #104's label+combo layout exceeded 320 px
            # even at the default font (task #109). Let the CLOSED control
            # elide instead; the dropdown list still shows every item text in
            # full, so the accepted label vocabulary of drt-pipeline.md
            # section 3 is unaffected.
            cb = QComboBox()
            cb.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(12)
            return cb

        # File load button
        # No extension list on the label: the file dialog's own filter already
        # states what is accepted, and the list was incomplete anyway (the
        # dialog also takes .csv and .dat).
        btn_load = QPushButton("📁 Load Data")
        btn_load.clicked.connect(self.load_data)
        btn_load.setStyleSheet("font-weight: bold; padding: 5px;")
        control_layout.addWidget(btn_load)

        btn_reference = QPushButton("🧪 Load Example Spectrum")
        btn_reference.clicked.connect(self.load_reference_spectrum)
        btn_reference.setStyleSheet("padding: 4px;")
        btn_reference.setToolTip(
            "A synthetic two-element spectrum built into the package, for "
            "trying the pipeline without a file of your own.\n"
            + reference_spectrum_summary())
        control_layout.addWidget(btn_reference)

        self.lbl_filename = QLabel("Selected File: None")
        self.lbl_filename.setWordWrap(True)
        control_layout.addWidget(self.lbl_filename)

        # Parameter tuning, grouped by pipeline stage (task #80): the layout
        # order mirrors the execution order, so reading top-to-bottom is
        # reading the data flow. Widget names and item texts are unchanged.
        # A QGroupBox title is plain text - no rich text, so no real subscript -
        # and Unicode has no subscript 'f' to spell "inf". R∞ is the standard
        # EIS symbol for the same quantity (resistance at infinite frequency),
        # so it reads correctly as a single glyph instead of faking a subscript.
        group_pre    = QGroupBox("① Preprocessing (Inductance, Extrapolation, R∞)")
        # Form layout, not a vertical stack: each caption sits BESIDE its
        # control instead of on its own line, which halves the group's height
        # (9 rows -> 5). The captions are short because the group title already
        # names the three stages; the full wording stays in the tooltips.
        param_layout = QFormLayout()                 # stage ① widgets
        param_layout.setSpacing(6)
        param_layout.setContentsMargins(6, 6, 6, 6)
        # A form's minimum width is the label column plus the field column,
        # which exceeded the panel even at the default font; with WrapLongRows
        # a row whose field does not fit beside its label wraps underneath it,
        # so the panel's minimum is bounded by the widest single control
        # rather than by label+field sums (task #109).
        param_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        # Stages 2 and 3 share one group: the smoothing block is only two
        # rows, and a separate header for it cost more height than it earned.
        # Execution order is still read top-to-bottom inside the group
        # (lambda -> deconvolution -> Warburg).
        group_fit    = QGroupBox("② Fitting (λ, Deconvolution, Warburg)")
        fit_layout   = QFormLayout()                 # stage ②+③ widgets
        fit_layout.setSpacing(6)
        fit_layout.setContentsMargins(6, 6, 6, 6)
        fit_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        # Inductance (L) correction mode
        self.combo_l_corr = _shrinkable_combo()
        self.combo_l_corr.addItems(["Auto (Smart Detection)", "Force ON (Force Correction)", "Force OFF (Keep Original)"])
        # Semantics per drt-preprocessing.md 2.1/2.2/2.5 (task #70): the
        # estimation chain is mode-independent; force differs only at gate B.
        self.combo_l_corr.setToolTip(
            "Auto: estimate L (primary fit + short-tail recovery) and apply it only if\n"
            "  it is physical (L > 0) AND significant (>= 0.5% of |Z| at f_max).\n"
            "Force ON: identical estimation; skips only the significance gate. On typical\n"
            "  data this equals Auto - it differs only when a real but tiny L is blocked.\n"
            "Force OFF: no correction. Remaining inductive (Im > 0) points are dropped\n"
            "  by the residual filter, shrinking the fitted window (information loss).")
        param_layout.addRow("Inductance (L):", self.combo_l_corr)

        # Extrapolation range (integer slider)
        dec_layout = QHBoxLayout()
        self.slider_decades = QSlider(Qt.Orientation.Horizontal)
        self.slider_decades.setRange(0, 5)
        self.slider_decades.setValue(2)
        self.spin_decades = QSpinBox()
        self.spin_decades.setRange(0, 5)
        self.spin_decades.setValue(2)
        self.slider_decades.valueChanged.connect(self.spin_decades.setValue)
        self.spin_decades.valueChanged.connect(self.slider_decades.setValue)
        dec_layout.addWidget(self.slider_decades)
        dec_layout.addWidget(self.spin_decades)
        self.slider_decades.setToolTip("Extrapolation range in decades, applied to both "
                                       "the low- and high-frequency ends")
        param_layout.addRow("Ext. range (dec):", dec_layout)

        # Extrapolation method (no regularisation / no lambda):
        #   Auto   -> linear log-scale; fall back to RC fit when an arc is open
        #   Linear -> linear log-scale extrapolation (Eqs. 8-10)
        #   RC     -> physical RC-series model fit (plain NNLS)
        self.combo_ext_method = _shrinkable_combo()
        self.combo_ext_method.addItems(["Auto (Linear → RC fallback)",
                                        "Linear",
                                        "RC fit"])
        # Semantics per drt-preprocessing.md 3.4 (per-end auto, task #55).
        self.combo_ext_method.setToolTip(
            "Auto: picks a method PER frequency edge. The high-frequency tail stays\n"
            "  linear (anchored to the corrected R_inf when the HF arc is open); the\n"
            "  low-frequency tail switches to the RC continuation only for a strongly\n"
            "  open diffusion arc with a trustworthy RC fit. A mixed choice shows as\n"
            "  'RC-LF/LINEAR-HF' in the label below.\n"
            "Linear / RC fit: force one method on BOTH tails. Forcing RC on a closed,\n"
            "  clean arc fragments the DRT into spurious peaks - use only for\n"
            "  genuinely open (diffusion-type) spectra.")
        param_layout.addRow("Ext. method:", self.combo_ext_method)

        # Shows which method Auto actually selected after a run. Added as a
        # single-widget row so it spans both form columns.
        self.lbl_ext_auto = QLabel("Auto result: (run to evaluate)")
        self.lbl_ext_auto.setWordWrap(True)
        self.lbl_ext_auto.setStyleSheet("color: #555; font-style: italic; padding-left: 4px;")
        param_layout.addRow(self.lbl_ext_auto)
        self.combo_ext_method.currentTextChanged.connect(self._on_ext_method_changed)

        # R_inf estimation mode. 'Auto' replaces the min-Z_re estimate with the
        # RC-model-fitted value when the high-frequency arc is open (a too-high
        # fixed R_inf prevents the NNLS from fitting the low-frequency tail).
        rinf_layout = QHBoxLayout()
        self.combo_rinf = _shrinkable_combo()
        self.combo_rinf.addItems(["Auto (open-arc corrected)",
                                  "Min Z_re (legacy)",
                                  "Manual"])
        # Semantics per drt-preprocessing.md 3.1-3.3 (the R_inf ladder).
        self.combo_rinf.setToolTip(
            "Auto: min(Z_re) when the high-frequency arc is closed; when it is OPEN,\n"
            "  min(Z_re) still contains unresolved arc real part, so the estimate is\n"
            "  LOWERED to the RC-model-fitted R_inf (never raised).\n"
            "Min Z_re: legacy - always the high-frequency real-axis minimum.\n"
            "Manual: use the value entered on the right.\n"
            "R_inf is shared by the NNLS fit, the reconstruction and the ECM.")
        self.spin_rinf = QDoubleSpinBox()
        self.spin_rinf.setDecimals(4)
        self.spin_rinf.setRange(0.0, 1e9)
        self.spin_rinf.setValue(0.0)
        self.spin_rinf.setToolTip("R_inf value in ohms (used when set to Manual; "
                                  "shows the value actually used after an Auto run)")
        # Give the mode combo the spare width: sharing the row evenly elided it
        # to "Auto (open-". The value box only needs room for four decimals.
        self.spin_rinf.setMaximumWidth(84)
        rinf_layout.addWidget(self.combo_rinf, 1)
        rinf_layout.addWidget(self.spin_rinf, 0)
        param_layout.addRow("R∞ estimate:", rinf_layout)

        # Smoothing strength mode
        self.combo_lambda_mode = _shrinkable_combo()
        self.combo_lambda_mode.addItems(["Auto (L-Curve)", "Manual"])
        # Semantics per drt-core.md 5.3/5.4; the 7.1 caution is the one known
        # regime where the automatic pick is untrustworthy.
        self.combo_lambda_mode.setToolTip(
            "Auto: scans 50 regularization strengths (window anchored on the Wiener\n"
            "  noise estimate) and picks the L-curve corner. The adopted value is\n"
            "  written back into the box below after the run.\n"
            "Manual: use the value from the box/slider directly.\n"
            "Caution: with widely separated arcs near the low-frequency edge under\n"
            "  noise, the automatic pick under-smooths (fragmented peaks); a manual\n"
            "  lambda >= 0.3 recovers the correct decomposition there.")
        fit_layout.addRow("λ mode:", self.combo_lambda_mode)

        # Smoothing strength value (log slider over 1e-6 .. 1000, covering the
        # full optimize_lambda L-Curve/GCV scan so the auto value is never clamped)
        lam_layout = QHBoxLayout()

        self._lam_log_min, self._lam_log_max = -6.0, 3.0   # log10 bounds
        self._lam_slider_steps = 1000

        self.slider_lambda = QSlider(Qt.Orientation.Horizontal)
        self.slider_lambda.setRange(0, self._lam_slider_steps)

        self.spin_lambda = QDoubleSpinBox()
        self.spin_lambda.setDecimals(6)
        self.spin_lambda.setRange(0.000001, 1000.0)
        self.spin_lambda.setSingleStep(0.01)
        self.spin_lambda.setValue(0.01)

        def lam_slider_to_value(s):
            frac = s / self._lam_slider_steps
            return 10.0 ** (self._lam_log_min + frac * (self._lam_log_max - self._lam_log_min))
        def lam_value_to_slider(lam):
            lam  = min(max(lam, 1e-6), 1000.0)
            frac = (np.log10(lam) - self._lam_log_min) / (self._lam_log_max - self._lam_log_min)
            return int(round(frac * self._lam_slider_steps))

        def update_spin_lambda(val):       # slider moved -> update spin
            self.spin_lambda.blockSignals(True)
            self.spin_lambda.setValue(lam_slider_to_value(val))
            self.spin_lambda.blockSignals(False)
        def update_slider_lambda(val):     # spin changed -> update slider
            self.slider_lambda.blockSignals(True)
            self.slider_lambda.setValue(lam_value_to_slider(val))
            self.slider_lambda.blockSignals(False)

        self.slider_lambda.valueChanged.connect(update_spin_lambda)
        self.spin_lambda.valueChanged.connect(update_slider_lambda)
        self.slider_lambda.setValue(lam_value_to_slider(0.01))  # sync to default
        self.spin_lambda.setMaximumWidth(96)     # six decimals, no more
        self.slider_lambda.setToolTip("Smoothing strength on a log scale, 1e-6 to 1000 "
                                      "(the full automatic scan range)")
        lam_layout.addWidget(self.slider_lambda, 1)
        lam_layout.addWidget(self.spin_lambda, 0)
        fit_layout.addRow("λ value:", lam_layout)

        # Peak model criterion
        self.combo_criterion = _shrinkable_combo()
        self.combo_criterion.addItems(["Conservative (BIC)", "Flexible (AICc)"])
        # Corpus-measured effect: the two criteria differ on exactly one
        # borderline dataset (SCNT3 2 vs 3 peaks); fit quality is identical.
        self.combo_criterion.setToolTip(
            "Penalty strength for the peak-count decision (fit quality is\n"
            "  unaffected - only how many peaks are reported).\n"
            "BIC (default): conservative - rejects borderline peaks.\n"
            "AICc: flexible - admits them (corpus: differs only on SCN_600C,\n"
            "  2 vs 4 peaks; small borderline processes).\n"
            "Whether a borderline peak is real is the experimenter's call,\n"
            "  not the data's - that is why this toggle exists.")
        fit_layout.addRow("Criterion:", self.combo_criterion)

        # Peak deconvolution method (task #42). The legacy LASSO / CWT paths
        # are tier-2 golden-pinned (task #32), so exposing them is safe.
        self.combo_deconv = _shrinkable_combo()
        self.combo_deconv.addItems(["ZARC + BIC", "LASSO + Curve-Fit", "CWT + BIC"])
        # Evidence per drt-deconvolution.md 4.0 (measured on the corpus).
        self.combo_deconv.setToolTip(
            "ZARC + BIC (default): analytic ZARC kernels - the only method that\n"
            "  recovers all five known-truth corpus spectra (5/5).\n"
            "LASSO / CWT: human cross-check tools against independent bases\n"
            "  (Gaussian dictionary / wavelet detection). LASSO fragments broad\n"
            "  arcs (single ZARC -> 5 peaks); CWT fails badly on some measured\n"
            "  data (550C: 1 peak at 21% misfit).\n"
            "Caution: a better misfit from MORE peaks is a trap, not evidence -\n"
            "  fragmentation always fits better.")
        fit_layout.addRow("Deconvolution:", self.combo_deconv)

        # Max number of peaks (integer slider)
        max_peaks_layout = QHBoxLayout()
        self.slider_max_peaks = QSlider(Qt.Orientation.Horizontal)
        self.slider_max_peaks.setRange(1, 15)
        self.slider_max_peaks.setValue(10)
        self.spin_max_peaks = QSpinBox()
        self.spin_max_peaks.setRange(1, 15)
        self.spin_max_peaks.setValue(10)
        self.slider_max_peaks.valueChanged.connect(self.spin_max_peaks.setValue)
        self.spin_max_peaks.valueChanged.connect(self.slider_max_peaks.setValue)
        max_peaks_layout.addWidget(self.slider_max_peaks)
        max_peaks_layout.addWidget(self.spin_max_peaks)
        self.slider_max_peaks.setToolTip("Upper bound on the peak count the model "
                                         "selection may choose (1-15)")
        fit_layout.addRow("Max peaks:", max_peaks_layout)

        # Warburg (diffusion) handling (task #77). Labels are the exact
        # drt-pipeline 3.1 schema strings - the label-matching contract.
        self.combo_warburg = _shrinkable_combo()
        self.combo_warburg.addItems(["Auto (With Tail-Cut)", "Joint (NNLS)", "Off"])
        self.combo_warburg.setToolTip(
            "Auto (default): fit a Warburg-open element post-hoc when a diffusion\n"
            "  tail-cut fires; on diffusion spectra the run is then upgraded to the\n"
            "  joint polish automatically (detect-then-joint).\n"
            "Joint (NNLS): ALWAYS fit the Warburg column jointly with the DRT.\n"
            "  Not the default on purpose: on non-diffusion spectra it asserts a\n"
            "  Warburg that is not there (reviewed false positive).\n"
            "Off: never fit a diffusion element.\n"
            "The fitted record appears below after the run.")
        fit_layout.addRow("Warburg:", self.combo_warburg)

        group_pre.setLayout(param_layout)
        group_fit.setLayout(fit_layout)
        control_layout.addWidget(group_pre)
        control_layout.addWidget(group_fit)

        # Validated-defaults reset (task #80): the defaults ARE the pinned
        # specification (drt-conformance.md 1 - the golden corpus is generated
        # from them), so one click returns to the validated configuration.
        self.btn_defaults = QPushButton("↺ Reset to Validated Defaults")
        self.btn_defaults.setToolTip(
            "Restore the configuration that generates the golden regression\n"
            "corpus (drt-conformance.md): every control back to its default,\n"
            "including clearing a manual tail-cut.")
        self.btn_defaults.clicked.connect(self.reset_to_defaults)
        control_layout.addWidget(self.btn_defaults)

        # Run button
        self.btn_run = QPushButton("🚀 Run DRT Optimization")
        self.btn_run.clicked.connect(self.run_optimization)
        self.btn_run.setStyleSheet("font-weight: bold; padding: 9px; background-color: #4CAF50; color: white;")
        control_layout.addWidget(self.btn_run)

        # Manual peak edit mode toggle
        self.btn_manual_mode = QPushButton("✍️ Manual Peak Edit Mode (OFF)")
        self.btn_manual_mode.setCheckable(True)
        self.btn_manual_mode.clicked.connect(self.toggle_manual_mode)
        control_layout.addWidget(self.btn_manual_mode)

        # Tail-cut mode toggle: click a data point to exclude all lower
        # frequencies (diffusion branch) from peak extraction and refinement.
        self.btn_tailcut_mode = QPushButton("✂️ Tail-Cut Mode (OFF)")
        self.btn_tailcut_mode.setCheckable(True)
        self.btn_tailcut_mode.clicked.connect(self.toggle_tailcut_mode)
        control_layout.addWidget(self.btn_tailcut_mode)

        # Export to Excel button
        # Kept as an attribute: the initial height is measured against this
        # button so it opens above the fold (task #110).
        self.btn_export = QPushButton("💾 Export Results to Excel")
        self.btn_export.clicked.connect(self.export_excel)
        control_layout.addWidget(self.btn_export)

        # System log
        # Shown in every build (task #133). Task #111 hid it in a release
        # build and kept the lines in the per-day file instead; the file is
        # still written, because it is what survives a crash and reads as one
        # transcript across a day's sessions - but a run that narrates itself
        # is worth more on screen than the vertical space it costs.
        control_layout.addWidget(QLabel("pyFDRT Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        control_layout.addWidget(self.log_text)

        # Wrap the control column in a scroll area. Stacked, the three stage
        # groups plus the log are ~1150 px tall, and as a plain child widget
        # that minimum forced the whole window to that height regardless of
        # the requested geometry. Scrolled, the window opens at whatever
        # height is asked for and the controls stay fully reachable.
        # Wheel guard: with the panel scrollable, an unguarded wheel over a
        # control changes that control instead of scrolling (see _WheelGuard).
        self._wheel_guard = _WheelGuard(self)
        self._wheel_guard.protect(
            self.combo_l_corr, self.combo_ext_method, self.combo_rinf,
            self.combo_lambda_mode, self.combo_criterion, self.combo_deconv,
            self.combo_warburg,
            self.spin_decades, self.spin_rinf, self.spin_lambda,
            self.spin_max_peaks,
            self.slider_decades, self.slider_lambda, self.slider_max_peaks)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setWidget(control_panel)
        self._wheel_guard.area = control_scroll
        # 320 (panel) + room for the vertical scrollbar, so the panel never
        # needs a horizontal one.
        control_scroll.setFixedWidth(_panel_w + 20)
        self._control_scroll = control_scroll
        main_layout.addWidget(control_scroll)

        # ==========================================
        # Right plot panel
        # ==========================================
        plot_panel  = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.fig_nd    = Figure(figsize=(13, 5))
        self.canvas_nd = FigureCanvas(self.fig_nd)
        self.toolbar_nd = NavigationToolbar(self.canvas_nd, self)
        plot_layout.addWidget(self.toolbar_nd)
        plot_layout.addWidget(self.canvas_nd)
        self.ax_nyquist = self.fig_nd.add_subplot(121)
        self.ax_drt     = self.fig_nd.add_subplot(122)
        self.fig_nd.tight_layout()
        self.canvas_nd.mpl_connect('button_press_event', self.on_plot_click)
        main_layout.addWidget(plot_panel)

    def toggle_manual_mode(self, checked):
        if checked:
            self.btn_manual_mode.setText("✍️ Manual Peak Edit Mode (ON)")
            self.btn_manual_mode.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
            self.log("-> [Mode] Manual peak edit mode activated "
                     "(DRT plot — Left-click: Add, Right-click: Remove)")
        else:
            self.btn_manual_mode.setText("✍️ Manual Peak Edit Mode (OFF)")
            self.btn_manual_mode.setStyleSheet("")
            self.log("-> [Mode] Manual peak edit mode deactivated")

    def toggle_tailcut_mode(self, checked):
        if checked:
            self.btn_tailcut_mode.setText("✂️ Tail-Cut Mode (ON)")
            self.btn_tailcut_mode.setStyleSheet("background-color: #DC143C; color: white; font-weight: bold;")
            self.log("-> [Mode] Tail-cut mode activated "
                     "(Nyquist plot — Left-click a data point: exclude lower "
                     "frequencies, Right-click: restore Auto)")
        else:
            self.btn_tailcut_mode.setText("✂️ Tail-Cut Mode (OFF)")
            self.btn_tailcut_mode.setStyleSheet("")
            self.log("-> [Mode] Tail-cut mode deactivated")

    def on_plot_click(self, event):
        """Route plot clicks by axes: tail-cut on the Nyquist plot, manual
        peak edit on the DRT plot. Both modes can be active at the same time.
        """
        if event.inaxes == self.ax_nyquist:
            if self.btn_tailcut_mode.isChecked():
                self._on_tailcut_click(event)
            return
        if event.inaxes != self.ax_drt:
            return
        if not self.btn_manual_mode.isChecked():
            return
        if self.optimizer is None or not hasattr(self.optimizer, 'active_peaks'):
            self.log("-> [Warning] Run optimization first before editing peaks.")
            return

        x_click = event.xdata
        if x_click is None:
            return

        opt = self.optimizer
        idx = np.argmin(np.abs(opt.log_tau - x_click))

        if event.button == 1:  # left-click: add peak
            if idx not in opt.active_peaks:
                opt.active_peaks.append(idx)
                self.log(f"-> [Manual Edit] Peak added: log(tau) = {opt.log_tau[idx]:.2f}")
        elif event.button == 3:  # right-click: remove peak
            if not opt.active_peaks:
                return
            closest = min(opt.active_peaks, key=lambda p: abs(opt.log_tau[p] - x_click))
            if abs(opt.log_tau[closest] - x_click) < 0.5:
                opt.active_peaks.remove(closest)
                self.log(f"-> [Manual Edit] Peak removed: log(tau) = {opt.log_tau[closest]:.2f}")
            else:
                return
        else:
            return

        try:
            self.results = opt.deconvolve_peaks(manual_peaks=opt.active_peaks,
                                                tail_cut=self._current_tail_cut())
            self.update_plots()
        except Exception as e:
            self.alert(f"-> [Error] Manual fitting failed: {str(e)}")

    def _current_tail_cut(self):
        """Tail-cut argument for deconvolve_peaks: manual tau or 'auto'."""
        return self.manual_tail_cut_tau if self.manual_tail_cut_tau is not None else 'auto'

    def _on_tailcut_click(self, event):
        """Set (left-click) or clear (right-click) the manual diffusion tail-cut.

        Operates on the Nyquist plot only: clicking a measured point excludes
        all frequencies below it. The cut applies to both the peak
        deconvolution and the impedance-domain ZARC refinement.
        """
        if self.optimizer is None:
            self.log("-> [Warning] Run optimization first before setting a tail-cut.")
            return

        if event.button == 3:  # right-click: back to automatic detection
            if self.manual_tail_cut_tau is not None:
                self.manual_tail_cut_tau = None
                self.log("-> [Tail-Cut] Manual cut cleared; automatic detection restored.")
                self._redeconvolve()
            return
        if event.button != 1 or event.xdata is None:
            return

        # Nearest measured data point in *screen* coordinates, so the
        # click tolerance does not depend on the axis scales.
        data = self.optimizer.target_data
        pts  = self.ax_nyquist.transData.transform(
            np.column_stack([data[:, 1], -data[:, 2]]))
        d2 = (pts[:, 0] - event.x) ** 2 + (pts[:, 1] - event.y) ** 2
        i  = int(np.argmin(d2))
        if d2[i] > 20.0 ** 2:   # require the click within 20 px of a point
            self.log("-> [Tail-Cut] No data point near the click; "
                     "click closer to a measured point.")
            return
        f_cut   = float(data[i, 0])
        tau_cut = 1.0 / (2 * np.pi * f_cut)

        self.manual_tail_cut_tau = tau_cut
        self.log(f"-> [Tail-Cut] Manual cut set: excluding f < {f_cut:.3g} Hz "
                 f"(tau > {tau_cut:.3g} s)")
        self._redeconvolve()

    def _redeconvolve(self):
        """Re-run peak deconvolution + ZARC refinement with the current
        tail-cut. Preprocessing and the NNLS polish do not depend on the cut,
        so they are reused — this makes the click feedback near-instant."""
        opt = self.optimizer
        crit = Criterion.from_label(self.combo_criterion.currentText()).value
        try:
            self.results = opt.deconvolve_peaks(
                max_peaks=self.spin_max_peaks.value(), criterion=crit,
                method=DeconvolutionMethod.from_label(
                    self.combo_deconv.currentText()).value,
                tail_cut=self._current_tail_cut())
            # The optimizer may reject a manual cut that leaves too few points;
            # mirror that in the GUI state so stale Manual settings are not
            # silently re-sent on every subsequent run.
            if (self.manual_tail_cut_tau is not None
                    and opt.deconv_tail_cut_tau is None):
                self.log("-> [Tail-Cut] Manual cut rejected (too few points "
                         "remain); reverting to automatic detection.")
                self.manual_tail_cut_tau = None
            pk = self.results.get('Peaks', [])
            self.log(f"-> [Tail-Cut] Re-deconvolution complete: {len(pk)} peak(s)")
            self.update_plots()
        except Exception as e:
            self.alert(f"-> [Error] Re-deconvolution failed: {str(e)}")

    def log(self, message):
        """Record a message: to the per-day file and to the pane.

        The pane is guarded because ``log`` is reachable before ``initUI``
        has built it, not because a build might lack one.
        """
        self._file_log.write(message)
        if self.log_text is not None:
            self.log_text.append(message)
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum())
        # Also keeps the window responsive during a long run.
        QApplication.processEvents()

    def alert(self, message):
        """Report a failure. Logged either way; in a release build it also
        gets a dialog.

        The dialog was added when the release build had no log pane (task
        #111). It survives the pane's return (task #133) on its own merit:
        someone running a distributed build is not watching a log pane, and a
        modal is the one channel a failure cannot slip past. It also names the
        log file, which is where the detail is.
        """
        self.log(message)
        if self._release:
            where = (f"\n\nLog file:\n{self._file_log.path}"
                     if self._file_log.path else "")
            QMessageBox.warning(self, "pyFDRT", f"{message}{where}")

    def _on_ext_method_changed(self, text):
        """Reset the Auto-result label when the extrapolation method changes."""
        ext = ExtMethod.from_label(text)
        if ext is ExtMethod.AUTO:
            self.lbl_ext_auto.setText("Auto result: (run to evaluate)")
        else:
            method = "LINEAR" if ext is ExtMethod.LINEAR else "RC"
            self.lbl_ext_auto.setText(f"Method: {method} (manual selection)")
        self.lbl_ext_auto.setStyleSheet("color: #555; font-style: italic; padding-left: 4px;")

    def _adopt_new_data(self, data, label):
        """Take `data` as the current spectrum and drop everything derived.

        Shared by the file dialog and the built-in example (task #122). The
        invalidation is the whole point: Export, Manual Peak Edit and Tail-Cut
        all read state produced by the previous run, so a second load that
        forgot any line here would let them mix old results with new data. One
        copy is how the two entry points stay unable to disagree.
        """
        self.raw_data = data
        self.optimizer = None
        self.results   = None
        self.manual_tail_cut_tau = None   # cut belongs to the old data
        if self.btn_manual_mode.isChecked():
            self.btn_manual_mode.setChecked(False)
            self.toggle_manual_mode(False)
        if self.btn_tailcut_mode.isChecked():
            self.btn_tailcut_mode.setChecked(False)
            self.toggle_tailcut_mode(False)
        self.lbl_filename.setText(f"Selected File: {label}")
        self.plot_raw_data_only()

    def load_data(self):
        start_dir = DATA_DIR if os.path.isdir(DATA_DIR) else ''
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Data File', start_dir, "Data Files (*.txt *.csv *.dat *.z);;ZView Files (*.z);;All Files (*)")
        if fname:
            try:
                self._adopt_new_data(self.analyzer.load_data(fname),
                                     os.path.basename(fname))
            except Exception as e:
                self.alert(f"-> [Error] Failed to read file: {str(e)}")

    def load_reference_spectrum(self):
        """Load the built-in reference spectrum (drt-io.md 2.5, task #122).

        Written to a temporary file and read back through the ordinary loader
        rather than handed over as an array: the example must exercise the same
        parse, monotonicity check and sign handling a user's own file does, or
        it could keep working while the format it claims to be in had broken.
        """
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'reference_spectrum.txt')
                with open(path, 'w', encoding='utf-8') as fh:
                    fh.write(reference_spectrum_text())
                data = self.analyzer.load_data(path)
            self._adopt_new_data(data, 'built-in example spectrum')
            # Construction parameters only. What the pipeline recovers from it
            # is not stated: no golden pins this spectrum (drt-io.md 2.5).
            self.log(f"-> [Example] {reference_spectrum_summary()}")
        except Exception as e:
            self.alert(f"-> [Error] Failed to build the example spectrum: {str(e)}")

    def plot_raw_data_only(self):
        """Display raw data before running optimization."""
        draw_raw_data_panels(self.fig_nd, self.ax_nyquist, self.ax_drt,
                             self.raw_data)
        self.canvas_nd.draw()

    def reset_to_defaults(self):
        """Restore the validated default configuration (task #80).

        These are the defaults that generate the golden regression corpus
        (drt-conformance.md section 1) - resetting also clears a manual
        tail-cut so the next run is the pinned configuration exactly.
        """
        self.combo_l_corr.setCurrentIndex(0)      # Auto (Smart Detection)
        self.spin_decades.setValue(2)
        self.combo_ext_method.setCurrentIndex(0)  # Auto
        self.combo_rinf.setCurrentIndex(0)        # Auto (open-arc corrected)
        self.spin_rinf.setValue(0.0)
        self.combo_lambda_mode.setCurrentIndex(0) # Auto (L-Curve)
        self.spin_lambda.setValue(0.01)
        self.combo_criterion.setCurrentIndex(0)   # Conservative (BIC)
        self.combo_deconv.setCurrentIndex(0)      # ZARC + BIC
        self.spin_max_peaks.setValue(10)
        self.combo_warburg.setCurrentIndex(0)     # Auto (With Tail-Cut)
        if self.manual_tail_cut_tau is not None:
            self.manual_tail_cut_tau = None
            self.log("-> [Defaults] Manual tail-cut cleared; automatic detection restored.")
        self.log("-> [Defaults] Restored the validated default configuration "
                 "(the golden-corpus settings).")

    def run_optimization(self):
        if self.raw_data is None:
            self.log("-> [Warning] Please load data first!")
            return
        # log() pumps the Qt event loop (processEvents), so a second click on
        # the Run button could re-enter this method mid-pipeline. Guard it.
        if self._running:
            return
        self._running = True
        self.btn_run.setEnabled(False)

        config = {
            'l_corr_mode': self.combo_l_corr.currentText(),
            'ext_decades': self.spin_decades.value(),
            'ext_method': self.combo_ext_method.currentText(),
            'lambda_mode': self.combo_lambda_mode.currentText(),
            'lambda_val': self.spin_lambda.value(),
            'max_peaks': self.spin_max_peaks.value(),
            'criterion': self.combo_criterion.currentText(),
            'deconvolution_method': self.combo_deconv.currentText(),
            'tail_cut_mode': 'Manual' if self.manual_tail_cut_tau is not None else 'Auto',
            'tail_cut_tau': (self.manual_tail_cut_tau
                             if self.manual_tail_cut_tau is not None else 1.0),
            'r_inf_mode': self.combo_rinf.currentText(),
            'r_inf_value': self.spin_rinf.value(),
            'warburg_mode': self.combo_warburg.currentText()
        }

        try:
            cfg = PipelineConfig.from_dict(config)
            # Route the optimizer's print()-based diagnostics into the log
            # pane for the duration of the run (task #108). The context
            # manager restores sys.stdout on every exit path.
            _tee = _StdoutToLog(self.log)
            with redirect_stdout(_tee):
                res = self.analyzer.run(cfg)
            _tee.flush()
            self.optimizer = res['optimizer']
            self.results = res['results']

            # Mirror a rejected manual tail-cut (too few points) in GUI state.
            if (self.manual_tail_cut_tau is not None
                    and self.optimizer.deconv_tail_cut_tau is None):
                self.log("-> [Tail-Cut] Manual cut rejected (too few points "
                         "remain); reverting to automatic detection.")
                self.manual_tail_cut_tau = None

            # Update UI values calculated during Auto optimization
            if cfg.lambda_mode is LambdaMode.AUTO:
                self.spin_lambda.setValue(res['lambda_reg'])
            if cfg.r_inf_mode is not RInfMode.MANUAL:
                self.spin_rinf.setValue(float(res['ext_dbg'].get('R_inf', 0.0)))

            _n_rc = res['ext_dbg'].get('rc_n_elements')
            _rc_suffix = (f" ({_n_rc} RC elements)"
                          if 'rc' in res['ext_dbg']['method'] and _n_rc else "")
            if cfg.ext_method is ExtMethod.AUTO:
                chosen = res['ext_dbg']['method'].upper()
                self.lbl_ext_auto.setText(f"Auto → {chosen}{_rc_suffix}")
                self.lbl_ext_auto.setStyleSheet("color: #0a7d28; font-weight: bold; padding-left: 4px;")
            else:
                method_name = res['ext_dbg']['method'].upper()
                self.lbl_ext_auto.setText(f"Method: {method_name} (manual selection){_rc_suffix}")
                self.lbl_ext_auto.setStyleSheet("color: #555; font-style: italic; padding-left: 4px;")

            # Warburg record (task #77). Per drt-core.md 5.5: when the knee
            # lies beyond the measured window only C_lim is identified, so
            # show C_lim there instead of the non-unique (R_W, tau_W) split.
            wb = self.results.get('Warburg')
            if wb:
                if 'slow_zarc' in wb:
                    kind = 'composite'
                elif 'joint' in str(wb.get('model', '')):
                    kind = 'joint'
                else:
                    kind = 'post-hoc'
                up = res['ext_dbg'].get('warburg_upgrade')
                if up:
                    kind += ', auto-upgraded'
                if wb.get('knee_resolved', True):
                    detail = f"R_W = {wb['R_W']:.4g} Ω, τ_W = {wb['Tau_W']:.3g} s"
                elif wb.get('C_lim') is None:
                    detail = ("diffusion column collapsed onto the search floor: "
                              "no knee identified, C_lim withheld")
                else:
                    detail = (f"C_lim = {wb['C_lim']:.3g} F "
                              "(knee beyond window: R_W/τ_W not unique)")
                if 'slow_zarc' in wb:
                    sz = wb['slow_zarc']
                    detail += (f" + slow element τ = {sz['Tau']:.3g} s, "
                               f"φ = {sz['Phi']:.2f} (task #89)")
                # Into the System Log (same move as the run summary, task
                # #104): the interpreted record - kind incl. the
                # detect-then-joint upgrade, and the identified quantities -
                # lands right above the summary line, and the wrapped
                # three-line label row is returned to the panel.
                self.log(f"-> [Warburg] ({kind}): {detail}")
            else:
                self.log("-> [Warburg] none (no diffusion element fitted)")

            # Run summary strip (task #79) - one line, everything at a glance.
            n_pk = len(self.results.get('Peaks', []))
            L_val = res.get('L_val', 0.0)
            # Kept for the Excel export, which receives the optimizer and the
            # results but not the pipeline's own output, where L_val lives.
            self._last_L_val = L_val
            L_s = f"L={L_val:.3g} H" if L_val else "L=0"
            lam_s = (f"λ={res['lambda_reg']:.4g} "
                     f"({'auto' if cfg.lambda_mode is LambdaMode.AUTO else 'manual'})")
            ext_s = f"ext={res['ext_dbg']['method'].upper()}"
            if wb:
                wb_s = "Warburg: " + ('composite' if 'slow_zarc' in wb
                                      else 'joint' if 'joint' in str(wb.get('model', ''))
                                      else 'post-hoc')
            else:
                wb_s = "Warburg: none"
            gate = res['ext_dbg'].get('kk_gate') or {}
            if 'skipped' in gate:
                kk_s = "KK: off"
            elif gate.get('ok'):
                kk_s = f"KK: PASS ({gate['rms']:.2%})"
            else:
                kk_s = "KK: ⚠ WARNING"
            # R-budget (task #87): reported-element R total vs windowed NNLS
            # mass; flagged when the ratio leaves the tolerance band.
            rb = getattr(self.optimizer, 'r_budget', None)
            if rb:
                from pyfdrt.core.DRTOptimizer import R_BUDGET_TOL
                flag = "⚠ " if abs(rb['ratio'] - 1.0) > R_BUDGET_TOL else ""
                rb_s = f"Rb: {flag}{rb['ratio']:.0%}"
            else:
                rb_s = "Rb: —"
            # The one-line summary (task #79) goes to the System Log instead
            # of a dedicated strip: the log keeps the history of every run and
            # auto-scrolls, so the latest summary is still the last thing on
            # screen while one widget row of panel height is returned.
            cdc = circuit_cdc(self.results, L_val)
            self.log(
                f"✓ {n_pk} peak{'s' if n_pk != 1 else ''} · {cdc} · {L_s} · {lam_s} · "
                f"{ext_s} · {wb_s} · {rb_s} · {kk_s}")

            self.update_plots()
        except Exception as e:
            self.alert(f"-> [Fatal Error] {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
            self.btn_run.setEnabled(True)

    def update_plots(self):
        """Refresh the Nyquist and DRT panels after an optimization run.

        The drawing itself lives in the package (task #120) so this window
        and the browser build cannot describe the same run differently.
        """
        draw_analysis_panels(self.fig_nd, self.ax_nyquist, self.ax_drt,
                             self.optimizer, self.results, self.raw_data)
        self.canvas_nd.draw()

    def export_excel(self):
        if self.optimizer is None or self.results is None:
            self.log("-> [Warning] No optimization results to export.")
            return

        os.makedirs(RESULTS_DIR, exist_ok=True)
        default_path = os.path.join(RESULTS_DIR, 'DRT_Results.xlsx')
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Results', default_path, "Excel Files (*.xlsx)")
        if fname:
            try:
                export_results_to_excel(fname, self.optimizer, self.results,
                                        logger=self.log,
                                        inductance=self._last_L_val)
            except ExcelExportError as e:
                # The exporter already logged the underlying error; catch it here
                # so a write failure is reported to the user instead of escaping
                # the Qt slot as an unhandled exception.
                self.alert(f"-> [Error] Export failed, no file written: {e}")


if __name__ == '__main__':
    # On Windows, set an explicit AppUserModelID so the taskbar shows the
    # pyFDRT icon instead of the generic Python interpreter icon.
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "pyFDRT.app.1.0")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    if os.path.exists(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    main_window = DRTApp()
    main_window.show()
    sys.exit(app.exec())

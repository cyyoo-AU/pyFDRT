"""Typed configuration for :class:`DRTAnalyzer.run`.

This module is the single boundary adapter between human-readable GUI combo-box
labels and the pipeline. Label interpretation lives here, in the ``from_label``
classmethods, and is **whole-value**: each accepted spelling is matched exactly,
and anything else raises ``ValueError`` naming the key and listing the accepted
values (task #95). No substring or prefix matching — the pipeline body and the
GUI call into these instead of matching strings themselves. See
docs/specs/drt-pipeline.md section 3.

The label vocabulary and the enum ``.value`` strings are deliberately separate
(task #95, acceptance criterion 2): labels are the GUI/CLI boundary vocabulary
pinned by the spec's section-3 tables, while ``.value`` strings are the
internal strings the optimizer expects and are NOT accepted as labels.
Programmatic callers skip labels entirely by passing enum members to
:meth:`PipelineConfig.from_dict`, which accepts them unchanged.

The default set is not arbitrary: ``DRTAnalyzer.run()`` with no argument uses
``DEFAULT_PIPELINE_CONFIG``, and that is the configuration the golden regression
corpus is generated with. Treat any change to a default as a behaviour change
under docs/specs/drt-conformance.md.
"""

from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional

# Bump when the default set or the schema changes. The golden corpus is pinned
# against CONFIG_VERSION 1; a bump signals that goldens may need regeneration
# under the regression protocol.
# Version 2: warburg_mode added (task #37). Additive only — the Warburg-open
# fit runs strictly after every pinned quantity is produced (see
# docs/specs/drt-conformance.md section 3.1), so no golden regeneration.
# Version 3: ext_decades_low added (task #41). Additive only - the default
# (None = follow ext_decades) keeps the pinned symmetric behaviour.
# Version 4: ext_range_mode added (task #43). Additive only - the default
# (FIXED) keeps the pinned single-pass behaviour.
# Version 5: kk_gate_mode added (task #45). Additive only - the gate is a
# warn-only diagnostic recorded in unpinned ext_dbg; pinned outputs untouched.
# Version 6: WarburgMode gains JOINT (task #56). Additive only - the default
# (AUTO, post-hoc fit) keeps the pinned polish path byte-identical.
# Version 7: AUTO semantics upgraded to detect-then-joint (task #60). A
# DEFAULT-BEHAVIOUR change on diffusion spectra: tier-1 goldens regenerated
# for the two datasets whose tail-cut fires (LIB_data, LIB_data_2).
CONFIG_VERSION = 7


def _match_whole_label(key, label, accepted):
    """Whole-value label matching for the ``from_label`` classmethods
    (drt-pipeline.md section 3, task #95).

    ``accepted`` maps each accepted spelling to its enum member. Anything not
    in the mapping — including substrings, prefixes, and case variants of
    accepted spellings — raises ``ValueError`` naming the configuration key
    and listing the accepted values, mirroring the unknown-key error of
    ``PipelineConfig.from_dict``.
    """
    s = label if isinstance(label, str) else str(label)
    try:
        return accepted[s]
    except KeyError:
        raise ValueError(
            f"Unrecognized value for {key}: {s!r}. Accepted values: "
            + ", ".join(repr(k) for k in accepted) + "."
        ) from None


class LCorrMode(Enum):
    """Inductance (Kramers-Kronig) correction mode."""
    AUTO = "auto"
    FORCE_ON = "force_on"
    FORCE_OFF = "force_off"

    @classmethod
    def from_label(cls, label) -> "LCorrMode":
        return _match_whole_label("l_corr_mode", label, {
            "Auto (Smart Detection)": cls.AUTO,
            "Force ON (Force Correction)": cls.FORCE_ON,
            "Force OFF (Keep Original)": cls.FORCE_OFF,
        })


class ExtMethod(Enum):
    """Frequency-window extrapolation method. Values are the strings
    ``dataExtrapolationKK`` expects."""
    AUTO = "auto"
    LINEAR = "linear"
    RC = "rc"

    @classmethod
    def from_label(cls, label) -> "ExtMethod":
        # The one key with two accepted spellings per meaning: the short forms
        # are the GUI combo's frozen item texts, the long forms are the active
        # test/golden-generator vocabulary (drt-pipeline.md section 3.3.1).
        # Whole-value matching also retires the historical trap where the
        # default label — which *contains* the word 'Linear' — matched a
        # substring test and silently forced linear mode.
        return _match_whole_label("ext_method", label, {
            "Auto (Linear → RC fallback)": cls.AUTO,
            "Linear (Force Linear)": cls.LINEAR,
            "Linear": cls.LINEAR,
            "RC fit (Force RC)": cls.RC,
            "RC fit": cls.RC,
        })


class LambdaMode(Enum):
    """Tikhonov strength selection: L-Curve search (AUTO) or a fixed value."""
    AUTO = "auto"
    MANUAL = "manual"

    @classmethod
    def from_label(cls, label) -> "LambdaMode":
        return _match_whole_label("lambda_mode", label, {
            "Auto (L-Curve)": cls.AUTO,
            "Manual": cls.MANUAL,
        })


class Criterion(Enum):
    """Model-selection criterion. Values are the strings
    ``deconvolve_peaks`` expects."""
    BIC = "BIC"
    AICC = "AICc"

    @classmethod
    def from_label(cls, label) -> "Criterion":
        return _match_whole_label("criterion", label, {
            "Conservative (BIC)": cls.BIC,
            "Flexible (AICc)": cls.AICC,
        })


class DeconvolutionMethod(Enum):
    """Peak deconvolution basis. Values are the strings
    ``deconvolve_peaks`` expects."""
    ZARC_BIC = "ZARC + BIC"
    LASSO_CURVEFIT = "LASSO + Curve-Fit"
    CWT_BIC = "CWT + BIC"

    @classmethod
    def from_label(cls, label) -> "DeconvolutionMethod":
        return _match_whole_label("deconvolution_method", label, {
            "ZARC + BIC": cls.ZARC_BIC,
            "LASSO + Curve-Fit": cls.LASSO_CURVEFIT,
            "CWT + BIC": cls.CWT_BIC,
        })


class TailCutMode(Enum):
    """Diffusion tail-cut control (ZARC method only)."""
    AUTO = "auto"
    MANUAL = "manual"
    OFF = "off"

    @classmethod
    def from_label(cls, label) -> "TailCutMode":
        return _match_whole_label("tail_cut_mode", label, {
            "Auto": cls.AUTO,
            "Manual": cls.MANUAL,
            "Off": cls.OFF,
        })


class RInfMode(Enum):
    """R_inf estimation mode."""
    AUTO = "auto"
    MIN_ZRE = "min_zre"
    MANUAL = "manual"

    @classmethod
    def from_label(cls, label) -> "RInfMode":
        return _match_whole_label("r_inf_mode", label, {
            "Auto (open-arc corrected)": cls.AUTO,
            "Min Z_re (legacy)": cls.MIN_ZRE,
            "Manual": cls.MANUAL,
        })


class WarburgMode(Enum):
    """Warburg diffusion-branch handling.

    AUTO (default): detect-then-joint (task #60) - run the default pass
    with the post-hoc Warburg-open fit (task #37); when the diffusion
    tail-cut fires, re-run with the JOINT polish and adopt it. OFF: never
    fit. JOINT (task #56): unconditionally fit the Warburg-open element
    jointly with the RC basis inside the NNLS polish (drt-core.md 5.5), so
    the calculated-DRT curve reproduces the measured low-frequency
    diffusion rise; the post-hoc fit is then skipped. Unconditional JOINT
    is deliberately NOT the default: it asserts a diffusion element on
    spectra that never showed one (false positive measured on synthetic
    negami_data).
    AUTO/OFF values are the strings ``deconvolve_peaks`` expects for its
    ``warburg`` argument; JOINT is resolved by the pipeline (which passes
    'off' to the deconvolution and reports the joint parameters instead).
    """
    AUTO = "auto"
    OFF = "off"
    JOINT = "joint"

    @classmethod
    def from_label(cls, label) -> "WarburgMode":
        return _match_whole_label("warburg_mode", label, {
            "Auto (With Tail-Cut)": cls.AUTO,
            "Joint (NNLS)": cls.JOINT,
            "Off": cls.OFF,
        })


class ExtRangeMode(Enum):
    """Extrapolation-range policy (drt-pipeline.md 2.4, task #43).

    FIXED uses the configured decade counts as-is (the pinned default).
    AUTO runs the guarded two-pass procedure: closed arc edges are reduced to
    zero decades and the reduced result is adopted only when the extracted ECM
    is stable (identical peak count, R/Tau drift < 5 %) and the reconstruction
    misfit is not worse.
    """
    FIXED = "fixed"
    AUTO = "auto"

    @classmethod
    def from_label(cls, label) -> "ExtRangeMode":
        return _match_whole_label("ext_range_mode", label, {
            "Fixed": cls.FIXED,
            "Auto": cls.AUTO,
        })


class KKGateMode(Enum):
    """lin-KK measurement-validity gate control (drt-preprocessing.md 2.6,
    task #45). WARN (default) runs the warn-only gate after inductance
    correction; OFF skips it. The gate never rejects a run either way."""
    WARN = "warn"
    OFF = "off"

    @classmethod
    def from_label(cls, label) -> "KKGateMode":
        return _match_whole_label("kk_gate_mode", label, {
            "Warn": cls.WARN,
            "Off": cls.OFF,
        })


# Per-field coercion tables used by ``from_dict``. Keeping them next to the
# dataclass keeps the 20-key schema in one place.
_ENUM_FIELDS = {
    "l_corr_mode": LCorrMode,
    "ext_method": ExtMethod,
    "lambda_mode": LambdaMode,
    "criterion": Criterion,
    "deconvolution_method": DeconvolutionMethod,
    "tail_cut_mode": TailCutMode,
    "r_inf_mode": RInfMode,
    "ext_range_mode": ExtRangeMode,
    "kk_gate_mode": KKGateMode,
    "warburg_mode": WarburgMode,
}
_INT_FIELDS = {"ext_decades", "ext_decades_low", "max_peaks", "lasso_channels"}
_FLOAT_FIELDS = {"lambda_val", "tail_cut_tau", "r_inf_value", "lambda_prior", "lambda_lasso"}


@dataclass(frozen=True)
class PipelineConfig:
    """Validated, typed configuration for the DRT pipeline (20 keys).

    Build one from GUI/CLI labels with :meth:`from_dict`, which is the only place
    label strings are interpreted. Fields are enums or scalars, never labels, so
    the pipeline body branches on identity (``cfg.ext_method is ExtMethod.AUTO``)
    rather than substring tests.
    """

    l_corr_mode: LCorrMode = LCorrMode.AUTO
    ext_decades: int = 2
    # None = symmetric (follow ext_decades). Set to override the LOW side only;
    # the high side always follows ext_decades (drt-preprocessing.md 6.2, task #41).
    ext_decades_low: Optional[int] = None
    ext_method: ExtMethod = ExtMethod.AUTO
    lambda_mode: LambdaMode = LambdaMode.AUTO
    lambda_val: float = 0.01
    max_peaks: int = 10
    criterion: Criterion = Criterion.BIC
    deconvolution_method: DeconvolutionMethod = DeconvolutionMethod.ZARC_BIC
    tail_cut_mode: TailCutMode = TailCutMode.AUTO
    tail_cut_tau: float = 1.0
    warburg_mode: WarburgMode = WarburgMode.AUTO
    r_inf_mode: RInfMode = RInfMode.AUTO
    ext_range_mode: ExtRangeMode = ExtRangeMode.FIXED
    kk_gate_mode: KKGateMode = KKGateMode.WARN
    r_inf_value: float = 0.0
    lambda_prior: float = 0.05
    solver_method: str = "bvls"
    lambda_lasso: float = 0.003
    lasso_channels: int = 100

    @classmethod
    def from_dict(cls, config=None) -> "PipelineConfig":
        """Build a config from a label dict (the GUI/CLI boundary).

        Absent keys take their default. An **unknown key raises** ``ValueError``
        naming the offending key, instead of being silently ignored — and so
        does an **unrecognized enum value** (task #95): enum fields accept
        either one of the exact accepted spellings (translated via
        ``from_label``) or an already-resolved enum member, never a substring
        or prefix. Numeric fields are coerced to int/float.
        ``None`` and an existing :class:`PipelineConfig` pass through.
        """
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config

        known = {f.name for f in fields(cls)}
        unknown = [k for k in config if k not in known]
        if unknown:
            raise ValueError(
                f"Unknown pipeline configuration key(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}."
            )

        kwargs = {}
        for key, value in config.items():
            if key in _ENUM_FIELDS:
                enum_cls = _ENUM_FIELDS[key]
                kwargs[key] = value if isinstance(value, enum_cls) else enum_cls.from_label(value)
            elif key in _INT_FIELDS:
                # ext_decades_low accepts None: symmetric, follow ext_decades.
                if value is None and key == 'ext_decades_low':
                    kwargs[key] = None
                else:
                    kwargs[key] = int(value)
            elif key in _FLOAT_FIELDS:
                kwargs[key] = float(value)
            else:  # solver_method (free-form string)
                kwargs[key] = value

        # A negative decade count has no meaning and would feed np.linspace a
        # negative point count deep inside the extrapolation; reject it here
        # with a clear error instead (task #39). Zero is valid: no extension.
        if kwargs.get('ext_decades', 0) < 0:
            raise ValueError(
                f"ext_decades must be >= 0 (0 means no extension), "
                f"got {kwargs['ext_decades']}.")
        if kwargs.get('ext_decades_low') is not None and kwargs['ext_decades_low'] < 0:
            raise ValueError(
                f"ext_decades_low must be >= 0 or None (None means follow "
                f"ext_decades), got {kwargs['ext_decades_low']}.")
        return cls(**kwargs)


# The single named, versioned default preset. `DRTAnalyzer.run()` with no
# argument resolves to this, so it is the anchor of the conformance contract.
DEFAULT_PIPELINE_CONFIG = PipelineConfig()

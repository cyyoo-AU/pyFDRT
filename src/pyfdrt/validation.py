"""Public data-validation utilities for EIS spectra.

Two Kramers-Kronig consistency tools live here, aimed at different users:

* :func:`kk_gate` - the automated, pipeline-integrated measurement-validity
  gate (drt-preprocessing.md 2.6). The pipeline runs it for you; import it
  only to gate a spectrum outside the pipeline.
* :func:`kk_correction` - a standalone manual-validation utility. It KK-
  transforms a spectrum in both directions, rescales each transform onto the
  measurement, and returns the corrected components so a user can inspect the
  KK-consistency residual by hand (drt-core.md 2.4). It is NOT part of the
  pipeline and never has been; task #46 designates it public rather than
  internal machinery, after task #45 established that its direct-integration
  residual is unsuitable as an automated gate (use :func:`kk_gate` for that).
"""
from pyfdrt.core.kk_transform import kk_correction, kkim2re, kkre2im
from pyfdrt.core.linKK import kk_gate

__all__ = ["kk_correction", "kk_gate", "kkim2re", "kkre2im"]

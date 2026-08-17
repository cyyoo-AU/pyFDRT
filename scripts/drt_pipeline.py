"""Compatibility shim: DRTAnalyzer moved to ``pyfdrt.pipeline`` (task #15).

The orchestrator now lives in the installed package and is re-exported as
``pyfdrt.DRTAnalyzer``. This module is kept so existing entry points that do
``from drt_pipeline import DRTAnalyzer`` (run_batch.py, main_app.py,
DRT_main.py, DRT_w_optimize_main.py) keep working while ``scripts`` is on the
path. New code should import from ``pyfdrt`` directly.
"""
from pyfdrt.pipeline import DRTAnalyzer

__all__ = ["DRTAnalyzer"]

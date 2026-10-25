"""``run_model``: execute a written simulation via the MODFLOW 6 binary.

Requires a MODFLOW 6 executable (>= 6.5 for GWE) resolvable as ``$MF6_BIN`` or
``mf6`` on PATH. Returns a private ``_RawOutput`` (success flag + output paths).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import flopy

from ATESTool.model.builder import _ModelPaths, resolve_mf6_exe


@dataclass
class _RawOutput:
    """Raw result of a MODFLOW 6 run: success flag and output file paths."""

    success: bool
    head_file: Path
    temperature_file: Path
    flow_budget_file: Path
    energy_budget_file: Path


def run_model(model_paths: _ModelPaths) -> _RawOutput:
    """Run the simulation in its workspace; raise on solver failure."""
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=str(model_paths.workspace), exe_name=resolve_mf6_exe(), verbosity_level=0
    )
    success, buff = sim.run_simulation(silent=True)
    if not success:
        raise RuntimeError("MODFLOW 6 run failed:\n" + "\n".join(buff))
    return _RawOutput(
        success=success,
        head_file=model_paths.head_file,
        temperature_file=model_paths.temperature_file,
        flow_budget_file=model_paths.flow_budget_file,
        energy_budget_file=model_paths.energy_budget_file,
    )

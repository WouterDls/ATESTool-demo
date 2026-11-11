"""``run_model``: execute a written simulation via the MODFLOW 6 binary.

Requires a MODFLOW 6 executable with current GWE input support, resolvable as ``$MF6_BIN`` or
``mf6`` on PATH. Returns a private ``_RawOutput`` (success flag + output paths).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import monotonic

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


_STEP = re.compile(r"Solving:\s+Stress period:\s*(\d+)\s+Time step:\s*(\d+)")


def _step_fraction(completed: int, count: int, multiplier: float) -> float:
    """Fraction of a period elapsed before the announced step starts."""
    if completed == 0:
        return 0.0
    if multiplier == 1.0:
        return completed / count
    log_multiplier = math.log(multiplier)
    exponent = count * log_multiplier
    if exponent > 700:
        return math.exp((completed - count) * log_multiplier)
    return math.expm1(completed * log_multiplier) / math.expm1(exponent)


def _run_with_progress(
    executable: str, workspace: Path, periods: list[tuple[float, int, float]], adaptive: bool
) -> tuple[bool, list[str]]:
    ends = [0.0]
    for length, _, _ in periods:
        ends.append(ends[-1] + length)
    started = monotonic()
    interactive = sys.stdout.isatty()
    previous: tuple[int, int] | None = None

    def display(percent: int, *, final: bool = False) -> None:
        nonlocal previous
        elapsed = int(monotonic() - started)
        if not final and (not interactive or previous == (percent, elapsed)):
            return
        previous = (percent, elapsed)
        filled = percent // 5
        bar = "█" * filled + "░" * (20 - filled)
        minutes, seconds = divmod(elapsed, 60)
        prefix = "\r" if interactive else ""
        print(
            f"{prefix}MODFLOW 6  [{bar}] {percent:3d}%  |  "
            f"{minutes:02d}:{seconds:02d} elapsed",
            end="\n" if final else "", flush=True,
        )

    output: list[str] = []
    percent = 0
    success = False
    normal_termination = False
    display(percent)
    try:
        with subprocess.Popen(
            [executable], cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace",
        ) as process:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    output.append(line.rstrip())
                    normal_termination |= "normal termination" in line.lower()
                    match = _STEP.search(line)
                    if match:
                        period, step = map(int, match.groups())
                        length, count, multiplier = periods[period - 1]
                        # ATS output announces step indices without the actual step length.
                        elapsed_model = ends[period - 1]
                        if not adaptive:
                            elapsed_model += length * _step_fraction(step - 1, count, multiplier)
                        percent = max(percent, min(99, int(100 * elapsed_model / ends[-1])))
                    display(percent)
                success = process.wait() == 0 and normal_termination
            except BaseException:
                if process.poll() is None:
                    process.kill()
                raise
    finally:
        if not success and interactive and previous is not None:
            print("\r" + " " * 65 + "\r", end="", flush=True)
    if success:
        display(100, final=True)
    return success, output


def run_model(model_paths: _ModelPaths, *, progress: bool = False) -> _RawOutput:
    """Run the simulation in its workspace; raise on solver failure."""
    executable = resolve_mf6_exe()
    version_output = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=True
    )
    version = re.search(r"\bmf6:\s*(\d+)\.(\d+)\.(\d+)", version_output.stdout)
    if version and tuple(map(int, version.groups())) <= (6, 5, 0):
        raise RuntimeError(
            f"MODFLOW {version.group(0).split()[-1]} is unsupported by ATESTool. "
            "Use MODFLOW 6.6.1, which was verified with the supplied examples."
        )
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=str(model_paths.workspace), exe_name=executable, verbosity_level=0
    )
    if progress:
        resolved = shutil.which(executable)
        assert resolved is not None  # The version probe already ran this executable.
        periods = [
            (float(length), int(count), float(multiplier))
            for length, count, multiplier in sim.tdis.perioddata.array
        ]
        success, buff = _run_with_progress(
            str(Path(resolved).resolve()), model_paths.workspace, periods,
            adaptive=getattr(sim.tdis.ats, "perioddata", None) is not None,
        )
    else:
        success, buff = sim.run_simulation(silent=True, report=True)
    if not success:
        raise RuntimeError("MODFLOW 6 run failed:\n" + "\n".join(buff))
    return _RawOutput(
        success=success,
        head_file=model_paths.head_file,
        temperature_file=model_paths.temperature_file,
        flow_budget_file=model_paths.flow_budget_file,
        energy_budget_file=model_paths.energy_budget_file,
    )

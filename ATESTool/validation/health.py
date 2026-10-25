"""Per-run numerical health of a MODFLOW 6 run.

This promotes the GWE-only ``_listing_health`` that used to live in
:mod:`ATESTool.validation.temporal_convergence` to a shared, both-listing reader used by
deterministic runs and convergence studies. Reading is done straight off the workspace's
listing files after a run, so it adds no solver cost.

A :class:`RunHealth` captures, for one completed run:

* ``converged``: the simulation reached normal termination and no time step failed;
* ``failed_steps``: how many outer time steps failed to meet solver convergence
  (GWF + GWE), which under ATS may still be retried;
* ``gwf_discrepancy`` / ``gwe_discrepancy``: the largest ``|percent discrepancy|`` in
  each flow/energy listing (a *conservation* gauge, not an accuracy one);
* ``runtime_seconds``: wall-clock for the run, measured by the caller (the listing's
  own elapsed-time line is not always present at parse time).

A small discrepancy proves the balance closed, not that the answer is right; accuracy is
what the convergence studies establish, this module only flags *broken* runs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

#: Largest acceptable listing mass-balance discrepancy, in percent. Applies to both the
#: GWF (flow) and GWE (energy) listings. Historically defined in ``temporal_convergence``;
#: it now lives here and is re-exported there so that module keeps working unchanged.
DISCREPANCY_TOL = 1.0

#: Standard MODFLOW 6 listing file names written by :mod:`ATESTool.model.builder`
#: (``GWF_NAME``/``GWE_NAME``) and by MF6 itself (the simulation listing).
GWF_LISTING = "gwf.lst"
GWE_LISTING = "gwe.lst"
SIM_LISTING = "mfsim.lst"

_DISCREPANCY = re.compile(r"PERCENT DISCREPANCY\s*=\s*(-?\d+\.\d+)")
_FAILED_STEP = re.compile(r"FAILED TO MEET SOLVER CONVERGENCE", re.IGNORECASE)
_NORMAL_TERMINATION = re.compile(r"Normal termination of simulation", re.IGNORECASE)
_MF6_VERSION = re.compile(r"VERSION\s+([\d.]+)")


def _read_text(path: Path) -> str | None:
    """The listing's text, or ``None`` when the file is absent (e.g. a synthetic run)."""
    if not path.exists():
        return None
    return path.read_text(errors="ignore")


def _max_discrepancy(text: str | None) -> float:
    """Largest ``|percent discrepancy|`` in a listing, or NaN when none is recorded.

    NaN means "no discrepancy line found" (missing listing, or a run so short none was
    written); callers treat that as *not a violation*, matching the pre-existing
    time-step convergence behaviour.
    """
    if text is None:
        return math.nan
    values = [abs(float(x)) for x in _DISCREPANCY.findall(text)]
    return max(values) if values else math.nan


def _failed_steps(text: str | None) -> int:
    """Count of ``FAILED TO MEET SOLVER CONVERGENCE`` occurrences in a listing."""
    if text is None:
        return 0
    return len(_FAILED_STEP.findall(text))


def discrepancy_ok(value: float) -> bool:
    """Whether a listing discrepancy is within tolerance. NaN (no line) passes."""
    return math.isnan(value) or abs(value) <= DISCREPANCY_TOL


@dataclass(frozen=True)
class RunHealth:
    """Numerical-health gauges for one completed MODFLOW 6 run."""

    converged: bool
    failed_steps: int
    gwf_discrepancy: float
    gwe_discrepancy: float
    runtime_seconds: float

    @property
    def ok(self) -> bool:
        """The run converged, failed no step, and both listings closed within tolerance."""
        return (
            self.converged
            and self.failed_steps == 0
            and discrepancy_ok(self.gwf_discrepancy)
            and discrepancy_ok(self.gwe_discrepancy)
        )

    def as_columns(self) -> dict[str, float | int | bool]:
        """The health gauges as flat serialisable columns."""
        return {
            "converged": self.converged,
            "failed_steps": self.failed_steps,
            "gwf_discrepancy": self.gwf_discrepancy,
            "gwe_discrepancy": self.gwe_discrepancy,
            "runtime_seconds": self.runtime_seconds,
        }


def read_run_health(workspace: Path, runtime_seconds: float = math.nan) -> RunHealth:
    """Read the flow/energy listings in ``workspace`` into a :class:`RunHealth`.

    ``runtime_seconds`` is supplied by the caller (it times the run); everything else comes
    from the listing files. A missing listing is treated leniently: no discrepancy line
    means NaN (passes), no failed-step line means zero, so a synthetic run directory with
    no MF6 output still yields a well-formed, healthy record rather than raising.
    ``converged`` is true when the simulation listing reports normal termination and no time
    step failed; when the simulation listing is absent (synthetic run), it falls back to
    "no failed steps".
    """
    gwf = _read_text(workspace / GWF_LISTING)
    gwe = _read_text(workspace / GWE_LISTING)
    sim = _read_text(workspace / SIM_LISTING)

    failed = _failed_steps(gwf) + _failed_steps(gwe) + _failed_steps(sim)
    if sim is not None:
        converged = bool(_NORMAL_TERMINATION.search(sim)) and failed == 0
    else:
        converged = failed == 0

    return RunHealth(
        converged=converged,
        failed_steps=failed,
        gwf_discrepancy=_max_discrepancy(gwf),
        gwe_discrepancy=_max_discrepancy(gwe),
        runtime_seconds=float(runtime_seconds),
    )


def read_mf6_version(workspace: Path) -> str | None:
    """The MODFLOW 6 version string (e.g. ``"6.7.0"``) from the simulation listing header.

    ``None`` when the listing is absent (a synthetic run, or one that failed before MF6
    wrote anything); a reproducibility record reads this once from the first run that has
    a listing rather than requiring every run's output to be kept.
    """
    text = _read_text(workspace / SIM_LISTING)
    if text is None:
        return None
    match = _MF6_VERSION.search(text)
    return match.group(1) if match else None


def gwe_listing_health(workspace: Path) -> tuple[float, int]:
    """The GWE-only ``(max |percent discrepancy|, failed_steps)`` pair.

    A backwards-compatible shim for :mod:`ATESTool.validation.temporal_convergence`, which
    read only the energy listing before this module existed. New code should prefer
    :func:`read_run_health`.
    """
    gwe = _read_text(workspace / GWE_LISTING)
    return (_max_discrepancy(gwe), _failed_steps(gwe))

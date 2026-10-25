"""Time-step convergence study: the temporal twin of :mod:`ATESTool.validation.convergence`.

Where ``run_convergence`` refines the *mesh* (cells), this refines the *time step* by
tightening the Courant target, holding the mesh fixed (temporal refinement does not change
geometry: the mesh is generated once and reused across levels). It reports:

* per-level outputs (recovery efficiency, peak/final temperature, drawdown, breakthrough);
* the **relative change** between the two finest levels and a ``converged`` verdict
  (meshing primer §6.1, same tolerances as the grid study);
* the **observed order of accuracy** and a **Richardson Δt→0 extrapolation** per metric: a
  constant-ratio refinement lets us confirm the scheme converges at its theoretical rate
  (backward Euler ⇒ first order in time) and bound the residual discretisation error;
* **numerical health** per level: the GWE mass-balance discrepancy and any failed time
  steps (meshing primer §7), the "avoids numerical issues" gauge. Note: a small discrepancy proves
  *conservation*, not *accuracy*; the convergence verdict is what proves accuracy.

Needs the MODFLOW 6 binary (runs the full pipeline once per Courant level).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ATESTool.config import Config, TimeSteppingConfig
from ATESTool.mesh import generate_mesh
from ATESTool.model import build_model, run_model
from ATESTool.validation.convergence import (
    HEAD_TOL,
    TRANSPORT_TOL,
    _cleanup_binary_outputs,
    _guard_dense_output_size,
    _metric_converged,
    _rel,
)
from ATESTool.validation.health import DISCREPANCY_TOL, gwe_listing_health

#: Coarse -> fine Courant targets (constant ratio 2; ≥3 levels needed for an observed order).
DEFAULT_TARGETS: tuple[float, ...] = (2.0, 1.0, 0.5)

#: Per-metric (relative tol, absolute floor): mirrors the grid study, plus recovery
#: efficiency (the headline ATES metric, in percentage points).
_METRIC_TOL: dict[str, tuple[float, float]] = {
    "max_drawdown": (HEAD_TOL, 0.01),
    "final_head": (HEAD_TOL, 0.01),
    "peak_temperature": (TRANSPORT_TOL, 0.05),
    "final_temperature": (TRANSPORT_TOL, 0.05),
    "breakthrough_time": (TRANSPORT_TOL, 5.0),
    "mean_recovery_efficiency": (TRANSPORT_TOL, 0.5),
}

@dataclass
class TemporalLevel:
    """One Courant level's time step, key outputs, and numerical-health gauges."""

    target_courant: float
    n_timesteps: int
    max_discrepancy: float  # max |percent discrepancy| over all steps (GWE listing)
    failed_steps: int
    peak_temperature: float
    max_drawdown: float
    breakthrough_time: float
    final_temperature: float
    final_head: float
    mean_recovery_efficiency: float
    deleted_binary_bytes: int = 0

    def metric(self, name: str) -> float:
        return float(getattr(self, name))


@dataclass
class TemporalConvergenceResult:
    """Outcome of a time-step convergence study (levels ordered coarse -> fine)."""

    levels: list[TemporalLevel]
    rel_change: dict[str, float]          # between the two finest levels
    observed_order: dict[str, float]      # per metric, from the three finest levels
    richardson_limit: dict[str, float]    # Δt->0 extrapolation per metric
    converged: bool

    @property
    def healthy(self) -> bool:
        """No level exceeded the mass-balance tolerance or failed a time step."""
        return all(
            lv.failed_steps == 0
            and (math.isnan(lv.max_discrepancy) or abs(lv.max_discrepancy) <= DISCREPANCY_TOL)
            for lv in self.levels
        )

    def summary(self) -> str:
        lines = ["time-step convergence (coarse -> fine):"]
        for lv in self.levels:
            lines.append(
                f"  Co<={lv.target_courant:<5g} steps={lv.n_timesteps:6d}  "
                f"meanEff={lv.mean_recovery_efficiency:.4f}  peakT={lv.peak_temperature:.4f}  "
                f"drawdown={lv.max_drawdown:.4f}  wellT={lv.final_temperature:.4f}  "
                f"discrepancy={lv.max_discrepancy:.2e}%  failed={lv.failed_steps}"
            )
        lines.append("  relative change (finest two levels):")
        for name, val in self.rel_change.items():
            lines.append(f"    {name:<26} {val:.2%}" if math.isfinite(val) else f"    {name:<26} n/a")
        lines.append("  observed order / Richardson Δt->0 limit:")
        for name in self.observed_order:
            p, lim = self.observed_order[name], self.richardson_limit[name]
            p_s = f"{p:.2f}" if math.isfinite(p) else "n/a"
            lim_s = f"{lim:.4f}" if math.isfinite(lim) else "n/a"
            lines.append(f"    {name:<26} p={p_s:>5}  limit={lim_s}")
        lines.append(f"  => converged: {self.converged}   healthy: {self.healthy}")
        return "\n".join(lines)


def _level_config(config: Config, target: float) -> Config:
    """A copy of ``config`` pinned to courant mode at ``target`` with uniform steps.

    Uniform steps (``tsmult=1.0``) keep the refinement at a constant ratio so the observed
    order of accuracy is clean.
    """
    variant = copy.deepcopy(config)
    variant.model.simulation.time_stepping = TimeSteppingConfig(
        mode="courant", target_courant=target, tsmult=1.0, ats_dtadj=2.0, ats_dtfailadj=5.0
    )
    return variant


def _observed_order(seq: Sequence[float], ratio: float) -> float:
    """Observed order ``p`` from three values (coarse->fine, constant step ratio ``ratio``)."""
    if len(seq) < 3 or ratio <= 1.0:
        return float("nan")
    d01, d12 = seq[-3] - seq[-2], seq[-2] - seq[-1]
    if d12 == 0.0 or d01 / d12 <= 0.0:  # non-monotone or flat -> order undefined
        return float("nan")
    return math.log(abs(d01 / d12)) / math.log(ratio)


def _richardson(f_fine: float, f_med: float, ratio: float, p: float) -> float:
    """Richardson Δt->0 extrapolation from the two finest levels."""
    if not math.isfinite(p):
        return float("nan")
    return f_fine + (f_fine - f_med) / (ratio**p - 1.0)


def _constant_ratio(targets: Sequence[float]) -> float:
    """The (assumed constant) refinement ratio between consecutive coarse->fine targets."""
    ordered = sorted(targets, reverse=True)
    return ordered[-2] / ordered[-1] if len(ordered) >= 2 and ordered[-1] > 0 else float("nan")


def run_temporal_convergence(
    config: Config,
    targets: Sequence[float] = DEFAULT_TARGETS,
    workspace: Path | str | None = None,
    *,
    compact_outputs: bool = False,
    max_level_bytes: int | None = None,
) -> TemporalConvergenceResult:
    """Run the pipeline at each Courant ``target`` (mesh fixed) and assess convergence."""
    from ATESTool.analysis import analyse

    if len(targets) < 2:
        raise ValueError("temporal convergence needs at least two Courant targets")
    ordered = sorted(targets, reverse=True)  # coarse (large Co) -> fine (small Co)
    base = Path(workspace) if workspace is not None else Path.cwd() / "temporal_convergence"

    mesh = generate_mesh(config)  # geometry is fixed across time-step levels
    levels: list[TemporalLevel] = []
    for target in ordered:
        run_ws = base / f"co_{target:g}"
        run_ws.mkdir(parents=True, exist_ok=True)
        cfg = _level_config(config, target)
        paths = build_model(cfg, mesh, run_ws, budget_save=None if compact_outputs else "ALL")
        if compact_outputs:
            _guard_dense_output_size(
                paths,
                n_cells=mesh.n_cells,
                max_bytes=max_level_bytes,
                level_label=f"Courant level {target:g}",
            )
        level: TemporalLevel | None = None
        deleted_binary_bytes = 0
        try:
            sim = analyse(paths, run_model(paths), cfg, mesh)
            stats = sim.stats
            ts = sim.timeseries
            discrepancy, fails = gwe_listing_health(run_ws)
            level = TemporalLevel(
                target_courant=target,
                n_timesteps=len(sim.times),
                max_discrepancy=discrepancy,
                failed_steps=fails,
                peak_temperature=stats["peak_temperature"],
                max_drawdown=stats["max_drawdown"],
                breakthrough_time=stats["breakthrough_time"],
                final_temperature=float(ts["temperature"].iloc[-1]),
                final_head=float(ts["head"].iloc[-1]),
                mean_recovery_efficiency=stats["mean_recovery_efficiency"],
            )
        finally:
            if compact_outputs:
                deleted_binary_bytes = _cleanup_binary_outputs(
                    paths,
                    max_bytes=max_level_bytes,
                    level_label=f"Courant level {target:g}",
                )
        assert level is not None
        level.deleted_binary_bytes = deleted_binary_bytes
        levels.append(level)

    coarser, finer = levels[-2], levels[-1]
    rel_change = {name: _rel(coarser.metric(name), finer.metric(name)) for name in _METRIC_TOL}
    converged = all(
        _metric_converged(coarser.metric(name), finer.metric(name), rel_tol, abs_floor)
        for name, (rel_tol, abs_floor) in _METRIC_TOL.items()
    )

    ratio = _constant_ratio(ordered)
    observed_order: dict[str, float] = {}
    richardson_limit: dict[str, float] = {}
    for name in _METRIC_TOL:
        seq = [lv.metric(name) for lv in levels]
        p = _observed_order(seq, ratio)
        observed_order[name] = p
        richardson_limit[name] = _richardson(seq[-1], seq[-2], ratio, p)

    return TemporalConvergenceResult(
        levels=levels,
        rel_change=rel_change,
        observed_order=observed_order,
        richardson_limit=richardson_limit,
        converged=converged,
    )

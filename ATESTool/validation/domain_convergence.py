"""Domain-distance sensitivity at fixed regional gradient and target mesh spacing.

Each level pairs the pumped model with an otherwise identical no-well reference.
The hydraulic gate is their maximum head difference at ONE fixed off-well coordinate;
raw extraction-cell drawdown remains a diagnostic. A prescribed regional head plane
is evaluated at boundary cell centres, preserving its datum and gradient as the
boundary moves. Target cell sizes and time steps are fixed; actual mesh topology can
still change. This costs more than coarsening the far field as the domain grows.

No recharge is added to force a pass. In a linear confined aquifer prescribed recharge
cancels from the pumped-minus-unpumped flow equation: it does not guarantee a finite
pumping influence radius. Boundary independence is tested over the stated finite run
horizon, not asserted for infinite-time steady pumping. See docs/validation_decisions.md.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ATESTool.config import Config, TimeSteppingConfig
from ATESTool.mesh import Domain, VertexGrid, generate_mesh
from ATESTool.mesh.generator import _background_cell
from ATESTool.model import build_model, run_model
from ATESTool.validation.convergence import (
    HEAD_TOL, TRANSPORT_TOL, _metric_converged, _rel, _observation_point,
    _observation_head, _cleanup_binary_outputs, _guard_dense_output_size,
)
from ATESTool.validation.health import DISCREPANCY_TOL, read_run_health

if TYPE_CHECKING:
    from ATESTool.analysis import SimulationResult

#: Small -> large domain sizes [m] (square domains). Ordered so "fine" = largest domain.
DEFAULT_SIZES: tuple[float, ...] = (5000.0, 10000.0, 20000.0, 40000.0)

#: Per-metric (relative tol, absolute floor): the drawdown and thermal headline metrics.
_METRIC_TOL: dict[str, tuple[float, float]] = {
    "pumping_drawdown": (HEAD_TOL, 0.01),
    "peak_temperature": (TRANSPORT_TOL, 0.05),
    "mean_recovery_efficiency": (TRANSPORT_TOL, 0.5),
}


@dataclass
class DomainLevel:
    """One domain size's mesh size, key outputs, and numerical-health gauges."""

    domain_size: float
    n_cells: int
    max_discrepancy: float  # max |percent discrepancy| over all steps (GWE listing)
    failed_steps: int
    peak_temperature: float
    max_drawdown: float
    mean_recovery_efficiency: float
    #: Max ``|T - ambient|`` over the boundary cells at the final time [degC]. ~0 confirms the
    #: thermal plume never reaches the boundary; a rising value means the domain is too small.
    boundary_thermal_influence: float

    pumping_drawdown: float = float("nan")
    reference_healthy: bool = True
    pumped_healthy: bool = True
    reference_max_discrepancy: float = float("nan")
    reference_failed_steps: int = 0
    background_cell_size: float = float("nan")
    deleted_binary_bytes: int = 0

    def metric(self, name: str) -> float:
        return float(getattr(self, name))


@dataclass
class DomainConvergenceResult:
    """Outcome of a domain-size convergence study (levels ordered small -> large)."""

    levels: list[DomainLevel]
    rel_change: dict[str, float]  # between the two largest domains
    converged: bool
    observation_point: tuple[float, float] = (float("nan"), float("nan"))
    boundary_mode: str = "linear"

    @property
    def healthy(self) -> bool:
        """No level exceeded the mass-balance tolerance or failed a time step."""
        return all(
            lv.reference_healthy and lv.pumped_healthy and lv.failed_steps == 0
            and (math.isnan(lv.max_discrepancy) or abs(lv.max_discrepancy) <= DISCREPANCY_TOL)
            for lv in self.levels
        )

    def summary(self) -> str:
        lines = ["domain-size convergence (small -> large):"]
        for lv in self.levels:
            lines.append(
                f"  L={lv.domain_size:<8g} cells={lv.n_cells:6d}  "
                f"meanEff={lv.mean_recovery_efficiency:.4f}  peakT={lv.peak_temperature:.4f}  "
                f"pumpingDrawdown={lv.pumping_drawdown:.4f}  "
                f"wellCellDrawdown(diagnostic)={lv.max_drawdown:.4f}  bndT={lv.boundary_thermal_influence:.2e}  "
                f"discrepancy={lv.max_discrepancy:.2e}%  failed={lv.failed_steps}"
            )
        lines.append("  relative change (two largest domains):")
        for name, val in self.rel_change.items():
            lines.append(f"    {name:<26} {val:.2%}" if math.isfinite(val) else f"    {name:<26} n/a")
        lines.append(f"  => converged: {self.converged}   healthy: {self.healthy}")
        return "\n".join(lines)


def _level_config(config: Config, size: float) -> Config:
    """Grow a regional-domain experiment without changing gradient or target cell size.

    Legacy constant-side heads are interpreted as edge-midpoint regional heads for
    this study only. Ordinary runs retain their configured boundary mode. Incompatible
    midpoint datums are rejected rather than silently fitted to an arbitrary plane.
    """
    if not math.isfinite(size) or size <= 0:
        raise ValueError("domain sizes must be finite and positive")
    config.model.chd_heads.validate_linear()
    variant = copy.deepcopy(config)
    variant.mesh.background_max_cell_size = _background_cell(config, 1.0)
    heads = config.model.chd_heads
    gx = (heads.east - heads.west) / float(config.mesh.width)
    gy = (heads.north - heads.south) / float(config.mesh.height)
    h0 = heads.reference_head
    variant.model.chd_heads.north = h0 + gy * size / 2
    variant.model.chd_heads.south = h0 - gy * size / 2
    variant.model.chd_heads.east = h0 + gx * size / 2
    variant.model.chd_heads.west = h0 - gx * size / 2
    variant.model.chd_heads.mode = "linear"
    variant.mesh.width = size
    variant.mesh.height = size
    variant.model.simulation.time_stepping = TimeSteppingConfig(
        mode="fixed", target_courant=1.0, tsmult=1.0, ats_dtadj=2.0, ats_dtfailadj=5.0
    )
    return variant


def _boundary_thermal_influence(
    sim: SimulationResult, mesh: VertexGrid, domain: Domain
) -> float:
    """Max |final temperature - ambient| over the boundary cells [degC].

    The boundary cells are :meth:`Domain.boundary_cell_indices`: the same definition the
    CHD builder and :mod:`ATESTool.analysis.boundary` use. This is the thermal half of the
    boundary-influence pair; see :func:`ATESTool.analysis.boundary.boundary_influence`.
    """
    boundary = domain.boundary_cell_indices(mesh)
    if boundary.size == 0:
        return float("nan")
    final_temp = sim.snapshot(float(sim.times[-1]), "temperature")
    return float(np.max(np.abs(final_temp[boundary] - sim.initial_temperature)))


def run_domain_convergence(
    config: Config,
    sizes: Sequence[float] = DEFAULT_SIZES,
    workspace: Path | str | None = None,
    *,
    compact_outputs: bool = False,
    max_level_bytes: int | None = None,
) -> DomainConvergenceResult:
    """Compare paired pumped/reference runs at each size, with fixed regional gradients.

    Legacy side heads are interpreted as regional edge-midpoint heads in this study;
    ordinary model boundary behaviour is unchanged. Background target cell size is
    held at the original config's value. ``compact_outputs`` removes dense outputs
    from both runs after extracting metrics; ``max_level_bytes`` bounds each compact
    run's head/temperature output, not the combined disk usage of the whole study.
    """
    from ATESTool.analysis import analyse

    if len(sizes) < 2:
        raise ValueError("domain convergence needs at least two domain sizes")
    ordered = sorted(sizes)  # small -> large (the boundary moves outward)
    if len(set(ordered)) != len(ordered):
        raise ValueError("domain sizes must be distinct")
    base = Path(workspace) if workspace is not None else Path.cwd() / "domain_convergence"

    levels: list[DomainLevel] = []
    observation_point: tuple[float, float] | None = None
    for size in ordered:
        run_ws = base / f"size_{size:g}"
        run_ws.mkdir(parents=True, exist_ok=True)
        cfg = _level_config(config, size)
        mesh = generate_mesh(cfg)  # geometry changes with the domain
        if observation_point is None:
            observation_point = _observation_point(cfg, mesh)
        # Validate the point before paying for either solve.
        _observation_head(cfg, mesh, np.zeros((1, mesh.n_cells)), observation_point)
        reference_cfg = copy.deepcopy(cfg)
        reference_cfg.model.extraction_well.flow_rate = 0.0
        for doublet in reference_cfg.model.doublets:
            doublet.flow_rate = 0.0
        # Reuse the SAME mesh, including auto thermal rings generated from pumped rates.
        # Run sequentially to avoid retaining both dense simulations in memory.
        head_series = []
        health = []
        deleted_bytes = 0
        for run_cfg, ws in ((reference_cfg, run_ws / "reference"), (cfg, run_ws)):
            paths = build_model(run_cfg, mesh, ws, budget_save=None if compact_outputs else "ALL")
            if compact_outputs:
                _guard_dense_output_size(paths, n_cells=mesh.n_cells,
                                         max_bytes=max_level_bytes, level_label=f"domain {size:g}")
            try:
                sim = analyse(paths, run_model(paths), run_cfg, mesh)
                head_series.append(_observation_head(cfg, mesh, sim.head, observation_point))
                health.append(read_run_health(ws))
                if run_cfg is reference_cfg:
                    reference_times = sim.times.copy()
                else:
                    if not np.array_equal(reference_times, sim.times):
                        raise ValueError("pumped and reference output times must match")
                    stats = sim.stats
                    boundary_influence = _boundary_thermal_influence(sim, mesh, Domain.from_config(cfg))
                del sim
            finally:
                if compact_outputs:
                    deleted_bytes += _cleanup_binary_outputs(
                        paths, max_bytes=max_level_bytes, level_label=f"domain {size:g}")
        ref_health, pumped_health = health
        levels.append(
            DomainLevel(
                domain_size=size,
                n_cells=mesh.n_cells,
                max_discrepancy=pumped_health.gwe_discrepancy,
                failed_steps=pumped_health.failed_steps,
                peak_temperature=stats["peak_temperature"],
                max_drawdown=stats["max_drawdown"],
                mean_recovery_efficiency=stats["mean_recovery_efficiency"],
                boundary_thermal_influence=boundary_influence,
                pumping_drawdown=float(np.max(head_series[0] - head_series[1])),
                reference_healthy=ref_health.ok,
                pumped_healthy=pumped_health.ok,
                reference_max_discrepancy=max(ref_health.gwf_discrepancy, ref_health.gwe_discrepancy),
                reference_failed_steps=ref_health.failed_steps,
                background_cell_size=_background_cell(cfg, 1.0),
                deleted_binary_bytes=deleted_bytes,
            )
        )

    smaller, larger = levels[-2], levels[-1]
    rel_change = {name: _rel(smaller.metric(name), larger.metric(name)) for name in _METRIC_TOL}
    converged = all(
        _metric_converged(smaller.metric(name), larger.metric(name), rel_tol, abs_floor)
        for name, (rel_tol, abs_floor) in _METRIC_TOL.items()
    )
    assert observation_point is not None
    return DomainConvergenceResult(levels=levels, rel_change=rel_change, converged=converged,
                                   observation_point=observation_point)

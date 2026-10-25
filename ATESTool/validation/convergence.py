"""Grid-convergence study (meshing primer §6.1, "Grid and temporal convergence").

``run_convergence`` rebuilds and reruns the model at progressively finer meshes (via
``generate_mesh``'s ``refinement_factor``) and reports how the key outputs (an off-well
observation head, thermal breakthrough, peak temperature) change between resolutions. If
they stop changing within tolerance, the two finest levels are stable for those metrics.
This does not establish accuracy everywhere or certify the coarsest baseline mesh.

This is the accuracy check the per-mesh quality gate cannot provide: a mesh can be
geometrically valid and mass-conservative yet still under-resolve the thermal front. Needs
the MODFLOW 6 binary (it runs the full pipeline once per refinement level).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ATESTool.config import Config, TimeSteppingConfig
from ATESTool.mesh import Domain, VertexGrid, generate_mesh
from ATESTool.model import build_model, run_model
from ATESTool.model.builder import _ModelPaths

#: Coarse -> fine refinement factors (each halving scales cell sizes by 1/2 ≈ 4x cells).
DEFAULT_FACTORS: tuple[float, ...] = (1.0, 0.5)
#: Relative-change tolerances between the two finest levels (meshing primer §6.1): heads
#: 1-5%, transport 5-10%.
HEAD_TOL = 0.05
TRANSPORT_TOL = 0.10

#: Per-metric (relative tol, absolute floor). The absolute floor stops a near-zero
#: quantity (e.g. ~0.3 mm of drawdown, or no thermal signal) from being flagged as
#: "not converged" purely from relative-change noise. Units: m for heads, degC for
#: temperatures, days for breakthrough.
_METRIC_TOL: dict[str, tuple[float, float]] = {
    "observation_drawdown": (HEAD_TOL, 0.01),
    "final_head": (HEAD_TOL, 0.01),
    "peak_temperature": (TRANSPORT_TOL, 0.05),
    "final_temperature": (TRANSPORT_TOL, 0.05),
    "breakthrough_time": (TRANSPORT_TOL, 5.0),
}


def _cleanup_binary_outputs(
    paths: _ModelPaths, *, max_bytes: int | None, level_label: str
) -> int:
    """Delete large solver binaries and return their combined logical size.

    Cleanup happens before enforcing ``max_bytes`` so a failed limit check never strands
    the oversized files that the compact mode exists to prevent.
    """
    binaries = (
        paths.head_file,
        paths.flow_budget_file,
        paths.temperature_file,
        paths.energy_budget_file,
    )
    total = sum(path.stat().st_size for path in binaries if path.exists())
    for path in binaries:
        path.unlink(missing_ok=True)
    if max_bytes is not None and total > max_bytes:
        raise RuntimeError(
            f"{level_label} binary output is {total / 1024**3:.1f} GiB; "
            f"limit is {max_bytes / 1024**3:.1f} GiB"
        )
    return total


def _guard_dense_output_size(
    paths: _ModelPaths, *, n_cells: int, max_bytes: int | None, level_label: str
) -> int:
    """Estimate the complete head+temperature output and reject oversized levels early.

    Compact mode disables cell-budget output, leaving two double-precision DISV field
    records per saved time. The 25% allowance covers binary record headers generously.
    """
    tdis_files = list(paths.workspace.glob("*.tdis"))
    if len(tdis_files) != 1:
        raise RuntimeError(f"expected one TDIS file in {paths.workspace}, found {len(tdis_files)}")
    in_period_data = False
    n_steps = 0
    for raw_line in tdis_files[0].read_text().splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered == "begin perioddata":
            in_period_data = True
            continue
        if lowered == "end perioddata":
            break
        if in_period_data and line and not line.startswith("#"):
            n_steps += int(line.split()[1])
    bytes_per_field_record = paths.n_layers * n_cells * 8 + 128
    estimate = int(2 * n_steps * bytes_per_field_record * 1.25)
    if max_bytes is not None and estimate > max_bytes:
        raise RuntimeError(
            f"{level_label} estimated dense output is {estimate / 1024**3:.1f} GiB; "
            f"limit is {max_bytes / 1024**3:.1f} GiB"
        )
    return estimate


@dataclass
class ConvergenceLevel:
    """One refinement level's mesh size and key outputs."""

    refinement_factor: float
    n_cells: int
    peak_temperature: float
    max_drawdown: float
    breakthrough_time: float
    final_temperature: float
    final_head: float
    deleted_binary_bytes: int = 0
    #: Maximum IC-referenced drawdown at the fixed off-well observation coordinate.
    observation_drawdown: float = float("nan")

    def metric(self, name: str) -> float:
        return float(getattr(self, name))


#: Observation-cell candidates must sit within this fraction of the domain half-width of
#: the centre, close enough to be in the well-resolved interior, not out at the boundary.
_OBSERVATION_RADIUS_FRACTION = 0.4


def _all_well_xy(config: Config) -> np.ndarray:
    """Every well coordinate (doublet warm+cold plus the extraction well) as ``(n, 2)``."""
    wells = [d.warm_well for d in config.model.doublets]
    wells += [d.cold_well for d in config.model.doublets]
    wells.append(config.model.extraction_well.coordinates)
    return np.array([[float(x), float(y)] for x, y in wells], dtype=float)


def _observation_point(config: Config, mesh: VertexGrid) -> tuple[float, float]:
    """The interior mesh cell centre farthest from every well, for a resolution-only head check.

    Head sampled at a pumping well follows a log singularity in cell size (Thiem drawdown ~
    ln r_cell), so refining the mesh changes it by construction and the ``final_head`` gate
    could never pass. So we pick, among the cells within
    ``_OBSERVATION_RADIUS_FRACTION`` of the domain half-width of the centre, the one whose
    centre has the largest minimum distance to any well. That samples genuine mesh-resolution
    error rather than the near-well singularity, and, unlike the old fixed domain centre,
    stays valid now the extraction well sits *at* the centre. It is chosen once, on the
    baseline mesh, and the same coordinate is reused at every refinement level.

    When ``analysis.observation_points`` has at least one entry, the first configured
    point is used verbatim instead of the heuristic: a user-chosen monitoring point is a
    more meaningful convergence probe than an automatically farthest-from-any-well cell,
    and it makes the study check the location the user actually cares about.
    """
    if config.analysis.observation_points:
        first = config.analysis.observation_points[0]
        return (float(first.x), float(first.y))

    domain = Domain.from_config(config)
    cx, cy = domain.center
    radius = _OBSERVATION_RADIUS_FRACTION * 0.5 * min(domain.width, domain.height)

    centroids = mesh.centroids
    from_center = np.hypot(centroids[:, 0] - cx, centroids[:, 1] - cy)
    candidates = np.where(from_center <= radius)[0]
    if candidates.size == 0:  # degenerate (tiny domain); fall back to the whole mesh
        candidates = np.arange(mesh.n_cells)

    wells = _all_well_xy(config)
    min_well_dist = np.min(
        np.linalg.norm(centroids[candidates][:, None, :] - wells[None, :, :], axis=2), axis=1
    )
    best = int(candidates[int(np.argmax(min_well_dist))])
    return (float(centroids[best, 0]), float(centroids[best, 1]))


def _observation_head(
    config: Config, mesh: VertexGrid, head: np.ndarray, point: tuple[float, float]
) -> np.ndarray:
    """Interpolate cell-centre heads to one physical point, avoiding well-cell samples.

    Nearest-cell sampling moves the effective monitor when the mesh changes. Linear
    interpolation reproduces a planar regional field and keeps the coordinate fixed.
    A configured point inside a well cell is rejected rather than silently relocated.
    """
    from scipy.interpolate import LinearNDInterpolator

    if not Domain.from_config(config).contains(*point):
        raise ValueError("convergence observation point must lie inside every domain")
    cell = mesh.locate_cell(*point)
    if cell in {mesh.locate_cell(*xy) for xy in _all_well_xy(config)}:
        raise ValueError("convergence observation point must be outside all well cells")
    values = np.asarray(LinearNDInterpolator(mesh.centroids, head.T)(*point), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("observation heads must be finite and inside the cell-centre convex hull")
    return values


@dataclass
class ConvergenceResult:
    """Outcome of a grid-convergence study (levels ordered coarse -> fine)."""

    levels: list[ConvergenceLevel]
    rel_change: dict[str, float]  # between the two finest levels
    converged: bool
    #: Where ``final_head`` was sampled (chosen once on the baseline mesh; see
    #: :func:`_observation_point`). Recorded so the choice is visible in the report.
    observation_point: tuple[float, float] = (float("nan"), float("nan"))

    def summary(self) -> str:
        lines = ["grid convergence (coarse -> fine):"]
        for lv in self.levels:
            lines.append(
                f"  factor {lv.refinement_factor:<4g} cells={lv.n_cells:6d}  "
                f"peakT={lv.peak_temperature:.4f}  obsDrawdown={lv.observation_drawdown:.4f}  "
                f"wellCellDrawdown(diagnostic)={lv.max_drawdown:.4f}  "
                f"breakthrough={lv.breakthrough_time:.3g}  wellT={lv.final_temperature:.4f}  "
                f"obsH={lv.final_head:.4f}"
            )
        lines.append("  relative change (finest two levels):")
        for name, val in self.rel_change.items():
            lines.append(f"    {name:<18} {val:.2%}" if math.isfinite(val) else f"    {name:<18} n/a")
        lines.append(f"  => converged: {self.converged}")
        return "\n".join(lines)


def run_convergence(
    config: Config,
    factors: Sequence[float] = DEFAULT_FACTORS,
    workspace: Path | str | None = None,
    *,
    compact_outputs: bool = False,
    max_level_bytes: int | None = None,
) -> ConvergenceResult:
    """Run the pipeline at each refinement factor and assess output convergence.

    ``compact_outputs`` disables cell-budget files and removes the head and temperature
    binaries after their metrics have been extracted. The model inputs and listing files
    remain available for provenance. ``max_level_bytes`` applies a conservative preflight
    limit to the dense outputs and verifies their actual logical size before cleanup.
    Time stepping is pinned to fixed, uniform steps on a copy of the configuration.
    ``observation_drawdown`` gates hydraulic refinement; ``max_drawdown`` remains the
    raw extraction-cell diagnostic. See ``docs/validation_decisions.md``.
    """
    from ATESTool.analysis import analyse

    if len(factors) < 2:
        raise ValueError("convergence needs at least two refinement factors")
    ordered = sorted(factors, reverse=True)  # 1.0 (coarse) -> 0.5 (fine)
    if any(not math.isfinite(f) or f <= 0 for f in ordered) or len(set(ordered)) != len(ordered):
        raise ValueError("refinement factors must be distinct, finite and positive")
    # Isolate space even when called directly with a Courant/ATS configuration.
    config = copy.deepcopy(config)
    config.model.simulation.time_stepping = TimeSteppingConfig("fixed", 1.0, 1.0, 2.0, 5.0)
    base = Path(workspace) if workspace is not None else Path.cwd() / "convergence"

    levels: list[ConvergenceLevel] = []
    observation_point: tuple[float, float] | None = None
    for factor in ordered:
        run_ws = base / f"level_{factor:g}"
        run_ws.mkdir(parents=True, exist_ok=True)
        mesh = generate_mesh(config, refinement_factor=factor)
        if observation_point is None:  # fix the sample point on the coarsest (baseline) mesh
            observation_point = _observation_point(config, mesh)
        paths = build_model(config, mesh, run_ws, budget_save=None if compact_outputs else "ALL")
        if compact_outputs:
            _guard_dense_output_size(
                paths,
                n_cells=mesh.n_cells,
                max_bytes=max_level_bytes,
                level_label=f"spatial level {factor:g}",
            )
        level: ConvergenceLevel | None = None
        deleted_binary_bytes = 0
        try:
            sim = analyse(paths, run_model(paths), config, mesh)
            stats = sim.stats
            ts = sim.timeseries
            observation_head = _observation_head(config, mesh, sim.head, observation_point)
            level = ConvergenceLevel(
                refinement_factor=factor,
                n_cells=mesh.n_cells,
                peak_temperature=stats["peak_temperature"],
                max_drawdown=stats["max_drawdown"],
                breakthrough_time=stats["breakthrough_time"],
                final_temperature=float(ts["temperature"].iloc[-1]),
                final_head=float(observation_head[-1]),
                observation_drawdown=float(np.max(sim.initial_head - observation_head)),
            )
        finally:
            if compact_outputs:
                deleted_binary_bytes = _cleanup_binary_outputs(
                    paths,
                    max_bytes=max_level_bytes,
                    level_label=f"spatial level {factor:g}",
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
    assert observation_point is not None  # at least one level ran (>=2 factors required)
    return ConvergenceResult(
        levels=levels,
        rel_change=rel_change,
        converged=converged,
        observation_point=observation_point,
    )


def _metric_converged(a: float, b: float, rel_tol: float, abs_floor: float) -> bool:
    """A metric has converged if its finest-two-level change is within tol or the floor."""
    if math.isnan(a) and math.isnan(b):
        return True  # no signal at either resolution -> stable
    if math.isnan(a) or math.isnan(b):
        return False  # a breakthrough appeared or vanished -> not resolved
    if abs(a - b) <= abs_floor:
        return True
    denom = abs(b)
    return (abs(a - b) / denom) <= rel_tol if denom > 1e-12 else False


def _rel(a: float, b: float) -> float:
    """Relative change of ``a`` vs the finer reference ``b`` (for reporting)."""
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return float("inf")
    denom = abs(b)
    return abs(a - b) / denom if denom > 1e-12 else abs(a - b)

"""Optional verification bundle for a deterministic run.

``run_diagnostic`` takes a run that has already happened and, without touching its results,
adds everything you would otherwise have to assemble by hand:

* **every plot** the package can draw for a single run, written to disk;
* the **spatial** convergence study (mesh refinement, :func:`~ATESTool.validation.run_convergence`);
* the **temporal** convergence study (Courant refinement,
  :func:`~ATESTool.validation.run_temporal_convergence`);
* the **domain** convergence study (domain-size refinement,
  :func:`~ATESTool.validation.run_domain_convergence`), the most expensive of the three;

It is an orchestrator, not a new user interface: it consumes the named seams
(``Config``, ``VertexGrid``, ``SimulationResult``, ``WellSchedule``) that
``run_from_config`` already has in hand, and the baseline mesh is passed in rather than
regenerated. The convergence studies do rebuild meshes; that is their entire purpose.

**This is expensive.** The two convergence studies run the full pipeline once per level, so
the default settings cost roughly five extra model runs on top of the baseline; the finest
levels are the slow ones (~4x the cells, ~4x the steps). Cost scales linearly with
``simulation.n_years``. The estimate is logged before any of it starts.

Each stage is isolated: a study or a plot that raises is recorded in
:attr:`DiagnosticReport.failures` and the rest continue, because losing a completed study to
a later unrelated failure is not an acceptable trade at this price.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd
from matplotlib.figure import Figure

from ATESTool.analysis import SimulationResult
from ATESTool.analysis.plots import (
    plot_arrival_time,
    plot_doublet_temperatures,
    plot_drawdown,
    plot_efficiency,
    plot_energy_balance,
    plot_model_overview,
    plot_plume_extent,
    plot_snapshot,
    plot_timeseries,
)
from ATESTool.analysis.plume import plume_extent
from ATESTool.analysis.report import write_html_report
from ATESTool.config import Config, TimeSteppingConfig
from ATESTool.mesh import VertexGrid, generate_mesh
from ATESTool.model.schedule import WellSchedule
from ATESTool.model.timestepping import assess_courant
from ATESTool.validation.convergence import (
    DEFAULT_FACTORS,
    ConvergenceResult,
    run_convergence,
)
from ATESTool.validation.domain_convergence import (
    DEFAULT_SIZES,
    DomainConvergenceResult,
    run_domain_convergence,
)
from ATESTool.validation.plots import (
    plot_convergence,
    plot_domain_convergence,
    plot_temporal_convergence,
)
from ATESTool.validation.temporal_convergence import (
    DEFAULT_TARGETS,
    TemporalConvergenceResult,
    run_temporal_convergence,
)

logger = logging.getLogger("ATESTool.diagnostic")

#: Everything the diagnostic writes lives under this subdirectory of the workspace, so the
#: normal single-run artifacts stay where they have always been.
DIAGNOSTIC_DIR = "diagnostic"

_DPI = 150


@dataclass
class DiagnosticReport:
    """What the diagnostic produced, and what it failed to produce."""

    directory: Path
    plots: list[Path] = field(default_factory=list)
    spatial: ConvergenceResult | None = None
    temporal: TemporalConvergenceResult | None = None
    domain: DomainConvergenceResult | None = None
    #: stage name -> error text, for every stage that raised.
    failures: dict[str, str] = field(default_factory=dict)
    #: stage name -> wall-clock seconds.
    elapsed: dict[str, float] = field(default_factory=dict)
    #: The standalone HTML report bundling the run's stats + every plot; ``None``
    #: if that stage failed (recorded in ``failures`` like any other stage).
    html_report: Path | None = None

    @property
    def ok(self) -> bool:
        """True when every stage completed."""
        return not self.failures

    def summary(self) -> str:
        lines = [f"diagnostic report ({self.directory}):", ""]
        for name, result in (
            ("spatial convergence", self.spatial),
            ("temporal convergence", self.temporal),
            ("domain convergence", self.domain),
        ):
            if result is None:
                lines.append(f"{name}: not produced ({self.failures.get(name, 'skipped')})")
            else:
                lines.append(result.summary())
                if name == "domain convergence":
                    lines.append("Boundary sensitivity diagnostic; domain independence is required only "
                                 "for scenarios claiming an effectively unbounded aquifer.")
            lines.append("")

        lines.append(f"plots written: {len(self.plots)}")
        lines.append("timings:")
        for name, seconds in self.elapsed.items():
            lines.append(f"  {name:<24} {seconds:8.1f} s")
        if self.failures:
            lines.append("failures:")
            for name, message in self.failures.items():
                lines.append(f"  {name:<24} {message}")
        else:
            lines.append("all stages completed")
        return "\n".join(lines)


def _finite(value: float) -> float | None:
    """``value``, or ``None`` when it is NaN/inf.

    ``json.dumps`` emits bare ``NaN``/``Infinity``, which is not valid JSON and trips strict
    parsers. Non-finite values are routine here (``breakthrough_time`` is NaN whenever the
    front never reaches the observation well), so every number is passed through this on the
    way out.
    """
    v = float(value)
    return v if math.isfinite(v) else None


def _finite_map(values: dict[str, float]) -> dict[str, float | None]:
    return {name: _finite(v) for name, v in values.items()}


def _levels_frame(levels: Sequence[Any]) -> pd.DataFrame:
    """One row per convergence level, columns from the level dataclass's own fields."""
    return pd.DataFrame([dataclasses.asdict(level) for level in levels])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


@dataclass
class _PinnedTimeStepping:
    """The spatial study's actual time-stepping config, and what changed from the input."""

    config: Config
    original_mode: str | None  # None when the input was already "fixed"
    configured_nstp: int  # input config's timesteps_per_season
    safe_nstp: int  # Courant-safe step count actually used (>= configured_nstp)


def _pin_fixed_timestepping(
    config: Config, factors: Sequence[float]
) -> _PinnedTimeStepping:
    """A copy of ``config`` with fixed time stepping at a Courant-safe step count.

    The spatial study must vary *only* the mesh. Two things would otherwise leak the time
    axis into a "spatial-only" comparison:

    1. Under ``courant``/``ats`` the time step is derived from the cell size, so refining the
       mesh would also refine the step. Pinning ``mode="fixed"`` handles this.
    2. Pinning to the *input config's own* ``timesteps_per_season`` is not enough by itself:
       if that step under-resolves the thermal front (Courant >> 1), the same temporal error
       is present at every mesh level and can dominate the true spatial error, especially at
       the finest (most accurate) level. So the pinned step count is raised, never lowered,
       to whatever :func:`~ATESTool.model.timestepping.assess_courant` recommends at the
       *finest* mesh level being studied, which keeps temporal error negligible everywhere.

    ``run_temporal_convergence`` already pins its own mesh for the mirror reason.

    Pinning here rather than inside ``run_convergence`` keeps that published library
    function's own behaviour unchanged for its other callers.
    """
    stepping = config.model.simulation.time_stepping
    original = stepping.mode if stepping is not None else None
    configured_nstp = max(int(round(float(config.model.simulation.timesteps_per_season))), 1)

    variant = _deepcopy_config(config)
    variant.model.simulation.time_stepping = TimeSteppingConfig(
        mode="fixed", target_courant=1.0, tsmult=1.0, ats_dtadj=2.0, ats_dtfailadj=5.0
    )

    finest_mesh = generate_mesh(variant, refinement_factor=min(factors))
    courant_report = assess_courant(finest_mesh, variant)
    safe_nstp = (
        configured_nstp
        if math.isinf(courant_report.dt_max)
        else max(configured_nstp, courant_report.recommended_nstp)
    )
    variant.model.simulation.timesteps_per_season = float(safe_nstp)

    return _PinnedTimeStepping(
        config=variant,
        original_mode=None if original in (None, "fixed") else original,
        configured_nstp=configured_nstp,
        safe_nstp=safe_nstp,
    )


def _deepcopy_config(config: Config) -> Config:
    import copy

    return copy.deepcopy(config)


def _snapshot_times(config: Config, sim: SimulationResult) -> list[float]:
    """Times to draw spatial snapshots at: the config's, else the last reported time."""
    requested = [float(t) for t in config.analysis.times_of_interest]
    if requested:
        return requested
    return [float(sim.times[-1])]


@contextmanager
def _stage(report: DiagnosticReport, name: str) -> Iterator[None]:
    """Run a stage, recording its duration and isolating its failure.

    ``Exception`` and not ``BaseException``: a ``KeyboardInterrupt`` part-way through a
    long diagnostic must still stop it.
    """
    logger.info("diagnostic: %s ...", name)
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - deliberate: one stage must not sink the rest
        report.failures[name] = f"{type(exc).__name__}: {exc}"
        logger.exception("diagnostic: %s failed", name)
    finally:
        report.elapsed[name] = time.perf_counter() - started


def _save(fig: Figure, path: Path) -> None:
    """Write ``fig`` to ``path``.

    ``Figure.savefig`` works without pyplot, so the package keeps its no-global-state
    guarantee, and with no global figure registry there is nothing to close and nothing
    to leak.
    """
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")


def _write_plots(
    report: DiagnosticReport,
    plot_dir: Path,
    config: Config,
    mesh: VertexGrid,
    sim: SimulationResult,
    schedule: WellSchedule,
) -> None:
    """Draw every single-run plot, isolating each one."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    def emit(name: str, build: Any) -> None:
        with _stage(report, f"plot {name}"):
            path = plot_dir / f"{name}.png"
            _save(build(), path)
            report.plots.append(path)

    emit("timeseries", lambda: plot_timeseries(sim))
    emit("efficiency", lambda: plot_efficiency(sim))
    emit("energy_balance", lambda: plot_energy_balance(sim, schedule, config))
    emit("doublet_temperatures", lambda: plot_doublet_temperatures(sim, mesh, schedule))
    emit("model_overview", lambda: plot_model_overview(sim, mesh, schedule))
    emit("arrival_time", lambda: plot_arrival_time(sim, mesh, schedule=schedule))
    emit("arrival_time_zoom", lambda: plot_arrival_time(sim, mesh, schedule=schedule, zoom=True))
    emit("plume_extent", lambda: plot_plume_extent(plume_extent(sim, mesh, schedule, config)))

    for t in _snapshot_times(config, sim):
        emit(f"snapshot_temperature_t{t:g}",
             lambda t=t: plot_snapshot(sim, t, mesh, variable="temperature", schedule=schedule))
        emit(f"snapshot_temperature_t{t:g}_zoom",
             lambda t=t: plot_snapshot(sim, t, mesh, variable="temperature", schedule=schedule, zoom=True))
        emit(f"snapshot_head_t{t:g}",
             lambda t=t: plot_snapshot(sim, t, mesh, variable="head", schedule=schedule))
        emit(f"snapshot_head_t{t:g}_zoom",
             lambda t=t: plot_snapshot(sim, t, mesh, variable="head", schedule=schedule, zoom=True))
        emit(f"drawdown_t{t:g}", lambda t=t: plot_drawdown(sim, t, mesh, schedule=schedule))
        emit(f"drawdown_t{t:g}_zoom", lambda t=t: plot_drawdown(sim, t, mesh, schedule=schedule, zoom=True))


def _cost_estimate(factors: Sequence[float], targets: Sequence[float]) -> tuple[float, float]:
    """Rough (spatial, temporal) cost in baseline-run-equivalents.

    Halving a refinement factor roughly quadruples the cell count (2-D refinement), so a
    spatial level costs ``1/factor**2``. Halving the Courant target doubles the step count,
    so a temporal level costs ``coarsest/target``. Both are crude (solver time is not
    exactly linear in either), but they get the order of magnitude right, which is the
    decision the user is actually making.
    """
    spatial = sum(1.0 / f**2 for f in factors if f > 0.0)
    coarsest = max(targets) if targets else 0.0
    temporal = sum(coarsest / t for t in targets if t > 0.0)
    return spatial, temporal


def _log_cost_estimate(factors: Sequence[float], targets: Sequence[float]) -> None:
    """Warn up front what the studies are about to cost."""
    spatial, temporal = _cost_estimate(factors, targets)
    logger.warning(
        "diagnostic: about to run %d spatial + %d temporal model runs on top of the baseline "
        "(~%.0f + ~%.0f = ~%.0f baseline-run-equivalents; the finest level of each dominates). "
        "Cost scales linearly with simulation.n_years.",
        len(factors), len(targets), spatial, temporal, spatial + temporal,
    )


def run_diagnostic(
    config: Config,
    mesh: VertexGrid,
    sim: SimulationResult,
    schedule: WellSchedule,
    workspace: Path,
    factors: Sequence[float] = DEFAULT_FACTORS,
    targets: Sequence[float] = DEFAULT_TARGETS,
    sizes: Sequence[float] = DEFAULT_SIZES,
) -> DiagnosticReport:
    """Write every plot and run all four verification studies for a completed run.

    ``config``/``mesh``/``sim``/``schedule`` describe the run that has already happened;
    nothing here re-runs or re-analyses it. Output lands under ``workspace/diagnostic/``.
    """
    directory = workspace / DIAGNOSTIC_DIR
    directory.mkdir(parents=True, exist_ok=True)
    report = DiagnosticReport(directory=directory)

    _write_plots(report, directory / "plots", config, mesh, sim, schedule)
    _log_cost_estimate(factors, targets)

    runs = directory / "runs"
    with _stage(report, "spatial convergence"):
        pinned = _pin_fixed_timestepping(config, factors)
        if pinned.original_mode is not None:
            logger.warning(
                "diagnostic: spatial convergence pinned to fixed time stepping (config "
                "requested %r) so refining the mesh does not also refine the time step",
                pinned.original_mode,
            )
        if pinned.safe_nstp != pinned.configured_nstp:
            logger.warning(
                "diagnostic: spatial convergence step count raised %d -> %d steps/season "
                "(Courant-safe at the finest mesh level) so temporal error does not "
                "contaminate the spatial-only comparison",
                pinned.configured_nstp,
                pinned.safe_nstp,
            )
        started = time.perf_counter()
        report.spatial = run_convergence(pinned.config, factors, workspace=runs / "spatial")
        _write_spatial(directory, report, pinned, time.perf_counter() - started)

    with _stage(report, "temporal convergence"):
        started = time.perf_counter()
        report.temporal = run_temporal_convergence(config, targets, workspace=runs / "temporal")
        _write_temporal(directory, report, time.perf_counter() - started)

    with _stage(report, "domain convergence"):
        logger.warning(
            "diagnostic: domain convergence runs a full-domain solve at each of %d domain "
            "sizes (%s m) and is the most expensive study; every level has pumped and reference runs and the "
            "largest domain is the slowest.",
            len(sizes), ", ".join(f"{s:g}" for s in sorted(sizes)),
        )
        started = time.perf_counter()
        report.domain = run_domain_convergence(config, sizes, workspace=runs / "domain")
        _write_domain(directory, report, time.perf_counter() - started)

    (directory / "summary.txt").write_text(report.summary() + "\n")

    with _stage(report, "html report"):
        report.html_report = write_html_report(
            directory / "report.html",
            stats=sim.stats,
            plot_paths=report.plots,
            summary_text=report.summary(),
            title=f"ATESTool diagnostic report: {workspace.name}",
        )

    if report.failures:
        logger.warning(
            "diagnostic: %d stage(s) failed: %s", len(report.failures), ", ".join(report.failures)
        )
    logger.info("diagnostic: outputs in %s", directory)
    return report


def _write_spatial(
    directory: Path, report: DiagnosticReport, pinned: _PinnedTimeStepping, elapsed: float
) -> None:
    result = report.spatial
    assert result is not None  # only called inside the stage that set it
    _levels_frame(result.levels).to_csv(directory / "spatial_convergence.csv", index=False)
    _write_json(
        directory / "spatial_convergence.json",
        {
            "converged": result.converged,
            "factors": [lv.refinement_factor for lv in result.levels],
            "n_cells": [lv.n_cells for lv in result.levels],
            "observation_point": [_finite(result.observation_point[0]), _finite(result.observation_point[1])],
            "rel_change": _finite_map(result.rel_change),
            "time_stepping_pinned": pinned.original_mode,
            "timesteps_per_season_pinned": (
                None
                if pinned.safe_nstp == pinned.configured_nstp
                else {"configured": pinned.configured_nstp, "safe": pinned.safe_nstp}
            ),
            "elapsed_seconds": _finite(elapsed),
        },
    )
    _save(plot_convergence(result), directory / "plots" / "spatial_convergence.png")
    report.plots.append(directory / "plots" / "spatial_convergence.png")


def _write_temporal(directory: Path, report: DiagnosticReport, elapsed: float) -> None:
    result = report.temporal
    assert result is not None
    _levels_frame(result.levels).to_csv(directory / "temporal_convergence.csv", index=False)
    _write_json(
        directory / "temporal_convergence.json",
        {
            "converged": result.converged,
            "healthy": result.healthy,
            "targets": [lv.target_courant for lv in result.levels],
            "rel_change": _finite_map(result.rel_change),
            "observed_order": _finite_map(result.observed_order),
            "richardson_limit": _finite_map(result.richardson_limit),
            "elapsed_seconds": _finite(elapsed),
        },
    )
    _save(plot_temporal_convergence(result), directory / "plots" / "temporal_convergence.png")
    report.plots.append(directory / "plots" / "temporal_convergence.png")


def _write_domain(directory: Path, report: DiagnosticReport, elapsed: float) -> None:
    result = report.domain
    assert result is not None
    _levels_frame(result.levels).to_csv(directory / "domain_convergence.csv", index=False)
    _write_json(
        directory / "domain_convergence.json",
        {
            "converged": result.converged,
            "healthy": result.healthy,
            "sizes": [lv.domain_size for lv in result.levels],
            "observation_point": [_finite(v) for v in result.observation_point],
            "boundary_mode": result.boundary_mode,
            "role": "boundary_sensitivity_diagnostic",
            "hydraulic_metric": "pumping_drawdown = max(reference head - pumped head) at observation_point",
            "n_cells": [lv.n_cells for lv in result.levels],
            "rel_change": _finite_map(result.rel_change),
            "boundary_thermal_influence": [
                _finite(lv.boundary_thermal_influence) for lv in result.levels
            ],
            "elapsed_seconds": _finite(elapsed),
        },
    )
    _save(plot_domain_convergence(result), directory / "plots" / "domain_convergence.png")
    report.plots.append(directory / "plots" / "domain_convergence.png")

"""Plotting for the validation studies.

Each function renders the result type produced by a deterministic validation study:
metric-vs-resolution for the convergence studies and a radial overlay for the analytical
benchmark. As in :mod:`ATESTool.analysis.plots`, every function builds a
standalone ``Figure`` (no pyplot, no global state), so they are safe to call headless.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.figure import Figure

from ATESTool.analysis.plots import _figure_and_axes
from ATESTool.validation.analytical import AnalyticalResult, radial_profile
from ATESTool.validation.convergence import ConvergenceResult
from ATESTool.validation.domain_convergence import DomainConvergenceResult
from ATESTool.validation.health import DISCREPANCY_TOL
from ATESTool.validation.temporal_convergence import TemporalConvergenceResult

if TYPE_CHECKING:
    from ATESTool.analysis import SimulationResult
    from ATESTool.mesh import VertexGrid

#: Metrics that carry a physical scale worth plotting (NaN-only metrics are skipped).
_CONVERGED_COLOR = "tab:green"
_NOT_CONVERGED_COLOR = "tab:red"

def _metric_names(levels: list[Any]) -> list[str]:
    """Metric fields of a convergence level that have a usable (finite) value somewhere."""
    if not levels:
        return []
    skip = {"refinement_factor", "n_cells", "target_courant", "n_timesteps",
            "max_discrepancy", "failed_steps", "deleted_binary_bytes"}
    names = [f for f in vars(levels[0]) if f not in skip]
    return [n for n in names if any(math.isfinite(lv.metric(n)) for lv in levels)]


def _normalised(levels: list[Any], name: str) -> list[float]:
    """Each level's metric divided by the finest level's value (so all metrics share an axis)."""
    values = [lv.metric(name) for lv in levels]
    reference = values[-1]
    if not math.isfinite(reference) or reference == 0.0:
        return [float("nan")] * len(values)
    return [v / reference for v in values]


def plot_convergence(result: ConvergenceResult, ax: Any | None = None) -> Figure:
    """Each metric versus mesh size, normalised to the finest level.

    Normalising puts metrics with different units (metres, degC, days) on one axis: a
    converged study shows every line flattening to 1.0 at the right-hand (finest) end.
    """
    fig, ax = _figure_and_axes(ax)
    levels = result.levels
    if not levels:
        ax.set_title("grid convergence: no levels")
        return fig

    cells = [lv.n_cells for lv in levels]
    for name in result.rel_change:
        if any(math.isfinite(lv.metric(name)) for lv in levels):
            ax.plot(cells, _normalised(levels, name), marker="o", label=name)

    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("mesh size [cells]")
    ax.set_ylabel("metric / finest-level value")
    verdict = "converged" if result.converged else "NOT converged"
    color = _CONVERGED_COLOR if result.converged else _NOT_CONVERGED_COLOR
    ax.set_title(f"grid convergence: {verdict}", color=color)
    ax.legend(fontsize="small")
    return fig


#: The domain study's verdict metrics (the level dataclass carries extra bookkeeping fields).
_DOMAIN_METRICS = ("pumping_drawdown", "peak_temperature", "mean_recovery_efficiency")


def plot_domain_convergence(result: DomainConvergenceResult, ax: Any | None = None) -> Figure:
    """Each verdict metric versus domain size, normalised to the largest domain.

    A converged study shows every line flattening to 1.0 at the right-hand (largest-domain)
    end: once the constant-head boundary is far enough away, growing the domain further stops
    moving the outputs over the simulated horizon. The hydraulic metric uses a paired
    no-well reference at a fixed monitoring point; raw well-cell heads are diagnostic.
    """
    fig, ax = _figure_and_axes(ax)
    levels = result.levels
    if not levels:
        ax.set_title("domain-size sensitivity: no levels")
        return fig

    sizes = [lv.domain_size for lv in levels]
    for name in _DOMAIN_METRICS:
        if any(math.isfinite(lv.metric(name)) for lv in levels):
            ax.plot(sizes, _normalised(levels, name), marker="o", label=name)

    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("domain size [m]")
    ax.set_ylabel("metric / largest-domain value")
    color = _CONVERGED_COLOR if result.converged else _NOT_CONVERGED_COLOR
    sensitivity = "stable over tested sizes" if result.converged else "boundary-sensitive"
    ax.set_title(f"domain-size sensitivity: {sensitivity}", color=color)
    ax.legend(fontsize="small")
    return fig


def plot_temporal_convergence(result: TemporalConvergenceResult) -> Figure:
    """Two panels: the error-vs-step-size trend, and per-level numerical health.

    Panel (a) plots each metric's distance from its Richardson Δt→0 limit against the
    Courant target on log-log axes; a first-order scheme should fall on a slope-1 line, and
    the fitted observed order is quoted per metric. Panel (b) is the health gauge: the GWE
    mass-balance discrepancy per level against its tolerance, plus any failed time steps.
    """
    fig = Figure(figsize=(11.0, 4.5), layout="constrained")
    ax_err, ax_health = fig.subplots(1, 2)
    levels = result.levels
    if not levels:
        ax_err.set_title("time-step convergence: no levels")
        return fig

    targets = [lv.target_courant for lv in levels]
    for name in _metric_names(levels):
        limit = result.richardson_limit.get(name, float("nan"))
        if not math.isfinite(limit):
            continue
        errors = [abs(lv.metric(name) - limit) for lv in levels]
        if not all(e > 0.0 and math.isfinite(e) for e in errors):
            continue  # a metric sitting exactly on its limit has no slope to show
        order = result.observed_order.get(name, float("nan"))
        label = f"{name} (p={order:.2f})" if math.isfinite(order) else f"{name} (p=n/a)"
        ax_err.plot(targets, errors, marker="o", label=label)

    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.invert_xaxis()  # refinement runs left-to-right (coarse -> fine)
    ax_err.set_xlabel("target Courant number (refining →)")
    ax_err.set_ylabel("|metric − Richardson limit|")
    ax_err.set_title("(a) error vs time-step size")
    if ax_err.get_legend_handles_labels()[0]:
        ax_err.legend(fontsize="small")

    positions = np.arange(len(levels))
    discrepancies = [
        lv.max_discrepancy if math.isfinite(lv.max_discrepancy) else 0.0 for lv in levels
    ]
    colors = [
        _NOT_CONVERGED_COLOR if d > DISCREPANCY_TOL else _CONVERGED_COLOR for d in discrepancies
    ]
    ax_health.bar(positions, discrepancies, color=colors)
    ax_health.axhline(
        DISCREPANCY_TOL, color="grey", linestyle="--", linewidth=1.0,
        label=f"tolerance ({DISCREPANCY_TOL:g}%)",
    )
    for pos, lv in zip(positions, levels):
        if lv.failed_steps:
            ax_health.annotate(
                f"{lv.failed_steps} failed", (pos, 0.0), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize="small", color=_NOT_CONVERGED_COLOR,
            )
    ax_health.set_xticks(positions)
    ax_health.set_xticklabels([f"Co≤{lv.target_courant:g}" for lv in levels])
    ax_health.set_ylabel("max |mass-balance discrepancy| [%]")
    health = "healthy" if result.healthy else "UNHEALTHY"
    ax_health.set_title(f"(b) numerical health: {health}")
    ax_health.legend(fontsize="small")

    verdict = "converged" if result.converged else "NOT converged"
    color = _CONVERGED_COLOR if result.converged else _NOT_CONVERGED_COLOR
    fig.suptitle(f"time-step convergence: {verdict}", color=color)
    return fig


def plot_thermal_front(
    result: AnalyticalResult,
    sim: SimulationResult,
    mesh: VertexGrid,
    ax: Any | None = None,
) -> Figure:
    """Radial temperature profiles with the modelled and analytical front radii marked.

    One profile per sampled time, blue (early, small plume) to red (late, large plume);
    matching the module-wide warm=red convention. The modelled front is where each profile
    crosses the midpoint temperature (solid vertical marker); the analytical radius is the
    dashed one. Divergence between the pair is the benchmark error made visible.
    """
    fig, ax = _figure_and_axes(ax)
    if not result.samples:
        ax.set_title("thermal front: no samples")
        return fig

    shades = np.linspace(0.35, 0.95, len(result.samples))
    for sample, shade in zip(result.samples, shades):
        color = (shade, 0.2, 1.0 - shade)  # blue (early) -> red (late)
        r, temp = radial_profile(sim.snapshot(sample.time, "temperature"), mesh, result.well)
        near = r < result.search_radius
        ax.plot(r[near], temp[near], color=color, linewidth=1.2, label=f"t={sample.time:.0f} d")
        ax.axvline(sample.model_radius, color=color, linestyle="-", alpha=0.6, linewidth=1.0)
        ax.axvline(sample.analytical_radius, color=color, linestyle="--", alpha=0.6, linewidth=1.0)

    ax.axhline(
        result.midpoint_temperature, color="grey", linestyle=":", linewidth=1.0,
        label="midpoint T (front definition)",
    )
    ax.set_xlabel("distance from warm well [m]")
    ax.set_ylabel("temperature [degC]")
    ax.set_title(
        f"thermal front: model (solid) vs analytical (dashed): "
        f"max error {result.max_rel_error:.1%}"
    )
    ax.legend(fontsize="small")
    return fig

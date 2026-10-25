"""Plotting helpers for a :class:`SimulationResult`.

All functions build a standalone ``matplotlib`` ``Figure`` (no pyplot/global
state), so they are safe to call headless.

Colour convention (shared across plots): the warm well is red, the cold well is
blue, matching the ATES injection (warm) / extraction (cold) phases.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ATESTool.analysis.energy import compute_energy_balance
from ATESTool.analysis.plume import DEFAULT_PLUME_TOL, PlumeSample, arrival_time_map
from ATESTool.analysis.results import SimulationResult
from ATESTool.mesh import VertexGrid
from ATESTool.model.schedule import SEASONS_PER_YEAR

if TYPE_CHECKING:
    from ATESTool.config import Config
    from ATESTool.model.schedule import WellSchedule

#: Warm/cold well colour convention, shared by every plot in this module.
_WARM_COLOR = "tab:red"
_COLD_COLOR = "tab:blue"


def _figure_and_axes(ax: Any | None) -> tuple[Figure, Any]:
    if ax is None:
        fig = Figure()
        return fig, fig.subplots()
    return ax.figure, ax


def _reference_head(sim: SimulationResult, fallback: Any) -> Any:
    """``sim.initial_head`` when known, else ``fallback`` (the first reported head;
    best-effort for synthetic results that don't set ``initial_head``, and for real
    results, avoids the drift-with-dt bug of using the first *reported* timestep).
    """
    return fallback if math.isnan(sim.initial_head) else sim.initial_head


#: Padding added around the well cluster's own bounding box, as a fraction of its
#: half-span, on every side (so a well sitting right at the cluster's edge is not
#: clipped against the axes border).
_ZOOM_PADDING_FACTOR = 0.5
#: Minimum zoom half-width/half-height [m], for the degenerate case of a single well
#: or a very tight cluster, where the padded bounding box would otherwise collapse.
_ZOOM_MIN_HALF_SIZE = 50.0


def _zoom_limits(
    sim: SimulationResult, mesh: VertexGrid, schedule: WellSchedule | None
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits framing every well (with headroom), clipped to the mesh's own extent.

    "Every well" is the extraction well (located via ``sim.extraction_cell``, since
    plotting functions do not otherwise see its coordinates) plus, when a ``schedule``
    is given, every doublet warm/cold well. Centred on that cluster's bounding box,
    padded by ``_ZOOM_PADDING_FACTOR`` of its own half-span and floored at
    ``_ZOOM_MIN_HALF_SIZE`` so a single well still gets a sensible window.
    """
    xs = [float(mesh.centroids[sim.extraction_cell, 0])]
    ys = [float(mesh.centroids[sim.extraction_cell, 1])]
    if schedule is not None:
        for x, y in schedule.well_locations.values():
            xs.append(float(x))
            ys.append(float(y))

    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    half_x = max((max(xs) - min(xs)) / 2.0 * (1.0 + _ZOOM_PADDING_FACTOR), _ZOOM_MIN_HALF_SIZE)
    half_y = max((max(ys) - min(ys)) / 2.0 * (1.0 + _ZOOM_PADDING_FACTOR), _ZOOM_MIN_HALF_SIZE)
    half_x = half_y = max(half_x, half_y)

    mesh_x_min, mesh_x_max = float(mesh.vertices[:, 0].min()), float(mesh.vertices[:, 0].max())
    mesh_y_min, mesh_y_max = float(mesh.vertices[:, 1].min()), float(mesh.vertices[:, 1].max())
    xlim = (max(cx - half_x, mesh_x_min), min(cx + half_x, mesh_x_max))
    ylim = (max(cy - half_y, mesh_y_min), min(cy + half_y, mesh_y_max))
    return xlim, ylim


def _visible_mask(
    mesh: VertexGrid, xlim: tuple[float, float] | None, ylim: tuple[float, float] | None
) -> np.ndarray | None:
    """Boolean mask of cells whose centre falls inside ``xlim``/``ylim``.

    ``None`` (meaning "every cell") when both are ``None``, so callers can treat the
    unzoomed case uniformly without a branch.
    """
    if xlim is None and ylim is None:
        return None
    centroids = mesh.centroids
    mask = np.ones(centroids.shape[0], dtype=bool)
    if xlim is not None:
        mask &= (centroids[:, 0] >= xlim[0]) & (centroids[:, 0] <= xlim[1])
    if ylim is not None:
        mask &= (centroids[:, 1] >= ylim[0]) & (centroids[:, 1] <= ylim[1])
    return mask


def _classify_well(well_id: str) -> str | None:
    """``"warm"``/``"cold"`` from a ``WellSchedule`` well id, else ``None``.

    Follows ``build_well_schedule``'s ``{doublet}_warm`` / ``{doublet}_cold``
    naming convention.
    """
    if well_id.endswith("_warm"):
        return "warm"
    if well_id.endswith("_cold"):
        return "cold"
    return None


def plot_timeseries(sim: SimulationResult, ax: Any | None = None) -> Figure:
    """Temperature and head at the extraction well over time (twin y-axes)."""
    fig, ax = _figure_and_axes(ax)
    ts = sim.timeseries

    ax.plot(ts.index, ts["temperature"], color="tab:red", label="temperature")
    ax.set_xlabel("time [days]")
    ax.set_ylabel("temperature [degC]", color="tab:red")

    ax_head = ax.twinx()
    ax_head.plot(ts.index, ts["head"], color="tab:blue", label="head")
    ax_head.set_ylabel("head [m]", color="tab:blue")
    return fig


def plot_doublet_temperatures(
    sim: SimulationResult, mesh: VertexGrid, schedule: WellSchedule, ax: Any | None = None
) -> Figure:
    """Temperature at each doublet's warm and cold well over time.

    The signature ATES figure: warm-well temperature (red) and cold-well
    temperature (blue) for every doublet in the ``schedule``, with the ambient
    temperature marked for reference. With more than one doublet the wells are
    distinguished by line style. Built entirely on the public seams: the well
    cell is located on the ``mesh`` and the per-cell series is read through
    :meth:`SimulationResult.snapshot`.
    """
    fig, ax = _figure_and_axes(ax)
    times = np.asarray(sim.timeseries.index, dtype=float)
    field = np.array([sim.snapshot(t, "temperature") for t in times])  # (n_times, n_cells)

    line_styles = ["-", "--", "-.", ":"]
    doublet_names = sorted({wid.rsplit("_", 1)[0] for wid in schedule.well_ids})
    style_of = {name: line_styles[i % len(line_styles)] for i, name in enumerate(doublet_names)}

    for well_id, (x, y) in schedule.well_locations.items():
        kind = _classify_well(well_id)
        if kind is None:
            continue
        cell = mesh.locate_cell(float(x), float(y))
        color = _WARM_COLOR if kind == "warm" else _COLD_COLOR
        ls = style_of[well_id.rsplit("_", 1)[0]]
        ax.plot(times, field[:, cell], color=color, linestyle=ls, label=well_id)

    ax.axhline(sim.initial_temperature, color="grey", linestyle=":", linewidth=0.8, label="ambient")
    ax.set_xlabel("time [days]")
    ax.set_ylabel("temperature [degC]")
    ax.legend(loc="best", fontsize="small")
    return fig


def _plot_field(
    ax: Any,
    mesh: VertexGrid,
    field: np.ndarray,
    label: str,
    schedule: WellSchedule | None = None,
    cmap: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Any:
    """Render a per-cell scalar ``field`` on the mesh; return the mappable.

    The shared core of every spatial map: a ``PolyCollection`` over the cells,
    equal aspect, a colorbar labelled ``label``, and (optionally) the well overlay.
    ``xlim``/``ylim`` default to the mesh's own extent; when given (a zoomed view),
    the colour range is also rescaled to only the cells actually visible in that
    window, so a localised signal is not washed out by the whole-domain range.
    """
    from matplotlib.collections import PolyCollection

    fig = ax.figure
    polys = [mesh.vertices[cell] for cell in mesh.cell_vertices]
    collection = PolyCollection(polys, array=np.asarray(field), edgecolors="none", cmap=cmap)
    ax.add_collection(collection)
    x_bounds = xlim if xlim is not None else (float(mesh.vertices[:, 0].min()), float(mesh.vertices[:, 0].max()))
    y_bounds = ylim if ylim is not None else (float(mesh.vertices[:, 1].min()), float(mesh.vertices[:, 1].max()))
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    mask = _visible_mask(mesh, xlim, ylim)
    if mask is not None:
        visible = np.asarray(field, dtype=float)[mask]
        visible = visible[np.isfinite(visible)]
        if visible.size:
            collection.set_clim(float(visible.min()), float(visible.max()))
    fig.colorbar(collection, ax=ax, label=label)
    if schedule is not None:
        _overlay_wells(ax, schedule)
    return collection


def plot_snapshot(
    sim: SimulationResult,
    t: float,
    mesh: VertexGrid,
    ax: Any | None = None,
    variable: str = "temperature",
    schedule: WellSchedule | None = None,
    zoom: bool = False,
) -> Figure:
    """Spatial field of ``variable`` on the mesh at the time nearest ``t``.

    When a ``schedule`` is supplied, the doublet well locations are overlaid as
    markers (warm wells red, cold wells blue), each outlined in white so they
    stay legible over any field colour. ``zoom=True`` frames every well (see
    :func:`_zoom_limits`) instead of the whole domain: the useful view once the
    domain is large relative to the well cluster (e.g. a wide-domain convergence
    study), where the wells would otherwise be a speck in an empty map.
    """
    fig, ax = _figure_and_axes(ax)
    xlim, ylim = _zoom_limits(sim, mesh, schedule) if zoom else (None, None)
    _plot_field(ax, mesh, sim.snapshot(t, variable), label=variable, schedule=schedule, xlim=xlim, ylim=ylim)
    return fig


def plot_drawdown(
    sim: SimulationResult,
    t: float,
    mesh: VertexGrid,
    ax: Any | None = None,
    schedule: WellSchedule | None = None,
    zoom: bool = False,
) -> Figure:
    """Hydraulic-head drawdown field at time ``t`` (initial head minus head at ``t``).

    Positive = head decline (the extraction well's cone of depression); negative =
    buildup (the injection well's mound). Drawn with a diverging colormap centred
    on zero so the two signs are visually symmetric. ``zoom=True`` frames every well
    (see :func:`_zoom_limits`); the symmetric colour limit is then taken from the
    zoomed-in cells only, so a local cone of depression is not flattened by a much
    larger far-field regional-gradient signal elsewhere in the domain.
    """
    fig, ax = _figure_and_axes(ax)
    times = np.asarray(sim.timeseries.index, dtype=float)
    initial_head = _reference_head(sim, sim.snapshot(float(times[0]), "head"))
    drawdown = initial_head - sim.snapshot(float(t), "head")
    xlim, ylim = _zoom_limits(sim, mesh, schedule) if zoom else (None, None)
    collection = _plot_field(
        ax, mesh, drawdown, label="drawdown [m]", schedule=schedule, cmap="RdBu_r", xlim=xlim, ylim=ylim
    )
    mask = _visible_mask(mesh, xlim, ylim)
    visible_drawdown = drawdown[mask] if mask is not None else drawdown
    limit = float(np.max(np.abs(visible_drawdown))) or 1.0
    collection.set_clim(-limit, limit)
    return fig


def plot_arrival_time(
    sim: SimulationResult,
    mesh: VertexGrid,
    ax: Any | None = None,
    schedule: WellSchedule | None = None,
    tol: float = DEFAULT_PLUME_TOL,
    zoom: bool = False,
) -> Figure:
    """Spatial map of the first time ``|T - ambient| > tol`` at each cell.

    The spatial generalisation of the scalar ``breakthrough_time`` stat: shows *where* the
    thermal front has (and has not) reached, not only whether it reached the extraction well.
    Cells with no signal at all stay unfilled (NaN), rendered by the colormap's "bad" colour.
    ``zoom=True`` frames every well (see :func:`_zoom_limits`) instead of the whole domain.
    """
    fig, ax = _figure_and_axes(ax)
    field = arrival_time_map(sim, tol=tol)
    xlim, ylim = _zoom_limits(sim, mesh, schedule) if zoom else (None, None)
    _plot_field(ax, mesh, field, label="arrival time [days]", schedule=schedule, cmap="viridis", xlim=xlim, ylim=ylim)
    return fig


def plot_plume_extent(samples: list[PlumeSample], ax: Any | None = None) -> Figure:
    """Equivalent plume radius vs. time per doublet, analytical √t overlaid.

    Solid line/markers: modelled equivalent radius (``sqrt(area/pi)``) per doublet, one colour
    each. Dashed, same colour: the analytical ``thermal_front_radius`` trajectory; the gap
    between the two lines is ``radius_error`` made visible. Plume area is drawn on a twin axis
    for scale, since a radius doubling is an area quadrupling.
    """
    fig, ax = _figure_and_axes(ax)
    if not samples:
        ax.text(
            0.5, 0.5, "no plume data\n(no ATES doublets)",
            ha="center", va="center", transform=ax.transAxes, color="grey",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    table = pd.DataFrame(
        [
            {"time": s.time, "doublet": s.doublet, "radius": s.equivalent_radius,
             "analytical": s.analytical_radius, "area": s.area}
            for s in samples
        ]
    )
    ax_area = ax.twinx()
    for i, (name, sub) in enumerate(table.groupby("doublet", sort=False)):
        sub = sub.sort_values("time")
        color = f"C{i}"
        ax.plot(sub["time"], sub["radius"], color=color, marker="o", label=f"{name} (model)")
        ax.plot(sub["time"], sub["analytical"], color=color, linestyle="--",
                 label=f"{name} (analytical)")
        ax_area.plot(sub["time"], sub["area"], color=color, linestyle=":", alpha=0.5)

    ax.set_xlabel("time [days]")
    ax.set_ylabel("equivalent plume radius [m]")
    ax_area.set_ylabel("plume area [m2]")
    ax.legend(loc="best", fontsize="small")
    ax.set_title("thermal plume extent")
    return fig


def plot_efficiency(sim: SimulationResult, ax: Any | None = None) -> Figure:
    """Per-discharge-season thermal-recovery efficiency, grouped bars per well.

    Reads :attr:`SimulationResult.efficiency`; shows a placeholder when there are
    no ATES doublets (an empty efficiency table).
    """
    fig, ax = _figure_and_axes(ax)
    eff = sim.efficiency
    if eff.empty:
        ax.text(
            0.5, 0.5, "no recovery-efficiency data\n(no ATES doublets)",
            ha="center", va="center", transform=ax.transAxes, color="grey",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    seasons = list(eff.index)
    wells = list(eff.columns)
    x = np.arange(len(seasons))
    width = 0.8 / max(len(wells), 1)
    for i, well in enumerate(wells):
        kind = _classify_well(str(well))
        color = _WARM_COLOR if kind == "warm" else _COLD_COLOR if kind == "cold" else None
        offset = (i - (len(wells) - 1) / 2.0) * width
        ax.bar(x + offset, eff[well].to_numpy(), width=width, label=str(well), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seasons])
    ax.set_xlabel("discharge season")
    ax.set_ylabel("recovery efficiency [%]")
    ax.legend(loc="best", fontsize="small")
    return fig


def _reporting_layer(config: Config) -> int:
    """The transmissivity-weighted-highest aquifer layer index.

    Mirrors ``model.builder``'s ``_transmissivity_weights`` argmax exactly (weighting by
    a positive constant does not change which index is the max, and the even-split
    fallback for zero total transmissivity is equivalent to an all-zero argmax, which is
    index 0 either way), so the energy-balance plot is judged against the same layer
    ``analyse()``'s own stats use, without importing a private symbol across modules.
    """
    layers = config.model.aquifer.layers
    trans = [abs(float(lay.top) - float(lay.bottom)) * float(lay.hydraulic_conductivity) for lay in layers]
    return int(np.argmax(trans))


def plot_energy_balance(
    sim: SimulationResult, schedule: WellSchedule, config: Config
) -> Figure:
    """Per-doublet yearly warm/cold energy, plus the cumulative net line
    and the domain-wide ambient-drift trajectory on a second axis.

    Rebuilds the per-doublet-per-year table via ``analysis.energy.compute_energy_balance``
    from ``schedule``/``config`` directly (not from a value stored on ``sim``), so this
    stays correct even called standalone, without re-running ``analyse``. Bars diverge
    from zero: warm-season injected energy above the axis, cold-season injected energy
    below it, so a doublet's seasonal imbalance is visible as the bar's net offset from
    zero, matching the cumulative net-imbalance line drawn over them. The ambient-drift
    trajectory (right axis) needs ``sim.cell_areas`` (set when ``analyse`` was given a
    mesh); it is omitted with a note when unavailable.
    """
    layer = config.model.aquifer.layers[_reporting_layer(config)]
    ambient = float(layer.initial_temperature)
    rho_c_water = float(layer.heat_capacity_water) * float(layer.density_water)
    season_length_days = float(config.model.simulation.season_length_days)
    table = compute_energy_balance(schedule, rho_c_water, season_length_days, ambient)

    fig = Figure(figsize=(9.0, 5.0), layout="constrained")
    ax = fig.subplots()
    if table.empty:
        ax.text(
            0.5, 0.5, "no energy-balance data\n(no ATES doublets)",
            ha="center", va="center", transform=ax.transAxes, color="grey",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    doublets = list(dict.fromkeys(table["doublet"]))  # stable, first-seen order
    years = sorted(table["year"].unique())
    x = np.arange(len(years))
    width = 0.8 / max(len(doublets), 1)

    for i, name in enumerate(doublets):
        sub = table[table["doublet"] == name].set_index("year").reindex(years, fill_value=0.0)
        offset = (i - (len(doublets) - 1) / 2.0) * width
        warm_gj = sub["e_warm_injected"].to_numpy() / 1.0e9
        cold_gj = sub["e_cold_injected"].to_numpy() / 1.0e9
        ax.bar(x + offset, warm_gj, width=width, color=_WARM_COLOR,
               label="warm injected" if i == 0 else None)
        ax.bar(x + offset, -cold_gj, width=width, color=_COLD_COLOR,
               label="cold injected" if i == 0 else None)

    cumulative_total = (
        table.groupby("year")["net_imbalance"].sum().reindex(years, fill_value=0.0).cumsum()
        / 1.0e9
    )
    ax.plot(x, cumulative_total.to_numpy(), color="black", marker="o", linewidth=1.5,
            label="cumulative net imbalance")
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("simulated year")
    ax.set_ylabel("energy [GJ]")

    if sim.cell_areas.size and sim.cell_areas.size == sim.temperature.shape[1]:
        times = np.asarray(sim.times, dtype=float)
        initial_field = np.asarray(sim.temperature[0], dtype=float)
        drift_series = np.array(
            [
                np.average(np.asarray(sim.temperature[i], dtype=float) - initial_field, weights=sim.cell_areas)
                for i in range(len(times))
            ]
        )
        year_times = [(y + 1) * SEASONS_PER_YEAR * season_length_days for y in years]
        drift_at_year = np.array(
            [drift_series[int(np.argmin(np.abs(times - t)))] for t in year_times]
        )
        ax_drift = ax.twinx()
        ax_drift.plot(x, drift_at_year, color="tab:green", linestyle="--", marker="s",
                      label="ambient drift")
        ax_drift.set_ylabel("area-weighted ambient drift [degC]", color="tab:green")
        ax_drift.tick_params(axis="y", labelcolor="tab:green")

    ax.legend(loc="best", fontsize="small")
    ax.set_title("ATES seasonal energy balance")
    return fig


def plot_model_overview(
    sim: SimulationResult,
    mesh: VertexGrid,
    schedule: WellSchedule,
    t: float | None = None,
    gwf_field: str = "drawdown",
) -> Figure:
    """A single overview figure of the coupled GWF + GWE run.

    Three rows: row 1 is the flow model (GWF), row 2 the energy-transport model
    (GWE), and a full-width row 3 with per-season recovery efficiency. In the top
    two rows the left column is a spatial field at time ``t`` (default: the last
    reported time) and the right column is the matching timeseries. ``gwf_field``
    selects the GWF view: ``"drawdown"`` (initial-minus-current head, the default)
    or ``"head"`` (absolute head). Each panel reuses a standalone plot helper, so
    the dashboard tracks any improvement to them.
    """
    if gwf_field not in ("drawdown", "head"):
        raise ValueError(f"gwf_field must be 'drawdown' or 'head', got {gwf_field!r}")
    snap_t = float(sim.timeseries.index[-1]) if t is None else float(t)

    fig = Figure(figsize=(13.0, 14.0), layout="constrained")
    gs = fig.add_gridspec(3, 2)
    ax_head_map = fig.add_subplot(gs[0, 0])
    ax_head_ts = fig.add_subplot(gs[0, 1])
    ax_temp_map = fig.add_subplot(gs[1, 0])
    ax_temp_ts = fig.add_subplot(gs[1, 1])
    ax_eff = fig.add_subplot(gs[2, :])

    ts = sim.timeseries

    # --- GWF (flow) ---
    if gwf_field == "drawdown":
        plot_drawdown(sim, snap_t, mesh, ax=ax_head_map, schedule=schedule, zoom=True)
        ax_head_map.set_title(f"GWF: drawdown [m] @ t={snap_t:.0f} d")
        initial_head = _reference_head(sim, float(ts["head"].iloc[0]))
        ax_head_ts.plot(ts.index, initial_head - ts["head"], color=_COLD_COLOR)
        ax_head_ts.set_ylabel("drawdown [m]")
        ax_head_ts.set_title("GWF: drawdown at extraction well")
    else:
        plot_snapshot(sim, snap_t, mesh, ax=ax_head_map, variable="head", schedule=schedule)
        ax_head_map.set_title(f"GWF: hydraulic head [m] @ t={snap_t:.0f} d")
        ax_head_ts.plot(ts.index, ts["head"], color=_COLD_COLOR)
        ax_head_ts.set_ylabel("head [m]")
        ax_head_ts.set_title("GWF: head at extraction well")
    ax_head_ts.set_xlabel("time [days]")

    # --- GWE (energy transport) ---
    plot_snapshot(sim, snap_t, mesh, ax=ax_temp_map, variable="temperature", schedule=schedule)
    ax_temp_map.set_title(f"GWE: temperature [degC] @ t={snap_t:.0f} d")

    plot_doublet_temperatures(sim, mesh, schedule, ax=ax_temp_ts)
    ax_temp_ts.set_title("GWE: doublet well temperatures")

    # --- recovery efficiency ---
    plot_efficiency(sim, ax=ax_eff)
    ax_eff.set_title("GWE: thermal-recovery efficiency per discharge season")

    fig.suptitle("ATES model overview: GWF + GWE", fontsize="x-large")
    return fig


def _overlay_wells(ax: Any, schedule: WellSchedule) -> None:
    """Scatter warm/cold doublet wells, one legend entry per well kind."""
    seen: set[str] = set()
    for well_id, (x, y) in schedule.well_locations.items():
        kind = _classify_well(well_id)
        if kind is None:
            continue
        color = _WARM_COLOR if kind == "warm" else _COLD_COLOR
        label = f"{kind} well" if kind not in seen else None
        seen.add(kind)
        ax.scatter(
            float(x), float(y), s=60, marker="o", facecolors=color,
            edgecolors="white", linewidths=1.2, zorder=5, label=label,
        )
    if seen:
        ax.legend(loc="upper right", fontsize="small")

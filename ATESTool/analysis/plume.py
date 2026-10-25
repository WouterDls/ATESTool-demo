"""Spatial plume metrics + arrival-time map.

Where ``SimulationResult.stats`` reads only the extraction well's cell (``peak_temperature``,
``breakthrough_time``), this module is the spatial generalisation: how big is each doublet's
plume over time, how does that compare to the analytical thermal radius, where do two
doublets' plumes actually overlap, and, cell by cell, when did each one first feel the
thermal front. Built entirely on ``SimulationResult.snapshot`` / ``.times`` / a
:class:`~ATESTool.mesh.VertexGrid` / a :class:`~ATESTool.model.schedule.WellSchedule` (the
public seams) plus :func:`ATESTool.mesh.thermal_front_radius`, the same physics
``validation.analytical`` uses (imported from ``mesh``, not ``validation``: this package must
not depend on ``validation``, only the reverse).

``thermal_front_radius`` assumes continuous radial injection from a single well since t=0, so
it (and everything derived from it here, ``analytical_radius``, ``radius_error``, the overlap
test) is only a faithful benchmark while a doublet's warm well is in its first, uninterrupted
charge season. Outside that window the analytical columns still compute a number, but it no
longer describes the modelled physics (the plume has started retreating); restrict ``times``
to the first charge season when the comparison matters, exactly as
the dedicated radial-injection verification does for its analytical comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ATESTool.config import Config
from ATESTool.mesh import VertexGrid, thermal_front_radius
from ATESTool.model.schedule import WellSchedule

from ATESTool.analysis.results import SimulationResult

#: Default ``|T - T_ambient|`` threshold [degC] marking "inside the plume", matches
#: SimulationResult's own default ``breakthrough_tol``.
DEFAULT_PLUME_TOL = 0.5
#: A doublet's own plume area/radius is measured only within this multiple of its analytical
#: thermal radius of its warm well, so a neighbouring doublet's plume is not counted toward it.
#: Generous (2x) so a genuinely over-sized modelled plume is not clipped: that mismatch is
#: exactly what ``radius_error`` should surface, not hide.
_CAPTURE_RADIUS_FACTOR = 2.0


@dataclass(frozen=True)
class PlumeSample:
    """One doublet's plume extent at one reported time, within its own capture radius."""

    time: float
    doublet: str
    area: float  # m^2: sum of cell areas with |T-ambient| > tol, within the capture radius
    equivalent_radius: float  # sqrt(area / pi)
    analytical_radius: float  # thermal_front_radius at this time, this doublet's own flow_rate
    radius_error: float  # equivalent_radius - analytical_radius


@dataclass(frozen=True)
class OverlapSample:
    """Two doublets' thermal interaction at one reported time.

    ``overlap_area``/``n_cells`` count cells that are simultaneously (a) thermally disturbed
    (``|T-ambient| > tol``) and (b) within *both* doublets' own analytical thermal radius of
    their respective warm wells, i.e. cells either plume could plausibly have reached. A
    non-zero overlap is direct evidence the two ATES systems are thermally interacting.
    """

    time: float
    doublet_a: str
    doublet_b: str
    overlap_area: float
    n_cells: int


def _doublet_warm_wells(schedule: WellSchedule) -> dict[str, tuple[float, float]]:
    """``{doublet_name: (x, y)}`` of each doublet's warm well, the plume's injection point."""
    return {
        well_id[: -len("_warm")]: xy
        for well_id, xy in schedule.well_locations.items()
        if well_id.endswith("_warm")
    }


def _layer_thermal_properties(config: Config) -> tuple[float, float, float, float]:
    """``(thickness, porosity, volumetric_heat_water, volumetric_heat_solid)`` of layer 0.

    Mirrors ``validation.analytical.compare_thermal_front``'s own convention of reading the
    first aquifer layer for the analytical benchmark.
    """
    layer = config.model.aquifer.layers[0]
    thickness = abs(float(layer.top) - float(layer.bottom))
    porosity = float(layer.porosity)
    vol_heat_water = float(layer.density_water) * float(layer.heat_capacity_water)
    vol_heat_solid = float(layer.density_solid) * float(layer.heat_capacity_solid)
    return thickness, porosity, vol_heat_water, vol_heat_solid


def _query_times(sim: SimulationResult, times: Sequence[float] | None) -> np.ndarray:
    """``times`` snapped to the nearest reported time, or every reported time when ``None``."""
    reported = np.asarray(sim.times, dtype=float)
    if times is None:
        return reported
    requested = np.asarray(list(times), dtype=float)
    return reported[np.argmin(np.abs(reported[:, None] - requested[None, :]), axis=0)]


def plume_extent(
    sim: SimulationResult,
    mesh: VertexGrid,
    schedule: WellSchedule,
    config: Config,
    times: Sequence[float] | None = None,
    tol: float = DEFAULT_PLUME_TOL,
) -> list[PlumeSample]:
    """Per ATES doublet, per requested time: plume area/radius and the analytical comparison.

    ``times`` defaults to every reported time (snapped to the nearest one when given
    explicitly). A doublet with no warm well in ``schedule`` (should not happen for a
    doublet-derived schedule) is skipped.
    """
    warm_wells = _doublet_warm_wells(schedule)
    thickness, porosity, vol_heat_water, vol_heat_solid = _layer_thermal_properties(config)
    query_times = _query_times(sim, times)
    centroids = mesh.centroids

    samples: list[PlumeSample] = []
    for doublet in config.model.doublets:
        if doublet.name not in warm_wells:
            continue
        x, y = warm_wells[doublet.name]
        dist = np.hypot(centroids[:, 0] - x, centroids[:, 1] - y)
        q = float(doublet.flow_rate)
        for t in query_times:
            field = sim.snapshot(float(t), "temperature")
            analytical = thermal_front_radius(
                q=q, t=float(t), thickness=thickness, porosity=porosity,
                volumetric_heat_water=vol_heat_water, volumetric_heat_solid=vol_heat_solid,
            )
            capture = _CAPTURE_RADIUS_FACTOR * analytical
            mask = (np.abs(field - sim.initial_temperature) > tol) & (dist <= capture)
            area = float(np.sum(mesh.areas[mask]))
            radius = float(np.sqrt(area / np.pi)) if area > 0.0 else 0.0
            samples.append(
                PlumeSample(
                    time=float(t), doublet=doublet.name, area=area,
                    equivalent_radius=radius, analytical_radius=analytical,
                    radius_error=radius - analytical,
                )
            )
    return samples


def plume_overlap(
    sim: SimulationResult,
    mesh: VertexGrid,
    schedule: WellSchedule,
    config: Config,
    times: Sequence[float] | None = None,
    tol: float = DEFAULT_PLUME_TOL,
) -> list[OverlapSample]:
    """Per doublet pair, per requested time: thermally-disturbed cells reachable by both.

    Only doublets with a warm well in ``schedule`` are considered; each unordered pair is
    reported once.
    """
    warm_wells = _doublet_warm_wells(schedule)
    thickness, porosity, vol_heat_water, vol_heat_solid = _layer_thermal_properties(config)
    query_times = _query_times(sim, times)
    centroids = mesh.centroids
    doublets = [d for d in config.model.doublets if d.name in warm_wells]

    samples: list[OverlapSample] = []
    for i in range(len(doublets)):
        for j in range(i + 1, len(doublets)):
            d_a, d_b = doublets[i], doublets[j]
            xa, ya = warm_wells[d_a.name]
            xb, yb = warm_wells[d_b.name]
            dist_a = np.hypot(centroids[:, 0] - xa, centroids[:, 1] - ya)
            dist_b = np.hypot(centroids[:, 0] - xb, centroids[:, 1] - yb)
            for t in query_times:
                field = sim.snapshot(float(t), "temperature")
                r_a = thermal_front_radius(
                    q=float(d_a.flow_rate), t=float(t), thickness=thickness, porosity=porosity,
                    volumetric_heat_water=vol_heat_water, volumetric_heat_solid=vol_heat_solid,
                )
                r_b = thermal_front_radius(
                    q=float(d_b.flow_rate), t=float(t), thickness=thickness, porosity=porosity,
                    volumetric_heat_water=vol_heat_water, volumetric_heat_solid=vol_heat_solid,
                )
                mask = (
                    (np.abs(field - sim.initial_temperature) > tol)
                    & (dist_a <= r_a) & (dist_b <= r_b)
                )
                samples.append(
                    OverlapSample(
                        time=float(t), doublet_a=d_a.name, doublet_b=d_b.name,
                        overlap_area=float(np.sum(mesh.areas[mask])), n_cells=int(mask.sum()),
                    )
                )
    return samples


def arrival_time_map(sim: SimulationResult, tol: float = DEFAULT_PLUME_TOL) -> np.ndarray:
    """Per-cell first time ``|T - initial_temperature| > tol``; ``NaN`` where it never happens.

    The spatial generalisation of ``SimulationResult.stats``'s scalar ``breakthrough_time``
    (which reads only the extraction well's cell): this is the same test over every cell.
    """
    times = np.asarray(sim.times, dtype=float)
    temperature = np.asarray(sim.temperature, dtype=float)
    deviation = np.abs(temperature - sim.initial_temperature) > tol  # (n_times, n_cells)

    n_cells = temperature.shape[1]
    result = np.full(n_cells, np.nan)
    ever_broken = deviation.any(axis=0)
    first_index = np.argmax(deviation, axis=0)  # first True per column; 0 if all False
    result[ever_broken] = times[first_index[ever_broken]]
    return result

"""Analytical benchmark solutions for verifying mesh/model accuracy (meshing primer §6.2,
"Analytical benchmarks").

These closed-form solutions bound the *absolute* error of a run against known physics:
the check the mesh quality gate and grid-convergence study cannot give. They apply to
idealised setups (single well, homogeneous confined aquifer, radial symmetry, distant
boundaries); compare them to a model only where those assumptions approximately hold
(e.g. near a well, early in time, away from other wells and boundaries).

Do not gate the same late-time finite-domain run against both Theis and Thiem:
their boundary/time assumptions differ. The historical V04 mismatch and the required
separate benchmark designs are documented in ``docs/validation_decisions.md``.

- ``thiem_drawdown``: steady-state radial flow to/from a well (Thiem 1906).
- ``theis_drawdown``: transient radial flow (Theis 1935).
- ``thermal_front_radius``: the radius of the thermal front for radial injection, from an
  energy balance with thermal retardation (the standard ATES "thermal radius").
- ``thermal_front_trajectory``: that radius at a series of times, ``R_th(t) ∝ √t``; the
  **transient** heat-transport benchmark (the front's *arrival over time*, which the time step
  controls, so it validates the temporal discretisation, not just the mesh).
- ``radial_profile``: extract a per-cell field as (distance-from-well, value) for overlay.
- ``compare_thermal_front``: the model-vs-analytical comparison itself: locate the modelled
  front on a radial profile and report its relative error against ``thermal_front_trajectory``.

References: Thiem (1906); Theis (1935); for the heat front see the energy-balance / thermal
radius used in ATES design (e.g. Bloemendal & Hartog 2018).

Note on Lauwerier (1955): that classic solution adds vertical heat **loss** by conduction into
the impermeable beds confining the aquifer. The current ATESTool model is 2-D *areal* (the
modelled domain is the aquifer; there are no over-/under-lying conducting beds), so it has no
such loss term; benchmarking it against Lauwerier-with-loss would be inconsistent (the modelled
front would systematically lead it). The energy-balance ``thermal_front_trajectory`` is the apt
transient benchmark here; reproducing Lauwerier would require explicitly modelling confining
layers (a low-K conduction-only cap and base around a flowing middle layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ATESTool.mesh import VertexGrid
from ATESTool.mesh.thermal import thermal_front_radius as thermal_front_radius  # re-export

if TYPE_CHECKING:  # import-time cycle guard: validation must not pull in analysis
    from ATESTool.analysis import SimulationResult
    from ATESTool.config import Config

Point = tuple[float, float]

#: Only cells within this radius of the well are searched for the front. Keeps the search
#: inside the region where radial symmetry approximately holds (away from the partner well
#: and the domain boundary), a verification knob, not a config value.
DEFAULT_SEARCH_RADIUS = 80.0


def thiem_drawdown(
    r: np.ndarray | float,
    q: float,
    transmissivity: float,
    radius_of_influence: float,
) -> np.ndarray:
    """Steady-state Thiem drawdown ``s(r) = Q/(2πT)·ln(R/r)`` for a confined aquifer.

    ``q`` is the pumping rate (positive = extraction → positive drawdown; negative =
    injection → mounding). ``s`` is zero at ``r = radius_of_influence`` ``R`` and rises
    toward the well. Units must be consistent (e.g. m, m³/day, m²/day).
    """
    r_arr = np.asarray(r, dtype=float)
    return q / (2.0 * np.pi * transmissivity) * np.log(radius_of_influence / r_arr)


def theis_drawdown(
    r: np.ndarray | float,
    t: float,
    q: float,
    transmissivity: float,
    storativity: float,
) -> np.ndarray:
    """Transient Theis drawdown ``s = Q/(4πT)·W(u)``, ``u = r²S/(4Tt)``.

    ``W(u)`` is the Theis well function, here the exponential integral ``E1(u)``. Sign
    convention for ``q`` matches :func:`thiem_drawdown`.
    """
    from scipy.special import exp1

    r_arr = np.asarray(r, dtype=float)
    u = r_arr**2 * storativity / (4.0 * transmissivity * t)
    return q / (4.0 * np.pi * transmissivity) * np.asarray(exp1(u), dtype=float)


def thermal_front_trajectory(
    q: float,
    times: np.ndarray | float,
    thickness: float,
    porosity: float,
    volumetric_heat_water: float,
    volumetric_heat_solid: float,
) -> np.ndarray:
    """Thermal-front radius at each time in ``times``: :func:`thermal_front_radius` vectorised.

    The radial energy balance gives ``R_th(t) = sqrt( Q·t·(ρc)_w / (π·b·(ρc)_aq) ) ∝ √t``. This
    is the *transient* heat-transport benchmark: the front's position **as a function of time** is
    governed by the time stepping, so a model that tracks ``R_th(t)`` over several snapshots is
    validating its temporal accuracy (not just its mesh, which the single-time front check probes).
    See the module docstring on why this, not Lauwerier (1955), is the apt benchmark for the
    2-D areal model. ``q`` is the injection rate (> 0); ``q·t`` has units of volume.
    """
    vol_heat_aquifer = porosity * volumetric_heat_water + (1.0 - porosity) * volumetric_heat_solid
    t = np.asarray(times, dtype=float)
    return np.sqrt(q * t * volumetric_heat_water / (np.pi * thickness * vol_heat_aquifer))


def radial_profile(
    field: np.ndarray, mesh: VertexGrid, center: Point
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell ``field`` as ``(distance-from-center, value)``, sorted by distance.

    Use to overlay a modelled head/drawdown/temperature field against an analytical radial
    solution centred on a well.
    """
    cx, cy = float(center[0]), float(center[1])
    dist = np.hypot(mesh.centroids[:, 0] - cx, mesh.centroids[:, 1] - cy)
    order = np.argsort(dist)
    return dist[order], np.asarray(field, dtype=float)[order]


@dataclass
class FrontSample:
    """The modelled and analytical thermal-front radius at one reported time."""

    time: float
    model_radius: float
    analytical_radius: float

    @property
    def rel_error(self) -> float:
        """Relative error of the modelled radius against the analytical one."""
        if self.analytical_radius == 0.0:
            return float("nan")
        return abs(self.model_radius - self.analytical_radius) / self.analytical_radius


@dataclass
class AnalyticalResult:
    """Outcome of a thermal-front benchmark (samples ordered early -> late)."""

    samples: list[FrontSample]
    well: Point
    midpoint_temperature: float
    search_radius: float

    @property
    def max_rel_error(self) -> float:
        """Worst relative error across the sampled times."""
        return max((s.rel_error for s in self.samples), default=float("nan"))

    @property
    def growth_ratio(self) -> float:
        """Modelled front growth, last / first sample (``∝ √t`` if transport is on track)."""
        return _ratio([s.model_radius for s in self.samples])

    @property
    def analytical_growth_ratio(self) -> float:
        """The same ratio for the analytical trajectory: what ``growth_ratio`` should match."""
        return _ratio([s.analytical_radius for s in self.samples])

    def summary(self) -> str:
        lines = [
            f"thermal-front benchmark (well at {self.well[0]:g}, {self.well[1]:g}; "
            f"midpoint T={self.midpoint_temperature:.3f} degC; search r<{self.search_radius:g} m):"
        ]
        for s in self.samples:
            lines.append(
                f"  t={s.time:8.2f} d  model={s.model_radius:7.3f} m  "
                f"analytical={s.analytical_radius:7.3f} m  error={s.rel_error:.2%}"
            )
        lines.append(f"  max relative error: {self.max_rel_error:.2%}")
        if len(self.samples) >= 2:
            lines.append(
                f"  growth ratio (last/first): model={self.growth_ratio:.3f}  "
                f"analytical={self.analytical_growth_ratio:.3f}"
            )
        return "\n".join(lines)


def _ratio(values: list[float]) -> float:
    """``values[-1] / values[0]``, or NaN when undefined."""
    if len(values) < 2 or values[0] == 0.0:
        return float("nan")
    return values[-1] / values[0]


def compare_thermal_front(
    sim: SimulationResult,
    mesh: VertexGrid,
    config: Config,
    times: Sequence[float],
    doublet_index: int = 0,
    search_radius: float = DEFAULT_SEARCH_RADIUS,
) -> AnalyticalResult:
    """Locate the modelled thermal front at each time and compare it to the analytical radius.

    The front is taken as the first radius (working outward from the warm well) where the
    temperature drops below the midpoint of the ambient and injection temperatures: the
    modelled front is a smeared contour, not a sharp step, so the midpoint is the natural
    stand-in for the analytical discontinuity. Each requested time is snapped to the nearest
    reported output time.

    Raises ``ValueError`` if no front is found within ``search_radius`` at some time: that
    means the plume has not arrived (too early), has already passed (too late), or the run
    produced no thermal signal at all.
    """
    if not times:
        raise ValueError("compare_thermal_front needs at least one time")

    doublet = config.model.doublets[doublet_index]
    layer = config.model.aquifer.layers[0]
    well: Point = (float(doublet.warm_well[0]), float(doublet.warm_well[1]))
    midpoint = 0.5 * (float(layer.initial_temperature) + float(doublet.warm_injection_temperature))
    reported = np.asarray(sim.times, dtype=float)

    actual_times: list[float] = []
    model_radii: list[float] = []
    for t_req in times:
        t_actual = float(reported[int(np.argmin(np.abs(reported - float(t_req))))])
        r, temp = radial_profile(sim.snapshot(t_actual, "temperature"), mesh, well)
        near = r < search_radius
        crossing = np.where(temp[near] < midpoint)[0]
        if not crossing.size:
            raise ValueError(
                f"no thermal front found within r<{search_radius:g} m of the warm well "
                f"at t={t_actual:.0f} d"
            )
        actual_times.append(t_actual)
        model_radii.append(float(r[near][crossing[0]]))

    analytical = thermal_front_trajectory(
        q=float(doublet.flow_rate),
        times=np.array(actual_times),
        thickness=abs(float(layer.top) - float(layer.bottom)),
        porosity=float(layer.porosity),
        volumetric_heat_water=float(layer.density_water) * float(layer.heat_capacity_water),
        volumetric_heat_solid=float(layer.density_solid) * float(layer.heat_capacity_solid),
    )

    samples = [
        FrontSample(time=t, model_radius=m, analytical_radius=float(a))
        for t, m, a in zip(actual_times, model_radii, analytical)
    ]
    return AnalyticalResult(
        samples=samples,
        well=well,
        midpoint_temperature=midpoint,
        search_radius=search_radius,
    )

"""``SimulationResult``: the single seam that handles all FloPy output parsing.

``analyse`` reads the MODFLOW 6 head and temperature output files and returns a
``SimulationResult``, which exposes ``.timeseries``, ``.snapshot(t)`` and ``.stats`` as its
core entry points, plus ``.observation_timeseries(points, mesh)`` for user-configured
monitoring points. Downstream code (plots, validation) touches only these; it
never opens an output file itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ATESTool.analysis.efficiency import compute_efficiency, compute_energy_table
from ATESTool.analysis.energy import compute_energy_balance
from ATESTool.config import Config, ObservationPoint
from ATESTool.mesh import Domain, VertexGrid
from ATESTool.model.builder import _ModelPaths
from ATESTool.model.runner import _RawOutput
from ATESTool.model.schedule import build_well_schedule

#: Temperature deviation (degC) from the initial value that counts as thermal
#: breakthrough at the extraction well.
_BREAKTHROUGH_TOL = 0.5


@dataclass
class SimulationResult:
    """The result of one model run.

    Holds the full per-cell, per-time head and temperature fields. ``head`` and
    ``temperature`` are ``(n_times, n_cells)`` arrays; ``extraction_cell`` indexes
    the column observed for the timeseries and stats.
    """

    times: np.ndarray
    head: np.ndarray
    temperature: np.ndarray
    extraction_cell: int
    initial_temperature: float
    #: The (uniform) GWF initial-condition head: the true pre-pumping reference for
    #: drawdown. NaN when unset (e.g. synthetic results in tests), in which case
    #: ``stats`` falls back to the first reported head.
    initial_head: float = field(default=math.nan)
    breakthrough_tol: float = field(default=_BREAKTHROUGH_TOL)
    #: Per-discharge-season thermal-recovery efficiency [%]; index = season,
    #: columns = doublet well ids. Empty when there are no ATES doublets.
    efficiency: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Tidy per-well, per-discharge-season energy table: columns ``season``,
    #: ``well``, ``e_injected``, ``e_extracted`` [J], ``efficiency`` [%], the same rows
    #: ``efficiency`` is pivoted from, in long form. See ``analysis.efficiency.compute_energy_table``.
    energy_balance: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Per-cell mesh areas (aligned with ``temperature``/``head`` columns), for the
    #: area-weighted ``ambient_drift`` stat. Empty when unset (e.g. synthetic
    #: results in tests), in which case ``stats`` reports ``ambient_drift`` as NaN.
    cell_areas: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Cumulative net doublet energy imbalance over the whole run [GJ, gigajoules]: the
    #: grand total of ``analysis.energy.compute_energy_balance``'s ``net_imbalance``
    #: column, converted from J to GJ. NaN when unset (e.g. synthetic results in tests).
    net_energy_imbalance: float = field(default=math.nan)
    #: Boundary-influence scalars: max ``|T - ambient|`` and max ``|h - h_initial|``
    #: over the domain's constant-head boundary cells at the final time. The thermal one
    #: diagnoses whether the plume has reached the edge (domain too small); see
    #: ``analysis.boundary``. NaN when unset (e.g. synthetic results, or ``analyse`` called
    #: without a mesh).
    boundary_temperature_influence: float = field(default=math.nan)
    boundary_head_influence: float = field(default=math.nan)

    @property
    def timeseries(self) -> pd.DataFrame:
        """Temperature and hydraulic head at the extraction well, indexed by time."""
        return pd.DataFrame(
            {
                "temperature": self.temperature[:, self.extraction_cell],
                "head": self.head[:, self.extraction_cell],
            },
            index=pd.Index(np.asarray(self.times, dtype=float), name="time"),
        )

    def snapshot(self, t: float, variable: str = "temperature") -> np.ndarray:
        """Full-grid spatial field of ``variable`` at the time nearest ``t``."""
        if variable not in ("temperature", "head"):
            raise ValueError(f"variable must be 'temperature' or 'head', got {variable!r}")
        idx = int(np.argmin(np.abs(np.asarray(self.times, dtype=float) - t)))
        field_array = self.temperature if variable == "temperature" else self.head
        return np.asarray(field_array[idx])

    def observation_timeseries(
        self,
        points: Sequence[ObservationPoint],
        mesh: VertexGrid,
        variable: str = "temperature",
    ) -> pd.DataFrame:
        """``variable`` at each named observation point, indexed by time.

        One column per point (named after ``point.name``); each point's nearest mesh cell is
        located once via ``mesh.locate_cell`` and its full timeseries read from the same
        per-cell field ``.timeseries`` uses for the extraction well. Returns a column-less
        (but still time-indexed) DataFrame when ``points`` is empty.
        """
        if variable not in ("temperature", "head"):
            raise ValueError(f"variable must be 'temperature' or 'head', got {variable!r}")
        field_array = self.temperature if variable == "temperature" else self.head
        index = pd.Index(np.asarray(self.times, dtype=float), name="time")
        data = {p.name: field_array[:, mesh.locate_cell(float(p.x), float(p.y))] for p in points}
        return pd.DataFrame(data, index=index)

    @property
    def stats(self) -> dict[str, float]:
        """Derived scalars: peak extraction temperature, max drawdown, breakthrough time,
        recovery efficiency, energy-balance and boundary-influence scalars (see the
        matching attribute docstrings above for units and what each one diagnoses)."""
        well_temp = np.asarray(self.temperature[:, self.extraction_cell], dtype=float)
        well_head = np.asarray(self.head[:, self.extraction_cell], dtype=float)
        times = np.asarray(self.times, dtype=float)

        peak_temperature = float(np.max(well_temp))
        # Drawdown is referenced to the true pre-pumping IC head, not the first
        # *reported* head (which is already end-of-first-timestep and drifts with dt).
        reference_head = well_head[0] if math.isnan(self.initial_head) else self.initial_head
        max_drawdown = float(np.max(reference_head - well_head))

        deviation = np.abs(well_temp - self.initial_temperature)
        broken = deviation > self.breakthrough_tol
        breakthrough_time = float(times[int(np.argmax(broken))]) if bool(broken.any()) else float("nan")

        mean_efficiency = float(np.nanmean(self.efficiency.values)) if self.efficiency.size else float("nan")

        # Area-weighted mean temperature change across the whole domain, true initial
        # condition -> end time. The first saved field is already the end of time step 1
        # and must not be substituted for the configured initial condition.
        if self.cell_areas.size and self.cell_areas.size == self.temperature.shape[1]:
            delta_t = np.asarray(self.temperature[-1], dtype=float) - self.initial_temperature
            ambient_drift = float(np.average(delta_t, weights=self.cell_areas))
        else:
            ambient_drift = float("nan")

        return {
            "peak_temperature": peak_temperature,
            "max_drawdown": max_drawdown,
            "breakthrough_time": breakthrough_time,
            "mean_recovery_efficiency": mean_efficiency,
            "net_energy_imbalance": self.net_energy_imbalance,
            "ambient_drift": ambient_drift,
            "boundary_temperature_influence": self.boundary_temperature_influence,
            "boundary_head_influence": self.boundary_head_influence,
        }


def _read_disv_series(
    path: str, text: str | None = None, nlay: int = 1, layer: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Read a MODFLOW 6 DISV cell-by-cell head/temperature binary file.

    The raw field is ``(nlay, ncpl)`` per time step; ``layer`` selects the reporting
    layer so ``data`` is reduced to ``(n_times, ncpl)`` over the 2-D mesh.
    """
    import flopy

    head_file = flopy.utils.HeadFile(path) if text is None else flopy.utils.HeadFile(path, text=text)
    times = np.asarray(head_file.get_times(), dtype=float)
    data = np.array([np.asarray(head_file.get_data(totim=t)).reshape(nlay, -1)[layer] for t in times])
    return times, data


def analyse(
    model_paths: _ModelPaths,
    raw_output: _RawOutput,
    config: Config,
    mesh: VertexGrid | None = None,
) -> SimulationResult:
    """Parse the run's output files into a :class:`SimulationResult`.

    ``mesh`` is optional and only needed for the area-weighted ``ambient_drift`` stat;
    when omitted, ``ambient_drift`` reports NaN rather than an unweighted (and therefore
    misleading, on a Voronoi mesh with wildly varying cell sizes) average.
    """
    nlay = model_paths.n_layers
    layer = model_paths.extraction_layer
    times, head = _read_disv_series(str(raw_output.head_file), nlay=nlay, layer=layer)
    _, temperature = _read_disv_series(
        str(raw_output.temperature_file), text="TEMPERATURE", nlay=nlay, layer=layer
    )
    lay = config.model.aquifer.layers[layer]
    ambient = float(lay.initial_temperature)
    rho_c_water = float(lay.heat_capacity_water) * float(lay.density_water)
    season_length_days = float(config.model.simulation.season_length_days)
    # The uniform GWF initial condition (mirrors builder.build_model's own strt_head).
    chd = config.model.chd_heads
    initial_head = (float(chd.north) + float(chd.south) + float(chd.east) + float(chd.west)) / 4.0
    schedule = build_well_schedule(config)
    efficiency = compute_efficiency(
        temperature,
        times,
        model_paths.well_cells,
        schedule,
        rho_c_water=rho_c_water,
        season_length_days=season_length_days,
        ambient=ambient,
    )
    energy_balance_table = compute_energy_table(
        temperature,
        times,
        model_paths.well_cells,
        schedule,
        rho_c_water=rho_c_water,
        season_length_days=season_length_days,
        ambient=ambient,
    )
    energy_table = compute_energy_balance(
        schedule, rho_c_water=rho_c_water, season_length_days=season_length_days, ambient=ambient
    )
    net_energy_imbalance = (
        float(energy_table["net_imbalance"].sum()) / 1.0e9 if not energy_table.empty else 0.0
    )
    result = SimulationResult(
        times=times,
        head=head,
        temperature=temperature,
        extraction_cell=model_paths.extraction_cell,
        initial_temperature=ambient,
        initial_head=initial_head,
        efficiency=efficiency,
        energy_balance=energy_balance_table,
        cell_areas=mesh.areas if mesh is not None else np.empty(0),
        net_energy_imbalance=net_energy_imbalance,
    )
    if mesh is not None:
        # Boundary-influence scalars need the mesh + domain to locate the boundary cells;
        # like ambient_drift they are only meaningful with a real mesh, so stay NaN otherwise.
        from ATESTool.analysis.boundary import boundary_influence

        influence = boundary_influence(result, mesh, Domain.from_config(config))
        result.boundary_temperature_influence = influence.max_temperature_change
        result.boundary_head_influence = influence.max_head_change
    return result

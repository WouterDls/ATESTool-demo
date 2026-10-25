"""ATES thermal-recovery efficiency.

For each ATES well the recovery efficiency of a discharge season is the energy extracted
divided by the energy injected in the preceding charge season::

    eta = E_extracted / E_injected x 100 %,   P = Q . rho_c_water . (T - T_ambient),
    E_injected  = |Q| . rho_c_water . (T_inject - T_ambient) . season_length,
    E_extracted = |Q| . rho_c_water . integral over the discharge season of (T_well - T_ambient) dt.

All information (seasons, flow rates, the heat-capacity constant) flows from the
:class:`WellSchedule` and the config; nothing is hard-coded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ATESTool.analysis.integration import integrate_end_step
from ATESTool.model.schedule import WellSchedule


@dataclass(frozen=True)
class _WellSeasonEnergy:
    """One well's energy in one discharge season with a preceding charge."""

    season: int
    well: str
    e_injected: float
    e_extracted: float
    efficiency: float


def _well_season_energies(
    temperature: np.ndarray,
    times: np.ndarray,
    well_cells: dict[str, int],
    schedule: WellSchedule,
    rho_c_water: float,
    season_length_days: float,
    ambient: float,
) -> list[_WellSeasonEnergy]:
    """Per well, per discharge season with a preceding charge: e_injected, e_extracted, efficiency.

    The shared core of :func:`compute_efficiency` (which pivots these rows to a season x well
    percentage table) and :func:`compute_energy_table` (the same rows as a tidy long-form
    table). One season-integral computation, two presentations. A discharge with no
    preceding charge season is omitted.
    """
    times = np.asarray(times, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    length = float(season_length_days)

    rows: list[_WellSeasonEnergy] = []
    for well_id, cell in well_cells.items():
        well_temp = temperature[:, cell]
        energy_injected: dict[int, float] = {}
        discharge_periods: list[tuple[int, float]] = []
        for entry in schedule.entries:
            if entry.well_id != well_id:
                continue
            q = abs(float(entry.flow_rate))
            if entry.injection_temperature is not None:  # charging (injecting)
                delta = float(entry.injection_temperature) - ambient
                energy_injected[entry.stress_period] = q * rho_c_water * delta * length
            else:  # discharging (extracting)
                discharge_periods.append((entry.stress_period, q))

        for period, q in discharge_periods:
            prior_charges = [c for c in energy_injected if c < period]
            if not prior_charges:
                continue
            e_inj = energy_injected[max(prior_charges)]
            if e_inj == 0.0:
                continue
            integral = integrate_end_step(
                times,
                well_temp - ambient,
                start=period * length,
                end=(period + 1) * length,
            )
            e_extr = q * rho_c_water * integral
            rows.append(
                _WellSeasonEnergy(
                    season=period,
                    well=well_id,
                    e_injected=e_inj,
                    e_extracted=e_extr,
                    efficiency=e_extr / e_inj * 100.0,
                )
            )
    return rows


def compute_efficiency(
    temperature: np.ndarray,
    times: np.ndarray,
    well_cells: dict[str, int],
    schedule: WellSchedule,
    rho_c_water: float,
    season_length_days: float,
    ambient: float,
) -> pd.DataFrame:
    """Per-discharge-season thermal-recovery efficiency for each ATES well.

    ``temperature`` is ``(n_times, ncpl)`` for the reporting layer; ``well_cells``
    maps each well id to its column. Returns a DataFrame indexed by the discharge
    season (stress period) with one column per well id, values in percent. A
    discharge with no preceding charge season is omitted.
    """
    rows = _well_season_energies(
        temperature, times, well_cells, schedule, rho_c_water, season_length_days, ambient
    )
    pivot: dict[int, dict[str, float]] = {}
    for r in rows:
        pivot.setdefault(r.season, {})[r.well] = r.efficiency
    table = pd.DataFrame(pivot).T.sort_index()
    table.index.name = "season"
    return table


def compute_energy_table(
    temperature: np.ndarray,
    times: np.ndarray,
    well_cells: dict[str, int],
    schedule: WellSchedule,
    rho_c_water: float,
    season_length_days: float,
    ambient: float,
) -> pd.DataFrame:
    """Tidy per-well, per-discharge-season energy table.

    Columns: ``season`` (the discharge stress period), ``well`` (the well id),
    ``e_injected``/``e_extracted`` [J], and ``efficiency`` [%]: the same season-integral
    quantities :func:`compute_efficiency` computes, in long form instead of pivoted to a
    season x well percentage table. Not to be confused with
    ``analysis.energy.compute_energy_balance``, which is a *different* table (per doublet,
    per simulated year, warm/cold injected energy) built directly from the schedule's charge
    seasons for the seasonal-imbalance check; this one is per well, per discharge season, and
    needs the simulated extraction temperature, so it lives beside :func:`compute_efficiency`
    rather than in ``analysis.energy``.
    """
    rows = _well_season_energies(
        temperature, times, well_cells, schedule, rho_c_water, season_length_days, ambient
    )
    columns = ["season", "well", "e_injected", "e_extracted", "efficiency"]
    table = pd.DataFrame([asdict(r) for r in rows], columns=columns)
    if table.empty:
        return table
    return table.sort_values(["season", "well"]).reset_index(drop=True)

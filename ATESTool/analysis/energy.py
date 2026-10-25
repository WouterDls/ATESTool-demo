"""Seasonal warm/cold energy balance for ATES doublets.

With one ``flow_rate`` per doublet and two equal-length seasons, the *volumes*
injected/extracted are balanced by construction, but the *energy* need not be::

    E_warm_injected (summer charge)  = Q . rho_c_w . (T_warm - T_amb) . L
    E_cold_injected (winter charge)  = Q . rho_c_w . (T_amb - T_cold) . L
    net annual load                  = E_warm_injected - E_cold_injected
                                      = Q . rho_c_w . L . (T_warm + T_cold - 2*T_amb)

A doublet is in energy balance iff ``T_warm + T_cold == 2*T_amb``. This module makes that
measurable: nothing else in the pipeline surfaces it (``analysis.efficiency.compute_efficiency``
normalises injected against extracted *within* one doublet and says nothing about a net
year-on-year load).

:func:`compute_energy_balance` builds the per-doublet, per-year table directly from a
:class:`~ATESTool.model.schedule.WellSchedule`, not by re-reading the config, since
each schedule entry already carries its signed flow rate and injection temperature.
:func:`assess_seasonal_balance` / :func:`gate_seasonal_balance` are the analytical,
no-solver config-time advisory (mirrors ``mesh.quality.gate_peclet`` /
``model.timestepping.gate_courant``), so an imbalanced doublet is flagged before any
solve, using only the *injected* energy, which is what makes it a config-time check
in the first place: it does not need the extraction phases' simulated temperatures.

:func:`check_energy_closure` (with :func:`stored_energy_change`,
:func:`well_energy_exchange` and :func:`extraction_well_energy`) is the strong
verification tool, and needs the *complete* picture: the modelled change in domain-wide
stored thermal energy must equal the doublets' net energy exchange (injecting *and*
extracting phases; a doublet recovering 100% of what it injects contributes ~zero net
energy over a full cycle, whatever its seasonal ``net_imbalance``) minus what leaves via
the extraction well and the domain boundary.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ATESTool.analysis.integration import integrate_end_step
from ATESTool.config import Config
from ATESTool.model.schedule import SEASONS_PER_YEAR, WellEntry, WellSchedule

#: Below this, a doublet's seasonal cycle is treated as balanced (float-rounding slack).
_BALANCE_TOL_K = 1e-6


def _doublet_names(schedule: WellSchedule) -> list[str]:
    """Every doublet name present in ``schedule`` (a well id with both a ``_warm`` and
    a matching ``_cold`` counterpart), in a stable order."""
    ids = set(schedule.well_locations)
    return sorted(
        well_id[: -len("_warm")]
        for well_id in ids
        if well_id.endswith("_warm") and well_id[: -len("_warm")] + "_cold" in ids
    )


def compute_energy_balance(
    schedule: WellSchedule, rho_c_water: float, season_length_days: float, ambient: float
) -> pd.DataFrame:
    """Per doublet, per simulated year: injected/removed energy and the net imbalance.

    Columns: ``doublet``, ``year``, ``e_warm_injected`` [J] (heat added during the
    summer charge), ``e_cold_injected`` [J] (heat *removed* during the winter charge,
    positive when the cold well runs colder than ``ambient``), ``net_imbalance`` [J]
    (``e_warm_injected - e_cold_injected``; zero iff the doublet is balanced that year),
    and ``cumulative_net_imbalance`` [J] (the running total across years, per doublet).
    Built entirely from ``schedule``; no config field is re-read here.
    """
    length = float(season_length_days)
    rows: list[dict[str, object]] = []
    for name in _doublet_names(schedule):
        warm_id, cold_id = f"{name}_warm", f"{name}_cold"
        warm_by_year = {
            e.stress_period // SEASONS_PER_YEAR: e
            for e in schedule.entries
            if e.well_id == warm_id and e.injection_temperature is not None
        }
        cold_by_year = {
            e.stress_period // SEASONS_PER_YEAR: e
            for e in schedule.entries
            if e.well_id == cold_id and e.injection_temperature is not None
        }
        for year in sorted(set(warm_by_year) | set(cold_by_year)):
            w = warm_by_year.get(year)
            c = cold_by_year.get(year)
            e_warm = 0.0
            if w is not None:
                assert w.injection_temperature is not None  # guaranteed by the filter above
                e_warm = abs(w.flow_rate) * rho_c_water * (w.injection_temperature - ambient) * length
            e_cold = 0.0
            if c is not None:
                assert c.injection_temperature is not None  # guaranteed by the filter above
                e_cold = abs(c.flow_rate) * rho_c_water * (ambient - c.injection_temperature) * length
            rows.append(
                {
                    "doublet": name,
                    "year": year,
                    "e_warm_injected": e_warm,
                    "e_cold_injected": e_cold,
                    "net_imbalance": e_warm - e_cold,
                }
            )
    columns = ["doublet", "year", "e_warm_injected", "e_cold_injected", "net_imbalance"]
    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        table["cumulative_net_imbalance"] = pd.Series(dtype=float)
        return table
    table = table.sort_values(["doublet", "year"]).reset_index(drop=True)
    table["cumulative_net_imbalance"] = table.groupby("doublet")["net_imbalance"].cumsum()
    return table


@dataclass(frozen=True)
class SeasonalBalanceCheck:
    """One doublet's seasonal energy-balance check: closed form, no solver.

    ``temperature_imbalance`` is ``warm_injection_temperature + cold_injection_temperature
    - 2*ambient`` [K]; zero means the doublet's seasonal cycle is exactly balanced.
    ``net_annual_load`` [J] is the same imbalance scaled to an energy quantity
    (``Q . rho_c_water . season_length_days . temperature_imbalance``).
    ``balancing_cold_temperature`` is the ``cold_injection_temperature`` that would zero
    the imbalance, holding ``warm_injection_temperature`` fixed.
    """

    doublet: str
    warm_injection_temperature: float
    cold_injection_temperature: float
    ambient: float
    temperature_imbalance: float
    net_annual_load: float
    balancing_cold_temperature: float


def assess_seasonal_balance(
    config: Config, ambient: float, rho_c_water: float
) -> list[SeasonalBalanceCheck]:
    """The closed-form seasonal balance check for every doublet in ``config``.

    ``ambient``/``rho_c_water`` are passed in (typically the reporting layer's, see
    ``model.builder``'s ``_transmissivity_weights``, mirroring ``analyse``'s own
    convention) rather than re-derived here, so the caller controls which layer the
    check is judged against in a multi-layer aquifer.
    """
    length = float(config.model.simulation.season_length_days)
    checks: list[SeasonalBalanceCheck] = []
    for d in config.model.doublets:
        q = abs(float(d.flow_rate))
        t_warm = float(d.warm_injection_temperature)
        t_cold = float(d.cold_injection_temperature)
        imbalance_k = t_warm + t_cold - 2.0 * ambient
        checks.append(
            SeasonalBalanceCheck(
                doublet=d.name,
                warm_injection_temperature=t_warm,
                cold_injection_temperature=t_cold,
                ambient=ambient,
                temperature_imbalance=imbalance_k,
                net_annual_load=q * rho_c_water * length * imbalance_k,
                balancing_cold_temperature=2.0 * ambient - t_warm,
            )
        )
    return checks


def gate_seasonal_balance(checks: list[SeasonalBalanceCheck], logger: logging.Logger) -> None:
    """Advisory (warn only): flag any doublet whose warm/cold cycle is not balanced.

    A doublet is balanced iff ``warm_injection_temperature + cold_injection_temperature
    == 2*ambient``, the only rule that matters here, regardless of flow rate.
    """
    for c in checks:
        if abs(c.temperature_imbalance) <= _BALANCE_TOL_K:
            continue
        logger.warning(
            "doublet %r has a seasonal energy imbalance: T_warm=%.2f + T_cold=%.2f != "
            "2*T_amb=%.2f (net annual load %+.3g GJ, %s); set cold_injection_temperature "
            "to %.2f to balance it",
            c.doublet,
            c.warm_injection_temperature,
            c.cold_injection_temperature,
            2.0 * c.ambient,
            c.net_annual_load / 1.0e9,
            "net warming" if c.net_annual_load > 0.0 else "net cooling",
            c.balancing_cold_temperature,
        )


def extraction_well_energy(
    times: np.ndarray,
    extraction_temperature: np.ndarray,
    flow_rate: float,
    rho_c_water: float,
    ambient: float,
) -> float:
    """Net thermal energy withdrawn via the extraction well over the run [J].

    Positive when the extracted water runs warmer than ``ambient`` on average (net
    heat removed from the ground). Integrated using the model's end-of-step values,
    mirroring ``analysis.efficiency.compute_efficiency`` and MODFLOW's implicit
    source/sink accounting. This is deliberately *not* a closed form, since the
    extraction well has no fixed injection temperature; its energy content is
    whatever the model simulated.
    """
    times = np.asarray(times, dtype=float)
    temperature = np.asarray(extraction_temperature, dtype=float)
    return float(abs(flow_rate)) * float(rho_c_water) * integrate_end_step(
        times, temperature - ambient
    )


def well_energy_exchange(
    entries: list[WellEntry],
    times: np.ndarray,
    well_temperature: np.ndarray,
    season_length_days: float,
    rho_c_water: float,
    ambient: float,
) -> float:
    """One well's *complete* net energy contribution to the domain [J]: injecting
    and extracting phases both included, over every stress period in ``entries``.

    This is deliberately broader than :func:`compute_energy_balance`'s injection-only
    ``net_imbalance`` (which is a config-time design check, not a physical energy
    tally): a doublet well spends half its periods injecting at a known temperature
    and the other half *extracting* at whatever the model actually simulated at its
    cell, and the true energy contribution needs both; a well that recovers 100% of
    what it injected contributes ~zero net energy over a full cycle, however far its
    ``net_imbalance`` is from zero. Injecting periods use the closed form (like
    :func:`compute_energy_balance`); extracting periods integrate ``well_temperature``
    via the trapezoidal rule (like :func:`extraction_well_energy` /
    ``analysis.efficiency.compute_efficiency``). Summed over every well (doublet and
    standalone), this is the term :func:`check_energy_closure` needs for a *true*
    physical energy balance, as opposed to the seasonal-imbalance design check.
    """
    length = float(season_length_days)
    times = np.asarray(times, dtype=float)
    well_temperature = np.asarray(well_temperature, dtype=float)
    total = 0.0
    for e in entries:
        if e.injection_temperature is not None:
            total += e.flow_rate * rho_c_water * (e.injection_temperature - ambient) * length
            continue
        t0, t1 = e.stress_period * length, (e.stress_period + 1) * length
        total += e.flow_rate * rho_c_water * integrate_end_step(
            times, well_temperature - ambient, start=t0, end=t1
        )
    return total


def stored_energy_change(
    initial_temperature_field: np.ndarray,
    final_temperature_field: np.ndarray,
    cell_areas: np.ndarray,
    thickness: float,
    porosity: float,
    volumetric_heat_water: float,
    volumetric_heat_solid: float,
) -> float:
    """The modelled change in domain-wide stored thermal energy [J].

    ``sum(area_i * thickness * (rho*c)_aquifer * delta_T_i)`` over every cell, where
    ``(rho*c)_aquifer = porosity*volumetric_heat_water + (1-porosity)*volumetric_heat_solid``,
    the same saturated-aquifer volumetric heat capacity ``mesh.thermal.thermal_front_radius``
    uses, so the two stay consistent.
    """
    delta_t = np.asarray(final_temperature_field, dtype=float) - np.asarray(
        initial_temperature_field, dtype=float
    )
    vol_heat_aquifer = porosity * volumetric_heat_water + (1.0 - porosity) * volumetric_heat_solid
    return float(thickness) * vol_heat_aquifer * float(np.sum(np.asarray(cell_areas) * delta_t))


@dataclass(frozen=True)
class EnergyClosure:
    """The energy-balance closure check: stored change vs. net injected minus losses.

    ``stored_energy_change`` [J]: the modelled change in domain-wide stored thermal
    energy (:func:`stored_energy_change`).
    ``net_injected_energy`` [J]: the *complete* net energy the doublet wells added to
    the domain over the whole run, the sum of :func:`well_energy_exchange` over every
    doublet well (injecting *and* extracting phases). This is **not**
    :func:`compute_energy_balance`'s ``net_imbalance`` total: that is an injection-only
    design check (see its own docstring) and on its own overstates the true input by
    whatever the doublet later recovers.
    ``energy_loss`` [J]: energy leaving through terms not captured by the doublet
    input, typically the standalone extraction well's net withdrawal
    (:func:`extraction_well_energy`) plus any boundary (CHD/RCH) exchange.
    ``residual`` [J]: ``stored_energy_change - (net_injected_energy - energy_loss)``,
    zero if the physical energy balance closes exactly.
    ``rel_residual``: ``residual`` normalised by the larger side, for a scale-free
    tolerance check.
    """

    stored_energy_change: float
    net_injected_energy: float
    energy_loss: float
    residual: float
    rel_residual: float

    def closes(self, rel_tol: float = 0.05) -> bool:
        return math.isfinite(self.rel_residual) and abs(self.rel_residual) <= rel_tol


def check_energy_closure(
    stored_energy_change: float, net_injected_energy: float, energy_loss: float = 0.0
) -> EnergyClosure:
    """Compare the modelled stored-energy change against net input minus losses."""
    expected = net_injected_energy - energy_loss
    residual = stored_energy_change - expected
    scale = max(abs(stored_energy_change), abs(expected))
    rel = residual / scale if scale > 0.0 else 0.0
    return EnergyClosure(
        stored_energy_change=stored_energy_change,
        net_injected_energy=net_injected_energy,
        energy_loss=energy_loss,
        residual=residual,
        rel_residual=rel,
    )

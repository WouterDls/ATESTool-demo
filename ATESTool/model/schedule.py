"""``WellSchedule``: the single owner of seasonal ATES well logic.

``build_well_schedule`` turns the doublets + season schedule into a flat table of
``(well_id, stress_period, flow_rate, injection_temperature)`` rows. It owns all
seasonal sign-flips (injection <-> extraction) and the warm/cold phase reversal.
GWF and GWE consume this table and never read doublet or season data directly.

Convention (summer-charge first): in even stress periods the warm well injects at
``warm_injection_temperature`` while the cold well extracts; in odd periods the
phase reverses (cold well injects at ``cold_injection_temperature``).
"""

from __future__ import annotations

from dataclasses import dataclass

from ATESTool.config import Config

#: Seasons per simulated year (summer charge + winter discharge).
SEASONS_PER_YEAR = 2


@dataclass(frozen=True)
class WellEntry:
    """One well's behaviour during one stress period.

    ``flow_rate`` is signed: positive = injection, negative = extraction.
    ``injection_temperature`` is the temperature of the injected water, or ``None``
    when the well is extracting (no temperature is imposed on outflow).
    """

    well_id: str
    stress_period: int
    flow_rate: float
    injection_temperature: float | None


@dataclass
class WellSchedule:
    """A full schedule of doublet well operations across all stress periods."""

    entries: list[WellEntry]
    well_locations: dict[str, tuple[float, float]]
    n_periods: int

    @property
    def well_ids(self) -> list[str]:
        return list(self.well_locations)

    def for_period(self, period: int) -> list[WellEntry]:
        return [e for e in self.entries if e.stress_period == period]


def build_well_schedule(config: Config) -> WellSchedule:
    """Derive the :class:`WellSchedule` from the config's doublets + season schedule."""
    n_years = int(round(float(config.model.simulation.n_years)))
    n_periods = n_years * SEASONS_PER_YEAR

    entries: list[WellEntry] = []
    locations: dict[str, tuple[float, float]] = {}

    for doublet in config.model.doublets:
        warm_id = f"{doublet.name}_warm"
        cold_id = f"{doublet.name}_cold"
        locations[warm_id] = (float(doublet.warm_well[0]), float(doublet.warm_well[1]))
        locations[cold_id] = (float(doublet.cold_well[0]), float(doublet.cold_well[1]))

        rate = float(doublet.flow_rate)
        warm_temp = float(doublet.warm_injection_temperature)
        cold_temp = float(doublet.cold_injection_temperature)

        for period in range(n_periods):
            charging = period % SEASONS_PER_YEAR == 0  # summer-charge first
            if charging:
                entries.append(WellEntry(warm_id, period, +rate, warm_temp))
                entries.append(WellEntry(cold_id, period, -rate, None))
            else:
                entries.append(WellEntry(warm_id, period, -rate, None))
                entries.append(WellEntry(cold_id, period, +rate, cold_temp))

    return WellSchedule(entries=entries, well_locations=locations, n_periods=n_periods)

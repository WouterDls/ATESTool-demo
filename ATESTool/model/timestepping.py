"""Temporal discretisation: choosing the MODFLOW 6 TDIS time stepping.

The temporal twin of the mesh's cell-Péclet gate (``mesh/quality.py``). For
advection-dominated GWE heat transport the Courant number ``Co = v·Δt/Δx`` governs
how much the thermal front smears *in time*; the near-well spatial refinement that
shrinks ``Δx`` correspondingly tightens the step (meshing primer §4, "Courant number:
an accuracy limit, not a stability one").

MODFLOW 6 GWE is implicit (the IMS solver), hence unconditionally *stable*: a large
Courant number does not blow the run up, but it does smear the front. So, like the
Péclet gate, the Courant check is an **accuracy advisory**, not a hard stability
limit.

Three strategies, selected by ``model.simulation.time_stepping.mode``:

* ``fixed``: ``timesteps_per_season`` verbatim (not recommended); the Courant
  gate still runs as an advisory.
* ``courant``: derive the step count from a target Courant number so the near-well
  front is resolved in time; accuracy is then independent of run length.
* ``ats``: MODFLOW 6 Adaptive Time Stepping, with the maximum step capped at the
  Courant step so the two strategies share one accuracy ceiling.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ATESTool.config import Config
from ATESTool.mesh import VertexGrid
from ATESTool.model.schedule import WellSchedule

#: ATS minimum step (days). MF6 recommends a small value so periods end cleanly.
_ATS_DTMIN = 1.0e-5
#: Total-timestep count above which we warn about runtime (advisory only).
_MANY_STEPS = 20_000


@dataclass
class CourantReport:
    """Near-well Courant summary for a mesh + temporal config (mirrors ``PecletReport``).

    target_courant   : the Co target the step is sized against
    near_well_radius  : radius (m) of the assessed near-well region (max refinement ring)
    peak_velocity     : largest near-well radial seepage velocity [m/day]
    binding_cell_size : √area (m) of the cell that sets ``dt_max``
    dt_max            : largest uniform step meeting the target [days] (``inf`` ⇒ not assessable)
    perlen            : season length [days]
    recommended_nstp  : uniform steps/season needed to meet the target
    n_near_well       : number of near-well cells assessed
    """

    target_courant: float
    near_well_radius: float
    peak_velocity: float
    binding_cell_size: float
    dt_max: float
    perlen: float
    recommended_nstp: int
    n_near_well: int


@dataclass
class TimeStepping:
    """Resolved TDIS time stepping: the per-period ``perioddata`` and optional ATS records."""

    #: One ``(perlen, nstp, tsmult)`` per stress period for ``ModflowTdis``.
    perioddata: list[tuple[float, int, float]]
    #: ATS ``(iperats, dt0, dtmin, dtmax, dtadj, dtfailadj)`` records, or ``None`` when not used.
    ats_perioddata: list[tuple[int, float, float, float, float, float]] | None


def _time_stepping(config: Config) -> tuple[str, float, float, float, float]:
    """``(mode, target_courant, tsmult, ats_dtadj, ats_dtfailadj)`` with fixed-mode defaults."""
    ts = config.model.simulation.time_stepping
    if ts is None:
        return "fixed", 1.0, 1.0, 2.0, 5.0
    return (
        ts.mode,
        float(ts.target_courant),
        float(ts.tsmult),
        float(ts.ats_dtadj),
        float(ts.ats_dtfailadj),
    )


def _wells(config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Well ``(x, y)`` coordinates and their ``|flow|`` [m³/day]: doublet warm/cold + extraction."""
    xy: list[tuple[float, float]] = []
    q: list[float] = []
    for d in config.model.doublets:
        rate = abs(float(d.flow_rate))
        xy.append((float(d.warm_well[0]), float(d.warm_well[1])))
        q.append(rate)
        xy.append((float(d.cold_well[0]), float(d.cold_well[1])))
        q.append(rate)
    ext = config.model.extraction_well
    xy.append((float(ext.coordinates[0]), float(ext.coordinates[1])))
    q.append(abs(float(ext.flow_rate)))
    return np.array(xy, dtype=float), np.array(q, dtype=float)


def assess_courant(grid: VertexGrid, config: Config) -> CourantReport:
    """Estimate the near-well Courant step for ``grid`` under ``config``'s flow + transport.

    For each near-well cell (within the largest refinement ring of any well, as in
    :func:`~ATESTool.mesh.quality.assess_peclet`) the radial seepage velocity is
    ``v = Q / (2π·r·b·θ)``: ``Q`` the nearest well's flow, ``r`` its distance (floored at
    half the cell size to avoid the well-cell singularity), ``b`` the aquifer thickness,
    ``θ`` the minimum porosity (conservative). The binding step is ``min(target·Δx/v)``.
    """
    sim = config.model.simulation
    perlen = float(sim.season_length_days)
    fallback_nstp = max(int(round(float(sim.timesteps_per_season))), 1)
    _, target, *_ = _time_stepping(config)

    layers = config.model.aquifer.layers
    thickness = float(layers[0].top) - float(layers[-1].bottom)
    porosity = min(float(layer.porosity) for layer in layers)

    well_xy, well_q = _wells(config)
    near_radius = max((float(r.radius) for r in config.mesh.refinement_rings), default=0.0)

    if near_radius > 0.0 and len(well_xy):
        dist = np.linalg.norm(grid.centroids[:, None, :] - well_xy[None, :, :], axis=2)
        min_dist = dist.min(axis=1)
        nearest = dist.argmin(axis=1)
        near = min_dist <= near_radius
    else:
        near = np.zeros(grid.n_cells, dtype=bool)
        min_dist = np.zeros(grid.n_cells, dtype=float)
        nearest = np.zeros(grid.n_cells, dtype=int)

    if not near.any() or thickness <= 0.0 or porosity <= 0.0 or target <= 0.0:
        return CourantReport(
            target_courant=target,
            near_well_radius=near_radius,
            peak_velocity=0.0,
            binding_cell_size=0.0,
            dt_max=math.inf,
            perlen=perlen,
            recommended_nstp=fallback_nstp,
            n_near_well=int(near.sum()),
        )

    dx = np.sqrt(grid.areas[near])
    r = np.maximum(min_dist[near], dx / 2.0)  # floor avoids the well-cell singularity (r→0)
    q = well_q[nearest[near]]
    v = q / (2.0 * np.pi * r * thickness * porosity)  # radial seepage velocity [m/day]

    # Zero-rate wells (notably paired unpumped references) impose no radial
    # advection ceiling. Avoid dividing by zero just to obtain that infinity.
    dt_cell = np.full_like(v, math.inf)
    np.divide(target * dx, v, out=dt_cell, where=v > 0)
    binding = int(dt_cell.argmin())
    dt_max = float(dt_cell[binding])

    return CourantReport(
        target_courant=target,
        near_well_radius=near_radius,
        peak_velocity=float(v.max()),
        binding_cell_size=float(dx[binding]),
        dt_max=dt_max,
        perlen=perlen,
        recommended_nstp=max(int(math.ceil(perlen / dt_max)), 1),
        n_near_well=int(near.sum()),
    )


def gate_courant(report: CourantReport, configured_nstp: int, logger: logging.Logger) -> None:
    """Advisory (warn only): flag if ``configured_nstp`` under-resolves the front in time."""
    if math.isinf(report.dt_max):
        if report.near_well_radius <= 0.0:
            logger.warning(
                "no refinement rings; cannot assess the Courant number / temporal resolution"
            )
        return
    if configured_nstp >= report.recommended_nstp:
        return
    implied_dt = report.perlen / max(configured_nstp, 1)
    implied_courant = report.target_courant * implied_dt / report.dt_max
    logger.warning(
        "time step under-resolves the thermal front: %d steps/season implies Courant ~%.1f "
        "(target %.1f); recommend >= %d steps/season (dt <= %.2f d). The front may smear in "
        "time - set time_stepping.mode to 'courant' or 'ats', or raise timesteps_per_season",
        configured_nstp,
        implied_courant,
        report.target_courant,
        report.recommended_nstp,
        report.dt_max,
    )


def _courant_nstp(perlen: float, dt_max: float, tsmult: float) -> tuple[int, float]:
    """Smallest ``nstp`` whose largest step ≤ ``dt_max``, plus the effective ``tsmult``.

    With uniform steps (``tsmult == 1``) this is ``ceil(perlen/dt_max)``. With geometric
    growth the *largest* (final) step of a season-spanning series has a floor of
    ``perlen·(m-1)/m`` as ``n→∞`` (~30 days for ``m=1.2``, ``perlen=182.5``); when
    ``dt_max`` is below that floor, grading cannot meet the ceiling, so we drop to uniform
    steps and report ``tsmult = 1.0``. When the Courant-derived step is below that floor,
    the usual ATES case, grading is infeasible; that means the default ``tsmult=1.2``
    under ``courant``/``ats`` rarely actually engages in practice, which is expected
    behaviour, not a bug. A guaranteed small first step (e.g. to resolve a sharp
    injection-temperature step change) would need splitting the season into two stress
    periods, not leaning on ``tsmult``.
    """
    uniform = max(int(math.ceil(perlen / dt_max)), 1)
    if tsmult <= 1.0:
        return uniform, 1.0
    infimum = perlen * (tsmult - 1.0) / tsmult  # last-step floor as n → ∞
    if dt_max <= infimum * 1.05:
        return uniform, 1.0  # grading infeasible at this Courant step
    cap = uniform * 20 + 100
    n = uniform
    while n <= cap:
        dt1 = perlen * (tsmult - 1.0) / (tsmult**n - 1.0)
        last_step = dt1 * tsmult ** (n - 1)
        if last_step <= dt_max:
            return n, tsmult
        n += 1
    return uniform, 1.0  # safety fallback


def _log_total(logger: logging.Logger, mode: str, nstp: int, n_periods: int) -> None:
    total = nstp * n_periods
    big = total > _MANY_STEPS
    (logger.warning if big else logger.info)(
        "time stepping (%s): %d periods x %d steps = %d timesteps%s",
        mode,
        n_periods,
        nstp,
        total,
        " - large; expect a long runtime" if big else "",
    )


def resolve_time_stepping(
    config: Config,
    grid: VertexGrid,
    schedule: WellSchedule,
    logger: logging.Logger,
) -> TimeStepping:
    """Resolve the TDIS time stepping for the configured mode (fixed / courant / ats).

    Always runs :func:`assess_courant` + :func:`gate_courant` so every mode logs the
    temporal-resolution advisory.
    """
    sim = config.model.simulation
    perlen = float(sim.season_length_days)
    n_periods = schedule.n_periods
    configured_nstp = max(int(round(float(sim.timesteps_per_season))), 1)
    mode, target, tsmult, ats_dtadj, ats_dtfailadj = _time_stepping(config)
    report = assess_courant(grid, config)

    if mode == "courant":
        if math.isinf(report.dt_max):
            nstp, eff_tsmult = configured_nstp, tsmult
            logger.warning(
                "courant mode: could not assess the Courant step; "
                "falling back to timesteps_per_season=%d",
                nstp,
            )
        else:
            nstp, eff_tsmult = _courant_nstp(perlen, report.dt_max, tsmult)
            logger.info(
                "courant mode: target Co=%.2f -> dt<=%.2f d, peak v=%.3g m/d "
                "-> %d steps/season (tsmult=%.2f)",
                target,
                report.dt_max,
                report.peak_velocity,
                nstp,
                eff_tsmult,
            )
        gate_courant(report, nstp, logger)
        _log_total(logger, mode, nstp, n_periods)
        return TimeStepping(perioddata=[(perlen, nstp, eff_tsmult)] * n_periods, ats_perioddata=None)

    if mode == "ats":
        if math.isinf(report.dt_max):
            dtmax = perlen / configured_nstp
            logger.warning(
                "ats mode: could not assess the Courant step; capping dtmax at "
                "perlen/timesteps_per_season=%.2f d",
                dtmax,
            )
        else:
            dtmax = report.dt_max
            logger.info(
                "ats mode: dtmax capped at the Courant step %.2f d (target Co=%.2f); "
                "dtadj=%.1f, dtfailadj=%.1f, base=%d steps/season",
                dtmax,
                target,
                ats_dtadj,
                ats_dtfailadj,
                configured_nstp,
            )
        gate_courant(report, configured_nstp, logger)
        dtmin = min(_ATS_DTMIN, dtmax * 0.01)
        # FloPy accepts zero-based stress-period indices and writes the MF6 ATS file's
        # one-based IPERATS values. Passing ``iper + 1`` here shifts adaptation to
        # periods 2..N+1, silently leaving the first season on the coarse TDIS base.
        ats_pd = [
            (iper, dtmax, dtmin, dtmax, ats_dtadj, ats_dtfailadj) for iper in range(n_periods)
        ]
        # TDIS base; ATS overrides the step within each period by solver convergence.
        return TimeStepping(
            perioddata=[(perlen, configured_nstp, tsmult)] * n_periods, ats_perioddata=ats_pd
        )

    # mode == "fixed"
    gate_courant(report, configured_nstp, logger)
    _log_total(logger, mode, configured_nstp, n_periods)
    return TimeStepping(
        perioddata=[(perlen, configured_nstp, tsmult)] * n_periods, ats_perioddata=None
    )

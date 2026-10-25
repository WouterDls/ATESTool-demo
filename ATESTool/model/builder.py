"""``build_model``: assemble the MODFLOW 6 GWF + GWE simulation via FloPy.

Writes all input files to ``workspace`` and returns a private ``_ModelPaths``
(paths only, never a ``SimulationResult``). Writing input files needs only FloPy;
running needs the MODFLOW 6 binary (see ``runner.py``).

GWF: DISV, NPF, IC, STO, CHD (per-side heads), WEL (from the WellSchedule plus the
standalone extraction well), OC. GWE: DISV, EST, CND, IC, ADV, SSM, OC, linked to
GWF by a GWF6-GWE6 exchange. Injection temperature enters via a WEL auxiliary
variable mixed into GWE by SSM.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import flopy
import numpy as np

from ATESTool.analysis.energy import assess_seasonal_balance, gate_seasonal_balance
from ATESTool.config import AquiferLayer, ChdHeads, Config
from ATESTool.mesh import Domain, VertexGrid
from ATESTool.model.schedule import WellSchedule, build_well_schedule
from ATESTool.model.timestepping import resolve_time_stepping

logger = logging.getLogger("ATESTool.model")

SIM_NAME = "atestool"
GWF_NAME = "gwf"
GWE_NAME = "gwe"


@dataclass
class _ModelPaths:
    """The written simulation, its (future) output files, and the schedule it was built from."""

    workspace: Path
    sim_name: str
    gwf_name: str
    gwe_name: str
    head_file: Path
    flow_budget_file: Path
    temperature_file: Path
    energy_budget_file: Path
    extraction_cell: int
    extraction_layer: int
    n_layers: int
    well_cells: dict[str, int]  # doublet well_id -> icpl (for the efficiency metric)
    # The schedule the GWF/GWE packages were built from. Carried (rather than rebuilt by
    # callers) so downstream consumers describe the schedule the model actually solved.
    schedule: WellSchedule


def resolve_mf6_exe() -> str:
    """The MODFLOW 6 executable: ``$MF6_BIN`` if set, else ``mf6`` on PATH."""
    return os.environ.get("MF6_BIN", "mf6")



def _boundary_chd(
    mesh: VertexGrid,
    domain: Domain,
    chd: ChdHeads,
    nlay: int,
    layer_temperatures: list[float],
) -> list[list[object]]:
    """One CHD record per boundary cell per layer: side constants or a regional plane.

    Cell-to-side assignment is :meth:`Domain.classify_boundary_cells` (>= 2 vertices on a
    side's edge; corner cells assigned to the first matching side): the same definition the
    boundary-influence check and the domain-size study use. The boundary head is imposed on
    every layer, since the domain is treated as embedded in a larger regional aquifer rather
    than bounded by a real impermeable edge; a partial-depth boundary would need a reason
    to stop partway down, which nothing in this model provides. Each record
    also carries a TEMPERATURE aux (the layer's ambient/initial temperature) so inflow through
    the boundary enters GWE at the right temperature instead of MF6's SSM default of 0.0
    (mirrors the WEL/RCH pattern).
    """
    side_head = {"west": float(chd.west), "east": float(chd.east),
                 "south": float(chd.south), "north": float(chd.north)}
    if chd.mode not in ("sides", "linear"):
        raise ValueError("CHD mode must be 'sides' or 'linear'")
    if chd.mode == "linear":
        chd.validate_linear()
    records: list[list[object]] = []
    for side, cells in domain.classify_boundary_cells(mesh).items():
        for ic in cells:
            # A regional gradient requires head to vary ALONG the boundary too.
            # Apply at the DISV centre, where MF6's head unknown is located.
            x, y = mesh.centroids[ic]
            head = (chd.regional_head(float(x), float(y), domain.width, domain.height)
                    if chd.mode == "linear" else side_head[side])
            for lay in range(nlay):
                records.append([(lay, ic), head, layer_temperatures[lay]])
    return records


def _transmissivity_weights(layers: list[AquiferLayer]) -> list[float]:
    """Fraction of a fully-penetrating well's flow assigned to each layer.

    Weighted by transmissivity (hydraulic conductivity x layer thickness); falls
    back to an even split if total transmissivity is zero."""
    trans = [abs(float(lay.top) - float(lay.bottom)) * float(lay.hydraulic_conductivity) for lay in layers]
    total = sum(trans)
    if total <= 0.0:
        return [1.0 / len(layers)] * len(layers)
    return [t / total for t in trans]


def _wel_stress_period_data(
    config: Config, mesh: VertexGrid, schedule: WellSchedule
) -> dict[int, list[list[object]]]:
    """Per-period WEL records: doublet wells (from the schedule) + extraction well.

    Wells fully penetrate the aquifer: each well emits one record per layer, with
    its flow split across layers by transmissivity. Each record is
    ``[(layer, cell), q, temperature]``; the temperature is the WEL auxiliary
    consumed by GWE SSM (0.0 for extraction, where it is unused)."""
    weights = _transmissivity_weights(config.model.aquifer.layers)
    extraction = config.model.extraction_well
    ext_cell = mesh.locate_cell(float(extraction.coordinates[0]), float(extraction.coordinates[1]))
    ext_q = -abs(float(extraction.flow_rate))

    spd: dict[int, list[list[object]]] = {}
    for period in range(schedule.n_periods):
        records: list[list[object]] = []
        for entry in schedule.for_period(period):
            x, y = schedule.well_locations[entry.well_id]
            cell = mesh.locate_cell(x, y)
            temp = entry.injection_temperature if entry.injection_temperature is not None else 0.0
            for lay, w in enumerate(weights):
                records.append([(lay, cell), entry.flow_rate * w, temp])
        for lay, w in enumerate(weights):
            records.append([(lay, ext_cell), ext_q * w, 0.0])
        spd[period] = records
    return spd


def build_model(
    config: Config,
    mesh: VertexGrid,
    workspace: Path,
    *,
    budget_save: Literal["ALL", "LAST"] | None = "ALL",
    schedule_override: WellSchedule | None = None,
) -> _ModelPaths:
    """Build and write the GWF + GWE simulation; return the output file paths.

    A few whole-model choices that are otherwise only visible by reading the FloPy calls
    below:

    * **Initial conditions.** GWF's ``strt`` is the plain average of the four CHD boundary
      heads (``north``/``south``/``east``/``west``): a single scalar, uniform over the
      whole grid, since MF6's transient solve only needs a reasonable starting point, not
      an exact one. GWE's ``strt`` is each layer's own ``initial_temperature`` (so a
      multi-layer config with different ambient temperatures per layer starts correctly
      stratified).
    * **The reporting layer** is the layer with the highest transmissivity weight (see
      :func:`_transmissivity_weights`): the same layer a fully-penetrating well sends most
      of its flow through. It is what :func:`~ATESTool.analysis.energy.assess_seasonal_balance`
      treats as "ambient" for the seasonal energy-balance advisory, and what
      ``extraction_layer`` on the returned :class:`_ModelPaths` points ``analyse()`` at for
      the extraction well's 2-D timeseries.
    * **CHD boundary temperature.** Every CHD cell also carries a ``TEMPERATURE`` auxiliary
      (see :func:`_boundary_chd`) equal to that layer's ambient temperature, so inflow across
      the boundary enters GWE at the right temperature instead of MF6's SSM default of 0.0.
    * **Recharge** (``model.recharge``, [m/day]) is off when ``0.0`` (the default). When
      positive, it is applied to the top layer only, as a GWF RCH package plus a matching
      GWE SSM source carrying the top layer's ``initial_temperature`` (mirroring the WEL/CHD
      pattern): recharge water enters at ambient temperature, not 0.0.
    * **Budget retention.** ``budget_save="ALL"`` preserves the normal full-resolution
      cell-budget output. Storage-constrained validation campaigns may request ``"LAST"``
      for one snapshot per stress period or ``None`` for no on-disk cell budgets, while
      retaining every head and temperature step used by the analysis routines.
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    layers = config.model.aquifer.layers
    # Normal application runs always derive their schedule from the public Config.  A
    # verification harness may supply an explicit schedule to isolate one source without
    # pretending that a distant balancing doublet is radial.  The resolved override is
    # archived by that harness; it is intentionally not a user-facing configuration mode.
    schedule = schedule_override if schedule_override is not None else build_well_schedule(config)
    recharge_rate = float(config.model.recharge)

    # The reporting layer (highest transmissivity weight) is also what `analyse()` reads
    # its own ambient temperature from, so the seasonal-balance advisory is judged against
    # the same layer the post-run stats describe (see analysis.energy module docstring).
    reporting_layer = int(np.argmax(_transmissivity_weights(layers)))
    if schedule_override is None:
        balance_checks = assess_seasonal_balance(
            config,
            ambient=float(layers[reporting_layer].initial_temperature),
            rho_c_water=float(layers[reporting_layer].heat_capacity_water)
            * float(layers[reporting_layer].density_water),
        )
        gate_seasonal_balance(balance_checks, logger)

    n_periods = schedule.n_periods
    nlay = len(layers)
    top = float(layers[0].top)
    botm = [float(layer.bottom) for layer in layers]
    disv = mesh.to_disv_args()

    def per_layer(attr: str) -> list[float]:
        """One value per aquifer layer (top-down) for a DISV-grid array."""
        return [float(getattr(layer, attr)) for layer in layers]

    sim = flopy.mf6.MFSimulation(
        sim_name=SIM_NAME, sim_ws=str(workspace), exe_name=resolve_mf6_exe()
    )
    ts = resolve_time_stepping(config, mesh, schedule, logger)
    tdis = flopy.mf6.ModflowTdis(
        sim, time_units="days", nper=n_periods, perioddata=ts.perioddata
    )
    if ts.ats_perioddata is not None:
        # ATS must parent the TDIS package, not the simulation, so FloPy writes the
        # `ATS6 FILEIN` record into TDIS. Parenting it to the simulation instead produces an
        # orphaned .ats file that MF6 silently ignores: the run then executes at the TDIS
        # base step, not the adaptive one, with no error or warning from MF6 itself. This was
        # caught by comparing a 10-year `ats` run's step count against `fixed` (they matched,
        # which they should not have) and is now asserted in tests ("ATS6" must appear in the
        # written .tdis file, and an integration run must exceed the base step count). Do not
        # revert to parenting `ats` on the simulation object.
        flopy.mf6.ModflowUtlats(tdis, maxats=len(ts.ats_perioddata), perioddata=ts.ats_perioddata)

    # --- GWF flow model ---------------------------------------------------- #
    gwf = flopy.mf6.ModflowGwf(sim, modelname=GWF_NAME, save_flows=True)
    # Tolerances tightened below the MODERATE defaults so the flow mass-balance
    # discrepancy stays < 0.1% on unstructured DISV grids with strong well sinks/sources
    # (meshing primer §7, "Mass balance proves conservation, not accuracy"); the default
    # left a ~1.5% discrepancy purely from solver slack.
    flopy.mf6.ModflowIms(
        sim,
        filename=f"{GWF_NAME}.ims",
        complexity="MODERATE",
        outer_dvclose=1e-7,
        inner_dvclose=1e-8,
    )
    sim.register_ims_package(sim.get_package(f"{GWF_NAME}.ims"), [GWF_NAME])

    flopy.mf6.ModflowGwfdisv(
        gwf,
        nlay=nlay,
        top=top,
        botm=botm,
        nvert=disv["nvert"],
        ncpl=disv["ncpl"],
        vertices=disv["vertices"],
        cell2d=disv["cell2d"],
    )
    flopy.mf6.ModflowGwfnpf(
        gwf, save_specific_discharge=True, save_saturation=True, icelltype=0,
        k=per_layer("hydraulic_conductivity"),
    )
    heads = config.model.chd_heads
    strt_head = (float(heads.north) + float(heads.south) + float(heads.east) + float(heads.west)) / 4.0
    flopy.mf6.ModflowGwfic(gwf, strt=strt_head)
    flopy.mf6.ModflowGwfsto(
        gwf, ss=per_layer("specific_storage"), sy=0.0, iconvert=0,
        transient={p: True for p in range(n_periods)},
    )
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD",
        auxiliary=["TEMPERATURE"],
        stress_period_data={
            0: _boundary_chd(
                mesh,
                Domain.from_config(config),
                heads,
                nlay,
                per_layer("initial_temperature"),
            )
        },
    )
    flopy.mf6.ModflowGwfwel(
        gwf,
        auxiliary=["TEMPERATURE"],
        stress_period_data=_wel_stress_period_data(config, mesh, schedule),
        pname="WEL",
    )
    if recharge_rate > 0.0:
        # Areal recharge ("voeding") on the top layer; MF6 treats RECHARGE as a flux
        # [L/T] and scales by cell area, so the same value is correct on the variable-
        # area DISV mesh. It enters at ambient temperature (AUX -> GWE SSM, as for WEL).
        ambient = float(layers[0].initial_temperature)
        rch_spd = [[(0, ic), recharge_rate, ambient] for ic in range(mesh.n_cells)]
        flopy.mf6.ModflowGwfrch(
            gwf, auxiliary=["TEMPERATURE"], stress_period_data={0: rch_spd}, pname="RCH"
        )
    gwf_save_records = [("HEAD", "ALL")]
    if budget_save is not None:
        gwf_save_records.append(("BUDGET", budget_save))
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{GWF_NAME}.hds",
        budget_filerecord=f"{GWF_NAME}.cbc",
        saverecord=gwf_save_records,
    )

    # --- GWE energy-transport model ---------------------------------------- #
    gwe = flopy.mf6.ModflowGwe(sim, modelname=GWE_NAME, save_flows=True)
    flopy.mf6.ModflowIms(sim, filename=f"{GWE_NAME}.ims", complexity="MODERATE")
    sim.register_ims_package(sim.get_package(f"{GWE_NAME}.ims"), [GWE_NAME])

    flopy.mf6.ModflowGwedisv(
        gwe,
        nlay=nlay,
        top=top,
        botm=botm,
        nvert=disv["nvert"],
        ncpl=disv["ncpl"],
        vertices=disv["vertices"],
        cell2d=disv["cell2d"],
    )
    flopy.mf6.ModflowGweest(
        gwe,
        porosity=per_layer("porosity"),
        heat_capacity_water=per_layer("heat_capacity_water"),
        density_water=per_layer("density_water"),
        heat_capacity_solid=per_layer("heat_capacity_solid"),
        density_solid=per_layer("density_solid"),
    )
    flopy.mf6.ModflowGwecnd(
        gwe,
        xt3d_off=True,
        alh=per_layer("longitudinal_dispersivity"),
        ath1=per_layer("transverse_dispersivity"),
        ktw=per_layer("thermal_conductivity_water"),
        kts=per_layer("thermal_conductivity_solid"),
    )
    flopy.mf6.ModflowGweic(gwe, strt=per_layer("initial_temperature"))
    flopy.mf6.ModflowGweadv(gwe, scheme="UPSTREAM")
    ssm_sources = [["WEL", "AUX", "TEMPERATURE"], ["CHD", "AUX", "TEMPERATURE"]]
    if recharge_rate > 0.0:
        ssm_sources.append(["RCH", "AUX", "TEMPERATURE"])
    flopy.mf6.ModflowGwessm(gwe, sources=ssm_sources)
    gwe_save_records = [("TEMPERATURE", "ALL")]
    if budget_save is not None:
        gwe_save_records.append(("BUDGET", budget_save))
    flopy.mf6.ModflowGweoc(
        gwe,
        temperature_filerecord=f"{GWE_NAME}.ucn",
        budget_filerecord=f"{GWE_NAME}.cbc",
        saverecord=gwe_save_records,
    )

    # --- GWF <-> GWE coupling ---------------------------------------------- #
    flopy.mf6.ModflowGwfgwe(sim, exgtype="GWF6-GWE6", exgmnamea=GWF_NAME, exgmnameb=GWE_NAME)

    sim.write_simulation(silent=True)

    extraction = config.model.extraction_well
    extraction_cell = mesh.locate_cell(float(extraction.coordinates[0]), float(extraction.coordinates[1]))
    # The extraction well's timeseries is read from the layer carrying most of its
    # flow: the highest-transmissivity (thus highest-weight) layer; same as reporting_layer.
    extraction_layer = reporting_layer
    well_cells = {wid: mesh.locate_cell(x, y) for wid, (x, y) in schedule.well_locations.items()}

    return _ModelPaths(
        workspace=workspace,
        sim_name=SIM_NAME,
        gwf_name=GWF_NAME,
        gwe_name=GWE_NAME,
        head_file=workspace / f"{GWF_NAME}.hds",
        flow_budget_file=workspace / f"{GWF_NAME}.cbc",
        temperature_file=workspace / f"{GWE_NAME}.ucn",
        energy_budget_file=workspace / f"{GWE_NAME}.cbc",
        extraction_cell=extraction_cell,
        extraction_layer=extraction_layer,
        n_layers=nlay,
        well_cells=well_cells,
        schedule=schedule,
    )

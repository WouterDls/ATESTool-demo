"""Boundary-influence check: is the domain big enough?

Two scalars over the *boundary cells* (the constant-head edge of the domain, identified by
:meth:`ATESTool.mesh.Domain.boundary_cell_indices`):

* ``max_temperature_change``: ``max |T_final - ambient|`` over the boundary cells [degC].
  This is the transient signal that actually diagnoses an undersized domain: once the thermal
  plume reaches the constant-head edge the model is contaminated by it, so a rising value means
  the boundary is too close. It matches the ``boundary_thermal_influence`` the domain-size
  convergence study already reports.
* ``max_head_change``: ``max |h_final - h_initial|`` over the boundary cells [m], referenced
  to the uniform GWF initial condition. Under the current model the boundary head is *pinned*
  by CHD, so this reduces to the imposed boundary-head offset from the domain-mean initial head,
  a static measure of how hard the boundary is being driven, not a transient influence. It
  becomes a genuine transient signal only once boundaries can vary in time (Fase 2). The
  *thermal* metric is therefore the primary domain-adequacy indicator; the head metric is kept
  for symmetry and because it is a cheap, always-available deterministic health scalar
  report.

Both are folded into :attr:`ATESTool.analysis.SimulationResult.stats` (computed once in
``analyse``), so every run reports them without extra plumbing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ATESTool.mesh import Domain, VertexGrid

if TYPE_CHECKING:  # avoid an import cycle: results.py imports this module
    from ATESTool.analysis.results import SimulationResult


@dataclass(frozen=True)
class BoundaryInfluence:
    """How disturbed the constant-head boundary is at the end of a run.

    ``n_boundary_cells`` is the count of cells the metrics were taken over; the two changes
    are NaN when there are no boundary cells (a degenerate mesh).
    """

    max_temperature_change: float
    max_head_change: float
    n_boundary_cells: int


def boundary_influence(sim: SimulationResult, mesh: VertexGrid, domain: Domain) -> BoundaryInfluence:
    """Max ``|ΔT|`` and max ``|Δh|`` over the domain's boundary cells at the final reported time.

    ``ΔT`` is relative to the ambient (initial) temperature; ``Δh`` is relative to the uniform
    GWF initial-condition head (``sim.initial_head`` when known, else the first reported head).
    See the module docstring for what each scalar does and does not diagnose.
    """
    boundary = domain.boundary_cell_indices(mesh)
    if boundary.size == 0:
        return BoundaryInfluence(float("nan"), float("nan"), 0)

    t_end = float(sim.times[-1])
    final_temp = np.asarray(sim.snapshot(t_end, "temperature"), dtype=float)
    final_head = np.asarray(sim.snapshot(t_end, "head"), dtype=float)

    max_dt = float(np.max(np.abs(final_temp[boundary] - float(sim.initial_temperature))))
    if math.isnan(sim.initial_head):
        # No IC recorded (e.g. a synthetic result): reference each boundary cell to its own
        # first reported head, so the metric measures the change *during* the run.
        reference = np.asarray(sim.head[0], dtype=float)[boundary]
    else:
        reference = float(sim.initial_head)
    max_dh = float(np.max(np.abs(final_head[boundary] - reference)))

    return BoundaryInfluence(
        max_temperature_change=max_dt,
        max_head_change=max_dh,
        n_boundary_cells=int(boundary.size),
    )

"""Mesh quality metrics and the acceptance gate (meshing primer §2 "The quality gate",
§3 "Cell Péclet").

A :class:`VertexGrid` that passes :func:`gate_quality` is adequate for a MODFLOW 6 DISV
run: cells are well-shaped and the control-volume finite-difference (CVFD) orthogonality
and bisection conditions hold to a usable tolerance. :func:`assess_peclet` adds the
transport-side check: whether the near-well cells are fine enough to resolve the thermal
front without excessive numerical dispersion. The metrics are cheap to compute and catches
when the mesh generator breaks/quality drops.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ATESTool.config import Config

from ATESTool.mesh.thermal import THERMAL_FRONT_DIVISIONS, THERMAL_FRONT_RADIUS_FACTOR, thermal_front_radius
from ATESTool.mesh.vertex_grid import VertexGrid

# Warning thresholds (meshing primer §2, "The quality gate"): below/above these the mesh
# is suspect.
MIN_ANGLE_WARN = 20.0  # degrees
ORTHOGONALITY_WARN = 30.0  # degrees
ASPECT_WARN = 10.0
BISECTION_WARN = 0.30  # fraction of face length

# Hard limits. For a Voronoi mesh the CVFD-correctness metrics are orthogonality (faces
# are perpendicular bisectors, so ~0 deg) and the minimum angle (degenerate cells break
# the conductance geometry). Aspect ratio and bisection error are quality *advisories*:
# a graded mesh legitimately has a few high-aspect cells and tiny faces at fine/coarse
# transitions and the domain boundary, which carry negligible flow and do not threaten
# correctness. So only orthogonality and min-angle gate hard; the rest warn.
ORTHOGONALITY_RAISE = 45.0  # degrees
MIN_ANGLE_RAISE = 5.0  # degrees


class MeshQualityError(ValueError):
    """Raised when a mesh violates a hard CVFD/shape quality limit."""


@dataclass
class MeshQuality:
    """Aggregate mesh-quality metrics over a :class:`VertexGrid`.

    Attributes
    ----------
    min_angle_deg : smallest interior cell-corner angle (want > 20 deg)
    max_orthogonality_deg : largest deviation of a cell-centre connector from the
        shared-face normal (want < 30 deg; 0 = perfectly orthogonal)
    max_aspect_ratio : largest per-cell longest-edge / shortest-edge ratio (want < 10)
    max_bisection_error : largest offset between where the cell-centre connector crosses
        a shared face and that face's midpoint, as a fraction of face length (want ~ 0)
    n_cells : number of cells assessed
    """

    min_angle_deg: float
    max_orthogonality_deg: float
    max_aspect_ratio: float
    max_bisection_error: float
    n_cells: int


def assess_quality(grid: VertexGrid) -> MeshQuality:
    """Compute :class:`MeshQuality` for ``grid``."""
    verts = grid.vertices
    min_angle = 180.0
    max_aspect = 1.0

    for cell in grid.cell_vertices:
        poly = verts[cell]
        n = len(poly)
        if n < 3:
            continue
        edge_len = np.linalg.norm(poly - np.roll(poly, -1, axis=0), axis=1)
        nonzero = edge_len[edge_len > 0.0]
        if nonzero.size:
            max_aspect = max(max_aspect, float(nonzero.max() / nonzero.min()))
        for i in range(n):
            a = poly[(i - 1) % n]
            b = poly[i]
            c = poly[(i + 1) % n]
            ang = _corner_angle(a, b, c)
            if ang is not None:
                min_angle = min(min_angle, ang)

    max_ortho, max_bisect = _face_metrics(grid)

    return MeshQuality(
        min_angle_deg=float(min_angle),
        max_orthogonality_deg=float(max_ortho),
        max_aspect_ratio=float(max_aspect),
        max_bisection_error=float(max_bisect),
        n_cells=grid.n_cells,
    )


def _corner_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Interior angle at ``b`` formed by edges b->a and b->c, in degrees."""
    u = a - b
    v = c - b
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return None
    cosang = float(np.dot(u, v) / (nu * nv))
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def _face_metrics(grid: VertexGrid) -> tuple[float, float]:
    """Max orthogonality angle (deg) and max bisection error over shared faces."""
    verts = grid.vertices
    centers = grid.centroids
    cell_sets = [set(c) for c in grid.cell_vertices]
    max_ortho = 0.0
    max_bisect = 0.0
    seen: set[tuple[int, int]] = set()

    for i, neighbours in grid.connectivity.items():
        for j in neighbours:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            shared = list(cell_sets[i] & cell_sets[j])
            if len(shared) != 2:
                continue
            a = verts[shared[0]]
            b = verts[shared[1]]
            face = b - a
            face_len = float(np.linalg.norm(face))
            if face_len == 0.0:
                continue
            connector = centers[j] - centers[i]
            conn_len = float(np.linalg.norm(connector))
            if conn_len == 0.0:
                continue
            # Orthogonality: angle between the connector and the face normal (0 = ideal).
            normal = np.array([-face[1], face[0]]) / face_len
            cosang = abs(float(np.dot(connector / conn_len, normal)))
            cosang = max(-1.0, min(1.0, cosang))
            max_ortho = max(max_ortho, math.degrees(math.acos(cosang)))
            # Bisection: distance from face midpoint to where the connector line crosses
            # the face line, as a fraction of face length.
            crossing = _line_intersection(centers[i], centers[j], a, b)
            if crossing is not None:
                midpoint = 0.5 * (a + b)
                max_bisect = max(max_bisect, float(np.linalg.norm(crossing - midpoint)) / face_len)

    return max_ortho, max_bisect


def _line_intersection(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> np.ndarray | None:
    """Intersection of line (p1,p2) with line (p3,p4); None if parallel."""
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def gate_quality(quality: MeshQuality, logger: logging.Logger, *, strict: bool = True) -> None:
    """Warn on soft-threshold breaches; raise (when ``strict``) on hard-limit breaches."""
    if quality.min_angle_deg < MIN_ANGLE_WARN:
        logger.warning(
            "mesh min interior angle %.1f deg < %.1f deg (near-degenerate cells)",
            quality.min_angle_deg,
            MIN_ANGLE_WARN,
        )
    if quality.max_orthogonality_deg > ORTHOGONALITY_WARN:
        logger.warning(
            "mesh max orthogonality angle %.1f deg > %.1f deg (CVFD error)",
            quality.max_orthogonality_deg,
            ORTHOGONALITY_WARN,
        )
    if quality.max_aspect_ratio > ASPECT_WARN:
        logger.warning(
            "mesh max aspect ratio %.1f > %.1f", quality.max_aspect_ratio, ASPECT_WARN
        )
    if quality.max_bisection_error > BISECTION_WARN:
        logger.warning(
            "mesh max bisection error %.2f > %.2f (face midpoint offset)",
            quality.max_bisection_error,
            BISECTION_WARN,
        )

    if not strict:
        return
    if quality.max_orthogonality_deg > ORTHOGONALITY_RAISE:
        raise MeshQualityError(
            f"mesh orthogonality angle {quality.max_orthogonality_deg:.1f} deg exceeds the "
            f"hard limit {ORTHOGONALITY_RAISE} deg; refusing to build an unreliable DISV grid"
        )
    if quality.min_angle_deg < MIN_ANGLE_RAISE:
        raise MeshQualityError(
            f"mesh min interior angle {quality.min_angle_deg:.1f} deg is below the hard limit "
            f"{MIN_ANGLE_RAISE} deg (near-degenerate cells); refusing to build an unreliable DISV grid"
        )


# --- Transport cell-Péclet check (meshing primer §3, "Cell Péclet") --------------- #
# Grid Péclet Pe = Δx / α_L (advection-dominated limit; ignoring thermal diffusion makes
# this the conservative bound). Pe ≲ 2 avoids numerical dispersion smearing the front;
# Pe > 4 is poor. Only the near-well region (inside the refinement rings, where the
# thermal plume lives) is gated; the coarse far field legitimately exceeds it.
PECLET_WARN = 2.0
PECLET_STRONG = 4.0


@dataclass
class PecletReport:
    """Grid cell-Péclet summary for a mesh, given the transport config.

    alpha_l : longitudinal dispersivity used (the minimum across layers, most restrictive)
    near_well_radius : radius (m) of the near-well region assessed (max refinement ring)
    max_peclet : largest grid Péclet over all cells (informational; far field is coarse)
    max_peclet_near_well : largest grid Péclet among near-well cells (the gated number)
    n_near_well : number of near-well cells assessed
    n_near_well_over_warn / n_near_well_over_strong : near-well cells exceeding the thresholds
    """

    alpha_l: float
    near_well_radius: float
    max_peclet: float
    max_peclet_near_well: float
    n_near_well: int
    n_near_well_over_warn: int
    n_near_well_over_strong: int


def assess_peclet(grid: VertexGrid, config: Config) -> PecletReport:
    """Compute the grid cell-Péclet report for ``grid`` under ``config``'s transport props."""
    alphas = [float(layer.longitudinal_dispersivity) for layer in config.model.aquifer.layers]
    alpha_l = min(alphas) if alphas else 0.0

    char_len = np.sqrt(grid.areas)
    if alpha_l > 0.0:
        peclet = char_len / alpha_l
    else:  # zero dispersivity -> pure advection, grid Péclet is unbounded
        peclet = np.full(grid.n_cells, np.inf)

    wells = [d.warm_well for d in config.model.doublets]
    wells += [d.cold_well for d in config.model.doublets]
    wells.append(config.model.extraction_well.coordinates)
    well_xy = np.array([[float(x), float(y)] for x, y in wells], dtype=float)
    near_radius = max((float(r.radius) for r in config.mesh.refinement_rings), default=0.0)

    if near_radius > 0.0 and len(well_xy):
        dist = np.min(
            np.linalg.norm(grid.centroids[:, None, :] - well_xy[None, :, :], axis=2), axis=1
        )
        near = dist <= near_radius
    else:
        near = np.zeros(grid.n_cells, dtype=bool)
    near_peclet = peclet[near]

    return PecletReport(
        alpha_l=alpha_l,
        near_well_radius=near_radius,
        max_peclet=float(peclet.max()) if grid.n_cells else 0.0,
        max_peclet_near_well=float(near_peclet.max()) if near_peclet.size else 0.0,
        n_near_well=int(near.sum()),
        n_near_well_over_warn=int((near_peclet > PECLET_WARN).sum()),
        n_near_well_over_strong=int((near_peclet > PECLET_STRONG).sum()),
    )


def gate_peclet(report: PecletReport, logger: logging.Logger) -> None:
    """Advisory (warn only): flag if near-well cells under-resolve the transport front."""
    if report.alpha_l <= 0.0:
        logger.warning(
            "longitudinal dispersivity is 0; grid Péclet is unbounded (pure advection) - "
            "the thermal front may be dominated by numerical dispersion"
        )
        return
    if report.max_peclet_near_well > PECLET_WARN:
        logger.warning(
            "near-well grid Péclet %.1f > %.1f (%d/%d cells within %.0f m of a well); "
            "cells are coarse relative to the %.1f m dispersivity, so numerical dispersion "
            "may smear the thermal front - refine near the wells if breakthrough accuracy matters",
            report.max_peclet_near_well,
            PECLET_WARN,
            report.n_near_well_over_warn,
            report.n_near_well,
            report.near_well_radius,
            report.alpha_l,
        )


# --- Thermal-front resolution check (ATES meshing guideline) --------------------- #
# Independent of ``mesh.thermal_refinement`` mode: runs whether the rings were auto-derived
# or hand-picked, so it catches exactly the gap that motivated this feature: a manual ring
# that happens to under-resolve a doublet's thermal radius R_th, silently, with no warning,
# the moment flow_rate/porosity/season_length_days/aquifer properties change. Mirrors
# ``gate_courant`` (``model/timestepping.py``) on the temporal axis: advisory only, since an
# under-resolved front smears rather than destabilises the solve.


@dataclass
class ThermalResolutionCheck:
    """One ATES doublet well's thermal-front resolution check.

    Attributes
    ----------
    well_id            : ``"{doublet}_warm"`` / ``"{doublet}_cold"``
    r_th               : thermal radius for this well's season (m)
    target_cell_size   : R_th / THERMAL_FRONT_DIVISIONS, the "10-20 cells across R_th" target
    achieved_cell_size : smallest cell (sqrt(area)) found in the front band
        ``[0.8, 1.2] * R_th`` around this well; ``NaN`` if the mesh has no cells there
        (e.g. the band falls outside the domain).
    """

    well_id: str
    r_th: float
    target_cell_size: float
    achieved_cell_size: float


def assess_thermal_resolution(
    grid: VertexGrid, config: Config
) -> list[ThermalResolutionCheck]:
    """Compute :class:`ThermalResolutionCheck` for every ATES doublet well in ``config``.

    Uses the first aquifer layer's properties (matches ``mesh.generator``'s auto-ring
    sizing) and each doublet's own ``flow_rate``; different doublets can have different
    R_th. The extraction well has no thermal radius (no injection), so it is not checked.
    """
    layer = config.model.aquifer.layers[0]
    season = float(config.model.simulation.season_length_days)
    thickness = abs(float(layer.top) - float(layer.bottom))
    porosity = float(layer.porosity)
    vol_heat_water = float(layer.heat_capacity_water) * float(layer.density_water)
    vol_heat_solid = float(layer.heat_capacity_solid) * float(layer.density_solid)

    checks: list[ThermalResolutionCheck] = []
    for doublet in config.model.doublets:
        r_th = thermal_front_radius(
            q=abs(float(doublet.flow_rate)),
            t=season,
            thickness=thickness,
            porosity=porosity,
            volumetric_heat_water=vol_heat_water,
            volumetric_heat_solid=vol_heat_solid,
        )
        target = r_th / THERMAL_FRONT_DIVISIONS
        band_min, band_max = r_th / THERMAL_FRONT_RADIUS_FACTOR, r_th * THERMAL_FRONT_RADIUS_FACTOR
        for well_id, (wx, wy) in (
            (f"{doublet.name}_warm", doublet.warm_well),
            (f"{doublet.name}_cold", doublet.cold_well),
        ):
            dist = np.hypot(grid.centroids[:, 0] - float(wx), grid.centroids[:, 1] - float(wy))
            band = (dist >= band_min) & (dist <= band_max)
            achieved = float(np.sqrt(grid.areas[band]).min()) if band.any() else float("nan")
            checks.append(
                ThermalResolutionCheck(
                    well_id=well_id, r_th=r_th, target_cell_size=target, achieved_cell_size=achieved
                )
            )
    return checks


def gate_thermal_resolution(checks: list[ThermalResolutionCheck], logger: logging.Logger) -> None:
    """Advisory (warn only): flag any doublet well whose mesh under-resolves its R_th."""
    for c in checks:
        if math.isnan(c.achieved_cell_size):
            continue  # front band has no cells to assess (e.g. off the edge of the domain)
        if c.achieved_cell_size > c.target_cell_size:
            logger.warning(
                "thermal front under-resolved near %r: R_th=%.1f m, cells in the %.1f-%.1f m "
                "band are %.2f m (target <= %.2f m, i.e. %.0f cells across R_th) - the front "
                "may smear from numerical dispersion; set mesh.thermal_refinement='auto' or "
                "tighten refinement_rings near this well",
                c.well_id,
                c.r_th,
                c.r_th / THERMAL_FRONT_RADIUS_FACTOR,
                c.r_th * THERMAL_FRONT_RADIUS_FACTOR,
                c.achieved_cell_size,
                c.target_cell_size,
                THERMAL_FRONT_DIVISIONS,
            )

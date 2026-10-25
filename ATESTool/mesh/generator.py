"""``generate_mesh(config) -> VertexGrid``

Primary path: a **Voronoi-Delaunay** mesh. A constrained-Delaunay triangulation of the
domain is built with the ``triangle`` package (wells fed in as nodes, plus
graded refinement seed points near each well); its **Voronoi dual** is then taken and
clipped to the domain. The Voronoi generators become the DISV cell centres, which
satisfies MODFLOW 6's CVFD orthogonality requirement.
- Wells are generators, so each well sits exactly at a cell centre.

If ``triangle`` is unavailable, falls back to ``scipy.spatial.Delaunay`` over a uniform
background grid and logs a warning; **refinement and the Voronoi dual are not applied
in that fallback** (the cells are raw triangles).

The domain is the axis-aligned rectangle centred on the origin,
``[-width/2, +width/2] x [-height/2, +height/2]``, see :class:`~ATESTool.mesh.domain.Domain`,
which owns that convention.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from ATESTool.config import Config
from ATESTool.mesh.domain import Domain
from ATESTool.mesh.thermal import (
    THERMAL_FRONT_DIVISIONS,
    THERMAL_FRONT_RADIUS_FACTOR,
    thermal_front_radius,
)
from ATESTool.mesh.vertex_grid import VertexGrid

logger = logging.getLogger(__name__)

# Default background resolution: when the config leaves mesh.background_max_cell_size unset,
# the far-field cell size is derived from the domain as min(width, height) / _BACKGROUND_DIVISIONS.
_BACKGROUND_DIVISIONS = 20
_QUALITY_ANGLE = 30
# Geometric expansion ratio for graded near-well sizing. Meshing primer §1, "Graded
# near-well sizing" wants 1.3-1.5; ratios > 2 introduce large conductance contrasts that
# hurt solver conditioning.
_EXPANSION_RATIO = 1.4
# Absolute sliver-cleanup tolerances (metres): merge Voronoi vertices this close, and
# snap vertices this close to a domain edge onto it. Sized to absorb degenerate-triangle
# circumcenter pairs and clip artefacts without disturbing real (sub-)metre faces.
_SNAP_TOL = 0.05
_BOUNDARY_SNAP_TOL = 0.1

Point = tuple[float, float]


def generate_mesh(config: Config, refinement_factor: float = 1.0) -> VertexGrid:
    """Build the constrained Voronoi mesh for a single (deterministic) configuration.

    ``refinement_factor`` uniformly scales every target cell size (the refinement-ring
    sizes and the derived background size); ``1.0`` reproduces the config's resolution,
    ``0.5`` halves all cell sizes (≈ 4x the cells). It is only used for grid-convergence
    studies (``validation.run_convergence``); the baseline resolution always comes from
    the config.
    """
    domain = Domain.from_config(config)
    well_rings = _well_refinement(config, refinement_factor)
    background_cell = _background_cell(config, refinement_factor)

    try:
        import triangle as triangle_mod
    except ImportError:
        logger.warning( #TODO: Make louder and more pressing. Let the end-user install traingle to have better mesh quality.
            "the 'triangle' package is not installed; falling back to "
            "scipy.spatial.Delaunay - per-well refinement and the Voronoi dual are NOT "
            "applied (raw triangle cells degrade MODFLOW 6 CVFD accuracy)"
        )
        wells = [well for well, _ in well_rings]
        grid = _generate_scipy(domain, wells, background_cell)
    else:
        grid = _generate_voronoi(triangle_mod, domain, well_rings, background_cell)

    _check_quality(grid, config)
    return grid


def _background_cell(config: Config, refinement_factor: float) -> float:
    """The far-field target cell size, scaled by ``refinement_factor``.

    Uses ``mesh.background_max_cell_size`` when set, else derives it from the domain
    (``min(width, height) / _BACKGROUND_DIVISIONS``). The derived branch is the historical
    behaviour, reproduced exactly when the field is absent.
    """
    configured = config.mesh.background_max_cell_size
    if configured is not None:
        base = float(configured)
    else:
        base = min(float(config.mesh.width), float(config.mesh.height)) / _BACKGROUND_DIVISIONS
    return base * refinement_factor


def _check_quality(grid: VertexGrid, config: Config) -> None:
    """Assess mesh quality and warn/raise per the meshing primer §2 ("The quality gate")
    thresholds.

    The geometric hard-limit raise applies only to the Voronoi path; the scipy fallback
    is an already-degraded raw-triangle mesh (logged loudly at generation), so it warns
    only. The transport cell-Péclet check (primer §3) is always advisory (warn only).
    """
    from ATESTool.mesh.quality import (
        assess_peclet,
        assess_quality,
        assess_thermal_resolution,
        gate_peclet,
        gate_quality,
        gate_thermal_resolution,
    )

    gate_quality(assess_quality(grid), logger, strict=grid.backend == "voronoi")
    gate_peclet(assess_peclet(grid, config), logger)
    gate_thermal_resolution(assess_thermal_resolution(grid, config), logger)


def _thermal_ring(config: Config, doublet_flow_rate: float) -> tuple[float, float]:
    """The auto front-band ring ``(radius, max_cell_size)`` for one doublet's thermal plume.

    Sized off the thermal radius R_th (see ``mesh.thermal.thermal_front_radius``) using the
    first aquifer layer's properties and one season's injection volume: a season is the
    plume's growth window before the doublet's flow reverses (charge -> discharge).
    """
    layer = config.model.aquifer.layers[0]
    r_th = thermal_front_radius(
        q=abs(doublet_flow_rate),
        t=float(config.model.simulation.season_length_days),
        thickness=abs(float(layer.top) - float(layer.bottom)),
        porosity=float(layer.porosity),
        volumetric_heat_water=float(layer.heat_capacity_water) * float(layer.density_water),
        volumetric_heat_solid=float(layer.heat_capacity_solid) * float(layer.density_solid),
    )
    return THERMAL_FRONT_RADIUS_FACTOR * r_th, r_th / THERMAL_FRONT_DIVISIONS


def _well_refinement(
    config: Config, refinement_factor: float
) -> list[tuple[Point, list[tuple[float, float]]]]:
    """Every well the mesh should resolve, paired with the refinement rings that apply to it.

    Every well gets ``config.mesh.refinement_rings`` (scaled by ``refinement_factor``, as
    before). When ``mesh.thermal_refinement == "auto"``, each ATES doublet well *also* gets a
    ring sized to its own thermal radius (:func:`_thermal_ring`) for front-band resolution.
    The extraction well never gets this: it has no injection temperature, so a thermal radius
    has no meaning for it, and refining near it anyway is what made an earlier attempt at this
    feature Courant-cost prohibitive: that first version added the thermal ring to *every*
    well, including the (much higher-flow) extraction well, which shrank its near-well cells
    far more than its own hydraulics called for and, since MODFLOW 6 uses one global time
    step, pushed the whole simulation's Courant-limited step count up by more than 15x.
    """
    base_rings = [
        (float(r.radius), float(r.max_cell_size) * refinement_factor)
        for r in config.mesh.refinement_rings
    ]
    auto = config.mesh.thermal_refinement == "auto"

    out: list[tuple[Point, list[tuple[float, float]]]] = []
    for doublet in config.model.doublets:
        rings = base_rings
        if auto:
            radius, size = _thermal_ring(config, float(doublet.flow_rate))
            rings = base_rings + [(radius, size * refinement_factor)]
        out.append(((float(doublet.warm_well[0]), float(doublet.warm_well[1])), rings))
        out.append(((float(doublet.cold_well[0]), float(doublet.cold_well[1])), rings))

    well = config.model.extraction_well
    out.append(((float(well.coordinates[0]), float(well.coordinates[1])), base_rings))
    return out


def _unique_points(arrays: list[np.ndarray], decimals: int = 6) -> np.ndarray:
    """Stack point arrays, dropping duplicates while preserving first-seen order.

    Order matters: the domain corners are passed first so they keep indices 0..3,
    which the boundary segments reference.
    """
    seen: set[tuple[float, float]] = set()
    out: list[Point] = []
    for arr in arrays:
        for px, py in arr:
            key = (round(float(px), decimals), round(float(py), decimals))
            if key not in seen:
                seen.add(key)
                out.append((float(px), float(py)))
    return np.array(out, dtype=float)


def _graded_seed_points(
    center: Point,
    rings: list[tuple[float, float]],
    background_cell: float,
    domain: Domain,
) -> np.ndarray:
    """Concentric seed points around a well, graded geometrically outward.

    The innermost target cell size is the smallest ring's ``max_cell_size``. Moving
    outward the size grows by :data:`_EXPANSION_RATIO` per ring, capped by any ring whose
    radius still encloses the current point and by ``background_cell``. Both the radial
    step and the circumferential spacing equal the local target size, so the resulting
    triangles (and thus Voronoi cells) grade smoothly to the far-field resolution.
    """
    if not rings:
        return np.empty((0, 2), dtype=float)
    cx, cy = center
    h_min = min(mc for _, mc in rings)
    if h_min <= 0.0:
        return np.empty((0, 2), dtype=float)

    def cap(r: float) -> float:
        enclosing = [mc for radius, mc in rings if radius >= r]
        return min(enclosing) if enclosing else background_cell

    coords: list[Point] = []
    r = h_min
    h = h_min
    # Stop once we are past the outermost ring AND have grown to the background size.
    r_outer = max(radius for radius, _ in rings)
    for _ in range(1000):  # generous iteration cap; loop normally breaks well before
        h = min(h, cap(r), background_cell)
        n_circ = max(6, int(math.ceil(2.0 * math.pi * r / h)))
        for k in range(n_circ):
            theta = 2.0 * math.pi * k / n_circ
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            if domain.contains(x, y):
                coords.append((x, y))
        if r >= r_outer and h >= background_cell:
            break
        r += h
        h = min(h * _EXPANSION_RATIO, background_cell)
    return np.array(coords, dtype=float) if coords else np.empty((0, 2), dtype=float)


def _generate_voronoi(
    triangle_mod: Any,
    domain: Domain,
    well_rings: list[tuple[Point, list[tuple[float, float]]]],
    background_cell: float,
) -> VertexGrid:
    corners = domain.corners
    point_arrays = [corners]
    wells = [well for well, _ in well_rings]
    if wells:
        point_arrays.append(np.array(wells, dtype=float))
    for well, rings in well_rings:
        seeds = _graded_seed_points(well, rings, background_cell, domain)
        if len(seeds):
            point_arrays.append(seeds)
    points = _unique_points(point_arrays)

    # Boundary segments reference the first four points (the corners).
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    background_area = 0.5 * background_cell**2
    pslg = {"vertices": points, "segments": segments}
    opts = f"pq{_QUALITY_ANGLE}a{background_area:.6f}"
    out = triangle_mod.triangulate(pslg, opts)
    nodes = np.asarray(out["vertices"], dtype=float)

    # Clean-up tolerances. Both target the small slivers the Voronoi/clip step produces
    # (degenerate-triangle circumcenter pairs and points landing just off the boundary);
    # they are deliberately *absolute*, since slivers arise at a float/geometry scale
    # rather than at the cell-size scale. Merging keeps neighbouring cells referencing
    # *identical* vertices, which DISV needs to derive shared-edge connections.
    snap_tol = _SNAP_TOL
    boundary_tol = _BOUNDARY_SNAP_TOL

    verts, cell_vertices, centers = _voronoi_cells(
        nodes, domain, snap_tol, boundary_tol
    )
    return VertexGrid.from_polygons(verts, cell_vertices, centers, backend="voronoi")


def _voronoi_cells(
    nodes: np.ndarray, domain: Domain, snap_tol: float, boundary_tol: float
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Voronoi dual of ``nodes``, clipped to ``domain``.

    Returns the snapped/deduplicated vertex array, the per-cell vertex-index lists, and
    the generator point of each cell (the cell centre). One cell per node that yields a
    valid (>= 3 vertex) clipped polygon. Vertices within ``boundary_tol`` of a domain
    edge are snapped onto it (this removes boundary slivers without touching interior
    faces), then vertices within ``snap_tol`` of each other are merged.
    """
    from scipy.spatial import Voronoi

    vor = Voronoi(nodes)
    regions, region_verts = _finite_regions(vor, radius=10.0 * max(domain.width, domain.height))

    raw_polys: list[tuple[int, list[Point]]] = []
    flat: list[Point] = []
    for p1, region in regions:
        poly = _clip_to_rect([tuple(region_verts[v]) for v in region], domain)
        if len(poly) < 3:
            continue
        poly = [_snap_to_boundary(pt, domain, boundary_tol) for pt in poly]
        raw_polys.append((p1, poly))
        flat.extend(poly)

    coords, labels = _snap_vertices(np.array(flat, dtype=float), snap_tol)

    cell_vertices: list[list[int]] = []
    centers: list[Point] = []
    cursor = 0
    for p1, poly in raw_polys:
        ids = [int(labels[cursor + k]) for k in range(len(poly))]
        cursor += len(poly)
        deduped: list[int] = []
        for vid in ids:  # drop consecutive duplicates (collapsed edges)
            if not deduped or deduped[-1] != vid:
                deduped.append(vid)
        if len(deduped) >= 2 and deduped[0] == deduped[-1]:
            deduped.pop()
        if len(deduped) < 3:
            continue
        cell_vertices.append(deduped)
        centers.append((float(vor.points[p1, 0]), float(vor.points[p1, 1])))

    return coords, cell_vertices, np.array(centers, dtype=float)


def _snap_to_boundary(pt: Point, domain: Domain, tol: float) -> Point:
    """Snap a point onto a domain edge if it lies within ``tol`` of it."""
    x, y = pt
    if abs(x - domain.x_min) < tol:
        x = domain.x_min
    elif abs(x - domain.x_max) < tol:
        x = domain.x_max
    if abs(y - domain.y_min) < tol:
        y = domain.y_min
    elif abs(y - domain.y_max) < tol:
        y = domain.y_max
    return (x, y)


def _snap_vertices(points: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge points within ``tol`` of each other (union-find) into canonical vertices.

    Returns ``(unique_coords, labels)`` where ``labels[i]`` indexes ``unique_coords``.
    """
    from scipy.spatial import cKDTree

    n = len(points)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if tol > 0.0 and n > 1:
        tree = cKDTree(points)
        for a, b in tree.query_pairs(tol):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    roots: dict[int, int] = {}
    labels = np.empty(n, dtype=int)
    coords: list[Point] = []
    for i in range(n):
        r = find(i)
        canon = roots.get(r)
        if canon is None:
            canon = len(coords)
            roots[r] = canon
            coords.append((float(points[r, 0]), float(points[r, 1])))
        labels[i] = canon
    return np.array(coords, dtype=float), labels


def _finite_regions(
    vor: Any, radius: float
) -> tuple[list[tuple[int, list[int]]], np.ndarray]:
    """Reconstruct finite Voronoi regions, extending unbounded ridges to ``radius``.

    Standard scipy recipe, returning ``(point_index, ordered vertex indices)`` per
    region and the (possibly extended) Voronoi vertex array.
    """
    new_vertices: list[list[float]] = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges.setdefault(int(p1), []).append((int(p2), int(v1), int(v2)))
        ridges.setdefault(int(p2), []).append((int(p1), int(v1), int(v2)))

    regions: list[tuple[int, list[int]]] = []
    for p1 in range(len(vor.points)):
        region_idx = vor.point_region[p1]
        verts = vor.regions[region_idx]
        if verts and all(v >= 0 for v in verts):
            regions.append((p1, [int(v) for v in verts]))
            continue
        finite = [int(v) for v in verts if v >= 0]
        for p2, v1, v2 in ridges.get(p1, []):
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue  # a fully finite ridge
            t = vor.points[p2] - vor.points[p1]
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            finite.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        if len(finite) < 3:
            continue
        pts = np.array([new_vertices[v] for v in finite])
        c = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        ordered = [finite[i] for i in np.argsort(angles)]
        regions.append((p1, ordered))
    return regions, np.array(new_vertices, dtype=float)


def _clip_to_rect(poly: list[tuple[float, float]], domain: Domain) -> list[Point]:
    """Sutherland-Hodgman clip of a polygon against ``domain``.

    Each edge is a half-plane ``(axis, bound, sign)``: a point is inside when
    ``(p[axis] - bound) * sign >= 0``.
    """
    edges: tuple[tuple[int, float, float], ...] = (
        (0, domain.x_min, 1.0),
        (0, domain.x_max, -1.0),
        (1, domain.y_min, 1.0),
        (1, domain.y_max, -1.0),
    )
    pts: list[Point] = [(float(x), float(y)) for x, y in poly]
    for axis, bound, sign in edges:
        if not pts:
            break
        out: list[Point] = []
        n = len(pts)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            ina = (a[axis] - bound) * sign >= 0.0
            inb = (b[axis] - bound) * sign >= 0.0
            if ina:
                out.append(a)
                if not inb:
                    out.append(_crossing(a, b, axis, bound))
            elif inb:
                out.append(_crossing(a, b, axis, bound))
        pts = out
    return pts


def _crossing(a: Point, b: Point, axis: int, bound: float) -> Point:
    """Point on segment a->b where coordinate ``axis`` equals ``bound``."""
    other = 1 - axis
    t = 0.0 if b[axis] == a[axis] else (bound - a[axis]) / (b[axis] - a[axis])
    o = a[other] + (b[other] - a[other]) * t
    return (bound, o) if axis == 0 else (o, bound)


def _generate_scipy(domain: Domain, wells: list[Point], background_cell: float) -> VertexGrid:
    from scipy.spatial import Delaunay

    corners = domain.corners
    xs = np.arange(domain.x_min + background_cell, domain.x_max, background_cell)
    ys = np.arange(domain.y_min + background_cell, domain.y_max, background_cell)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    well_pts = np.array(wells, dtype=float) if wells else np.empty((0, 2), dtype=float)
    points = _unique_points([corners, grid, well_pts])

    delaunay = Delaunay(points)
    return VertexGrid.from_triangulation(
        points, delaunay.simplices, delaunay.neighbors, backend="scipy"
    )

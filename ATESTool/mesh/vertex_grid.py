"""``VertexGrid``

No module outside ``mesh.generator`` ever sees raw Triangle/scipy/Voronoi output.

Cells are general convex polygons (Voronoi cells in the primary path; triangles in the
scipy fallback). Each cell carries an explicit cell **centre** (``centroids``) which is the value MODFLOW 6's
control-volume finite-difference scheme uses for inter-cell distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _signed_area(poly_pts: np.ndarray) -> float:
    """Signed area of a polygon given its (n, 2) vertices in order (CCW > 0)."""
    x = poly_pts[:, 0]
    y = poly_pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _connectivity_from_neighbors(neighbors: np.ndarray, n_cells: int) -> dict[int, list[int]]:
    nb = np.asarray(neighbors, dtype=int)
    return {ic: [int(j) for j in nb[ic] if j >= 0] for ic in range(n_cells)}


def _connectivity_from_edges(cell_vertices: list[list[int]]) -> dict[int, list[int]]:
    """Cell-to-cell neighbours via shared edges (works for any polygon)."""
    edge_to_cells: dict[tuple[int, int], list[int]] = {}
    for ic, verts in enumerate(cell_vertices):
        n = len(verts)
        for k in range(n):
            a, b = verts[k], verts[(k + 1) % n]
            edge = (int(min(a, b)), int(max(a, b)))
            edge_to_cells.setdefault(edge, []).append(ic)
    conn: dict[int, list[int]] = {ic: [] for ic in range(len(cell_vertices))}
    for cells in edge_to_cells.values():
        if len(cells) == 2:
            conn[cells[0]].append(cells[1])
            conn[cells[1]].append(cells[0])
    return conn


@dataclass
class VertexGrid:
    """An unstructured mesh of convex polygonal cells.

    Attributes
    ----------
    vertices : (n_vertices, 2) float array
    cell_vertices : list of per-cell vertex-index lists (variable length; 3 for the
        triangle fallback, typically 4-7 for Voronoi cells)
    centroids : (n_cells, 2) float array. The DISV cell centre per cell: the Voronoi
        generator on the primary path, the triangle centroid on the scipy fallback.
    areas : (n_cells,) float array
    connectivity : {cell_index: [neighbour cell indices]}
    backend : "voronoi", "triangle", or "scipy": which generator produced this grid
    """

    vertices: np.ndarray
    cell_vertices: list[list[int]]
    centroids: np.ndarray
    areas: np.ndarray
    connectivity: dict[int, list[int]]
    backend: str = "voronoi"

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_cells(self) -> int:
        return len(self.cell_vertices)

    def locate_cell(self, x: float, y: float) -> int:
        """Index of the cell whose centre is nearest to ``(x, y)``.

        Used to map a well coordinate onto its DISV cell (e.g. for placing wells
        or reading a per-well timeseries).
        """
        d = (self.centroids[:, 0] - x) ** 2 + (self.centroids[:, 1] - y) ** 2
        return int(np.argmin(d))

    @property
    def triangles(self) -> np.ndarray:
        """The (n_cells, 3) vertex-index array; only valid for an all-triangle mesh.

        Raises if any cell is not a triangle (e.g. a Voronoi grid). Kept for the
        scipy-fallback path and back-compat; downstream code should prefer
        :attr:`cell_vertices`.
        """
        if any(len(v) != 3 for v in self.cell_vertices):
            raise ValueError("triangles is only defined for an all-triangle mesh")
        return np.array(self.cell_vertices, dtype=int)

    @classmethod
    def from_triangulation(
        cls,
        vertices: np.ndarray,
        triangles: np.ndarray,
        neighbors: np.ndarray | None = None,
        backend: str = "scipy",
    ) -> VertexGrid:
        vertices = np.asarray(vertices, dtype=float)
        triangles = np.asarray(triangles, dtype=int)
        tri_pts = vertices[triangles]  # (n_cells, 3, 2)
        centroids = tri_pts.mean(axis=1)
        x = tri_pts[:, :, 0]
        y = tri_pts[:, :, 1]
        areas = 0.5 * np.abs(
            x[:, 0] * (y[:, 1] - y[:, 2])
            + x[:, 1] * (y[:, 2] - y[:, 0])
            + x[:, 2] * (y[:, 0] - y[:, 1])
        )
        cell_vertices = [[int(i) for i in tri] for tri in triangles]
        if neighbors is not None:
            connectivity = _connectivity_from_neighbors(neighbors, len(triangles))
        else:
            connectivity = _connectivity_from_edges(cell_vertices)
        return cls(vertices, cell_vertices, centroids, areas, connectivity, backend)

    @classmethod
    def from_polygons(
        cls,
        vertices: np.ndarray,
        cell_vertices: list[list[int]],
        centers: np.ndarray,
        backend: str = "voronoi",
    ) -> VertexGrid:
        """Build from explicit polygon cells and their generator/centre points."""
        vertices = np.asarray(vertices, dtype=float)
        centers = np.asarray(centers, dtype=float)
        areas = np.array(
            [abs(_signed_area(vertices[v])) for v in cell_vertices], dtype=float
        )
        connectivity = _connectivity_from_edges(cell_vertices)
        return cls(vertices, cell_vertices, centers, areas, connectivity, backend)

    def to_disv_args(self) -> dict[str, Any]:
        """Planar FloPy DISV keyword arguments for this mesh.

        Returns horizontal properties only (``nvert``, ``vertices`` and ``ncpl``, ``cell2d``).
        The vertical discretisation (``nlay``, ``top``, ``botm``) is *not* a mesh property. It
        comes from the aquifer config and is added by the model builder.
        Cell vertices are emitted in clockwise order for MODFLOW 6 DISV, and each
        cell's centre is its stored ``centroid`` (the Voronoi generator).
        """
        vertices = [[i, float(vx), float(vy)] for i, (vx, vy) in enumerate(self.vertices)]
        cell2d: list[list[float]] = []
        for ic in range(self.n_cells):
            verts = list(self.cell_vertices[ic])
            if _signed_area(self.vertices[verts]) > 0.0:  # CCW -> reverse to CW
                verts = verts[::-1]
            xc, yc = self.centroids[ic]
            row: list[float] = [ic, float(xc), float(yc), len(verts)]
            row.extend(int(iv) for iv in verts)
            cell2d.append(row)
        return {
            "nvert": self.n_vertices,
            "vertices": vertices,
            "ncpl": self.n_cells,
            "cell2d": cell2d,
        }

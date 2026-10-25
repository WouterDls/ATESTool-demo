"""``Domain``: the axis-aligned model rectangle, centred on the origin.

The domain is ``[-width/2, +width/2] x [-height/2, +height/2]``. The extraction well sits
at the origin by config convention, and every well coordinate is expressed relative to it.
This ``±W/2`` interpretation of ``mesh.width``/``mesh.height`` lives **here and only here**:
mesh generation, the CHD boundary, and the convergence studies all take a :class:`Domain`
rather than re-deriving the bounds, so the convention has a single home (and moving it is a
one-line change, not a hunt for literals). Exported from :mod:`ATESTool.mesh`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ATESTool.config import Config

if TYPE_CHECKING:  # only for typing; Domain must stay a light dependency of mesh/model/analysis
    from ATESTool.mesh.vertex_grid import VertexGrid

#: A vertex is "on" a domain edge when its coordinate is within this many metres of it.
#: Shared so the CHD builder, the boundary-influence check, and the domain-size study all
#: agree on which cells sit on the boundary.
BOUNDARY_TOL = 1e-6

#: The four domain sides, in the fixed priority order used to break the corner-cell tie
#: (a cell touching two edges is assigned to the first side listed here).
SIDES = ("west", "east", "south", "north")


@dataclass(frozen=True)
class Domain:
    """The axis-aligned rectangle the mesh fills, given by its four bounds."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @classmethod
    def from_config(cls, config: Config) -> Domain:
        """The domain implied by ``mesh.width``/``mesh.height``, centred on the origin."""
        w = float(config.mesh.width)
        h = float(config.mesh.height)
        return cls(-w / 2.0, w / 2.0, -h / 2.0, h / 2.0)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    @property
    def corners(self) -> np.ndarray:
        """The four corners as a ``(4, 2)`` array, CCW (counter-clockwise) from the lower-left."""
        return np.array(
            [
                [self.x_min, self.y_min],
                [self.x_max, self.y_min],
                [self.x_max, self.y_max],
                [self.x_min, self.y_max],
            ],
            dtype=float,
        )

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """True when ``(x, y)`` lies strictly inside the domain, optionally inset by ``margin``.

        Strict on the open interior so a point exactly on an edge (a corner or boundary
        segment, handled separately by the mesher) is *not* reported as interior.
        """
        return (
            self.x_min + margin < x < self.x_max - margin
            and self.y_min + margin < y < self.y_max - margin
        )

    def _side_vertex_masks(
        self, vertices: np.ndarray, tol: float = BOUNDARY_TOL
    ) -> dict[str, np.ndarray]:
        """Per-side boolean mask over ``vertices``; True where a vertex lies on that edge."""
        v = np.asarray(vertices, dtype=float)
        return {
            "west": np.isclose(v[:, 0], self.x_min, atol=tol),
            "east": np.isclose(v[:, 0], self.x_max, atol=tol),
            "south": np.isclose(v[:, 1], self.y_min, atol=tol),
            "north": np.isclose(v[:, 1], self.y_max, atol=tol),
        }

    def classify_boundary_cells(
        self, mesh: VertexGrid, tol: float = BOUNDARY_TOL
    ) -> dict[str, list[int]]:
        """Assign each boundary cell to exactly one domain side.

        A cell belongs to a side if at least two of its vertices lie on that side's edge.
        A corner cell touching two edges is assigned to the first matching side in
        :data:`SIDES` order (west, east, south, north), so it is never listed under two
        sides. This is the single definition of "which cell sits on which boundary",
        shared by the CHD builder (which needs the per-side split to impose each side's
        head) and by the boundary-influence / domain-size checks (which take the union).
        """
        masks = self._side_vertex_masks(mesh.vertices, tol)
        used: set[int] = set()
        result: dict[str, list[int]] = {side: [] for side in SIDES}
        for side in SIDES:
            mask = masks[side]
            for ic in range(mesh.n_cells):
                if ic in used:
                    continue
                if int(mask[mesh.cell_vertices[ic]].sum()) >= 2:
                    result[side].append(ic)
                    used.add(ic)
        return result

    def boundary_cell_indices(self, mesh: VertexGrid, tol: float = BOUNDARY_TOL) -> np.ndarray:
        """Indices of every cell with >= 2 vertices on *any* domain edge, ascending.

        The union of :meth:`classify_boundary_cells` (each boundary cell belongs to exactly
        one side), computed directly from the combined edge mask so callers that do not care
        which side a cell is on need not build the per-side split.
        """
        masks = self._side_vertex_masks(mesh.vertices, tol)
        on_edge = masks["west"] | masks["east"] | masks["south"] | masks["north"]
        return np.array(
            [ic for ic in range(mesh.n_cells) if int(on_edge[mesh.cell_vertices[ic]].sum()) >= 2],
            dtype=int,
        )

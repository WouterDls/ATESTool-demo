"""Regenerate the static figures referenced by meshing_primer.md.

Run from the repo root: ``python docs/generate_figures.py``. Overwrites
docs/_static/voronoi_vs_delaunay.png and docs/_static/peclet_resolution.png.
matplotlib is already an ATESTool runtime dependency, so no extra install is needed
in a normal ATESTool environment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent / "_static"


def make_voronoi_vs_delaunay() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    # --- Left panel: raw Delaunay triangle, centroid vs bisector mismatch ---
    ax = axes[0]
    tri = np.array([[0.0, 0.0], [4.0, 0.3], [1.2, 3.4]])
    tri_closed = np.vstack([tri, tri[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], "k-", lw=1.5)
    ax.fill(tri[:, 0], tri[:, 1], color="#cfe3ff", alpha=0.5)

    centroid = tri.mean(axis=0)
    # a neighbouring triangle sharing the edge tri[0]-tri[1]
    neighbour_apex = np.array([2.2, -2.0])
    tri2 = np.array([tri[0], tri[1], neighbour_apex])
    tri2_closed = np.vstack([tri2, tri2[0]])
    ax.plot(tri2_closed[:, 0], tri2_closed[:, 1], "k-", lw=1.5)
    ax.fill(tri2[:, 0], tri2[:, 1], color="#ffe1cf", alpha=0.5)
    centroid2 = tri2.mean(axis=0)

    edge_mid = (tri[0] + tri[1]) / 2
    ax.plot(*edge_mid, "ks", ms=4)
    ax.plot(*centroid, "o", color="#1f5fbf", ms=7, label="triangle centroid")
    ax.plot(*centroid2, "o", color="#bf4a1f", ms=7)
    ax.plot(
        [centroid[0], centroid2[0]], [centroid[1], centroid2[1]], "--", color="#555555", lw=1.2
    )
    ax.annotate(
        "centroid-centroid line\nmisses the edge midpoint\nat a non-right angle",
        xy=((centroid[0] + centroid2[0]) / 2, (centroid[1] + centroid2[1]) / 2),
        xytext=(2.6, 1.6),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"),
    )
    ax.set_title("Raw Delaunay triangles", fontsize=10.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Right panel: Voronoi cells, generator-to-generator perpendicular bisector ---
    ax = axes[1]
    gen_a = np.array([1.0, 1.0])
    gen_b = np.array([3.4, 1.6])
    mid = (gen_a + gen_b) / 2
    direction = gen_b - gen_a
    perp = np.array([-direction[1], direction[0]])
    perp = perp / np.linalg.norm(perp)
    face = np.array([mid - 1.6 * perp, mid + 1.6 * perp])

    # rough Voronoi-like cell outlines around each generator (illustrative, not exact)
    cell_a = np.array(
        [
            [-0.6, -0.6],
            [2.05, -0.9],
            [face[0][0], face[0][1]],
            [face[1][0], face[1][1]],
            [1.9, 3.1],
            [-0.9, 2.4],
        ]
    )
    cell_b = np.array(
        [
            [face[0][0], face[0][1]],
            [4.9, -0.3],
            [5.4, 3.2],
            [2.9, 3.4],
            [face[1][0], face[1][1]],
        ]
    )
    ax.fill(cell_a[:, 0], cell_a[:, 1], color="#cfe3ff", alpha=0.5)
    ax.fill(cell_b[:, 0], cell_b[:, 1], color="#ffe1cf", alpha=0.5)
    ax.plot(
        np.append(cell_a[:, 0], cell_a[0, 0]), np.append(cell_a[:, 1], cell_a[0, 1]), "k-", lw=1.2
    )
    ax.plot(
        np.append(cell_b[:, 0], cell_b[0, 0]), np.append(cell_b[:, 1], cell_b[0, 1]), "k-", lw=1.2
    )

    ax.plot(*gen_a, "o", color="#1f5fbf", ms=7, label="Voronoi generator")
    ax.plot(*gen_b, "o", color="#bf4a1f", ms=7)
    ax.plot([gen_a[0], gen_b[0]], [gen_a[1], gen_b[1]], "--", color="#555555", lw=1.2)
    ax.plot(*mid, "ks", ms=4)
    ax.annotate(
        "shared face is the exact\nperpendicular bisector\nof the generator line",
        xy=mid,
        xytext=(2.6, -0.9),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"),
    )
    ax.set_title("Voronoi cells (DISV path)", fontsize=10.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle(
        "Cell-centre placement: Delaunay centroid vs Voronoi generator",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "voronoi_vs_delaunay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def make_peclet_resolution() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)

    x = np.linspace(0, 10, 400)
    front_x = 5.0

    # Sharp/true front (thin sigmoid), shown on both panels for reference
    true_front = 1.0 / (1.0 + np.exp((x - front_x) / 0.15))

    # Low-Pe (well-resolved): numerical solution tracks the true front closely
    low_pe = 1.0 / (1.0 + np.exp((x - front_x) / 0.25))
    # High-Pe (poorly resolved): numerical dispersion smears the front over many cells
    high_pe = 1.0 / (1.0 + np.exp((x - front_x) / 1.6))

    for ax, smeared, pe_label, cell_size in (
        (axes[0], low_pe, "Pe ≲ 2 (well resolved)", 0.4),
        (axes[1], high_pe, "Pe > 4 (poorly resolved)", 2.5),
    ):
        ax.plot(x, true_front, color="#888888", lw=1.2, ls=":", label="sharp thermal front")
        ax.plot(x, smeared, color="#bf4a1f", lw=2.2, label="modelled temperature")
        # draw a few cell boundaries near the front to illustrate cell size
        n_cells = int(6 / cell_size)
        starts = front_x - 3 + cell_size * np.arange(n_cells + 1)
        for cx in starts:
            ax.axvline(cx, color="#cccccc", lw=0.6, zorder=0)
        ax.set_title(pe_label, fontsize=10.5)
        ax.set_xlabel("distance from well")
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([])
        ax.legend(fontsize=8, loc="upper right", frameon=False)

    axes[0].set_ylabel("normalised temperature")
    fig.suptitle("Grid Péclet number and near-well thermal-front resolution", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT_DIR / "peclet_resolution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_voronoi_vs_delaunay()
    make_peclet_resolution()

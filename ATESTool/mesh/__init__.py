"""Triangle-based Voronoi mesh generation.

``generate_mesh(config)`` is the sole producer of a :class:`VertexGrid`, which is
the only mesh representation the rest of the pipeline consumes. :func:`assess_quality`
reports the CVFD/shape metrics used to gate (i.e. check the quality of) the generated mesh.
"""

from ATESTool.mesh.domain import Domain
from ATESTool.mesh.generator import generate_mesh
from ATESTool.mesh.quality import (
    MeshQuality,
    MeshQualityError,
    PecletReport,
    ThermalResolutionCheck,
    assess_peclet,
    assess_quality,
    assess_thermal_resolution,
)
from ATESTool.mesh.thermal import thermal_front_radius
from ATESTool.mesh.vertex_grid import VertexGrid

__all__ = [
    "generate_mesh",
    "Domain",
    "VertexGrid",
    "assess_quality",
    "MeshQuality",
    "MeshQualityError",
    "assess_peclet",
    "PecletReport",
    "assess_thermal_resolution",
    "ThermalResolutionCheck",
    "thermal_front_radius",
]

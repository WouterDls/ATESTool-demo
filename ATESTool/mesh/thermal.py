"""The ATES thermal-radius formula, shared by the analytical benchmark and mesh sizing.

Kept dependency-free (floats + numpy only) so both ``validation.analytical`` (which composes
it into the closed-form benchmark) and ``mesh.generator`` (which sizes refinement rings
against it, see ``thermal_refinement="auto"``) import from here rather than from each other.
``mesh`` must not depend on ``validation``; the reverse dependency already exists
(``validation.analytical`` imports ``VertexGrid`` from ``mesh``).
"""

from __future__ import annotations

import numpy as np

#: Front-band outer radius, as a multiple of R_th (meshing primer §5, "Thermal radius as a
#: meshing target": resolve 0.8-1.2*R_th sharply). Shared by ``mesh.generator``'s
#: ``thermal_refinement="auto"`` ring sizing and ``mesh.quality``'s advisory gate, so the
#: two can never silently disagree on what "resolved" means.
THERMAL_FRONT_RADIUS_FACTOR = 1.2
#: Target cells spanning R_th ("10-20 cells across R_th" guideline; midpoint-ish default).
THERMAL_FRONT_DIVISIONS = 18.0


def thermal_front_radius(
    q: float,
    t: float,
    thickness: float,
    porosity: float,
    volumetric_heat_water: float,
    volumetric_heat_solid: float,
) -> float:
    """Radius of the thermal front for radial injection, from an energy balance.

    Equating the heat carried by the injected water ``ρ_w c_w · Q·t·ΔT`` to the heat stored
    in the swept cylinder ``π R² b · (ρc)_aq · ΔT`` gives

        ``R_th = sqrt( Q·t·(ρc)_w / (π·b·(ρc)_aq) )``,

    where ``(ρc)_aq = n·(ρc)_w + (1-n)·(ρc)_s`` is the saturated-aquifer volumetric heat
    capacity. This is the thermally-retarded front: ``R_th = R_hydraulic / sqrt(R_retard)``
    with ``R_retard = (ρc)_aq / (n·(ρc)_w)``. ``q`` is the injection volumetric rate (> 0);
    ``volumetric_heat_*`` are ``ρ·c`` for water and solid. Units of ``q·t`` are volume.
    """
    vol_heat_aquifer = porosity * volumetric_heat_water + (1.0 - porosity) * volumetric_heat_solid
    return float(np.sqrt(q * t * volumetric_heat_water / (np.pi * thickness * vol_heat_aquifer)))

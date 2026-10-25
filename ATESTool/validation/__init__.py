"""Deterministic convergence studies, health checks, and analytical benchmarks.

The three convergence studies each
refine one axis and report whether key outputs have converged: ``run_convergence`` refines
the *mesh*, ``run_temporal_convergence`` the *time step* (mesh fixed), and
``run_domain_convergence`` the *domain size* (distance to the boundary).
``compare_thermal_front`` benchmarks the modelled thermal front against the closed-form
energy-balance radius, bounding *absolute* error where convergence bounds only *relative*.

:mod:`ATESTool.diagnostic` runs all of these in one pass; the functions here are the
individually-callable pieces.
"""

from ATESTool.validation.analytical import (
    AnalyticalResult,
    FrontSample,
    compare_thermal_front,
    radial_profile,
    theis_drawdown,
    thermal_front_radius,
    thermal_front_trajectory,
    thiem_drawdown,
)
from ATESTool.validation.convergence import (
    ConvergenceLevel,
    ConvergenceResult,
    run_convergence,
)
from ATESTool.validation.domain_convergence import (
    DomainConvergenceResult,
    DomainLevel,
    run_domain_convergence,
)
from ATESTool.validation.health import RunHealth, read_run_health
from ATESTool.validation.temporal_convergence import (
    TemporalConvergenceResult,
    TemporalLevel,
    run_temporal_convergence,
)

__all__ = [
    "run_convergence",
    "ConvergenceResult",
    "ConvergenceLevel",
    "run_domain_convergence",
    "DomainConvergenceResult",
    "DomainLevel",
    "run_temporal_convergence",
    "TemporalConvergenceResult",
    "TemporalLevel",
    "thiem_drawdown",
    "theis_drawdown",
    "thermal_front_radius",
    "thermal_front_trajectory",
    "radial_profile",
    "compare_thermal_front",
    "AnalyticalResult",
    "FrontSample",
    "RunHealth",
    "read_run_health",
]

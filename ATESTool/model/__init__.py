"""FloPy GWF + GWE model setup.

``build_well_schedule`` -> ``WellSchedule`` owns all seasonal logic; ``build_model``
assembles the simulation; ``run_model`` executes it.
"""

from ATESTool.model.builder import build_model
from ATESTool.model.runner import run_model
from ATESTool.model.schedule import WellSchedule, build_well_schedule
from ATESTool.model.timestepping import (
    CourantReport,
    TimeStepping,
    assess_courant,
    gate_courant,
    resolve_time_stepping,
)

__all__ = [
    "build_model",
    "run_model",
    "WellSchedule",
    "build_well_schedule",
    "assess_courant",
    "gate_courant",
    "resolve_time_stepping",
    "CourantReport",
    "TimeStepping",
]

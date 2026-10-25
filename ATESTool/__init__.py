"""ATESTool: config-driven ATES groundwater-interaction modelling.

Builds and runs MODFLOW 6 (GWF + GWE) models from a YAML config via FloPy.
See README.md for an overview.
"""

from ATESTool.config import load_config
from ATESTool.run import run_from_config

__version__ = "0.1.0"

__all__ = ["run_from_config", "load_config", "__version__"]

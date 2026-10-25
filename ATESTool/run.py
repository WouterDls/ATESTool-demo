"""Top-level pipeline + CLI (Command-Line Interface).

``run_from_config`` wires the whole pipeline together; ``cli`` is a thin argparse
wrapper used by the ``ATESTool`` console entry point. Each step logs to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ATESTool.analysis import analyse
from ATESTool.config import load_config
from ATESTool.mesh import generate_mesh
from ATESTool.model import build_model, run_model

logger = logging.getLogger("ATESTool")

_DEFAULT_WORKSPACE = "atestool_output"


def run_from_config(
    config_path: str | Path,
    workspace: str | Path | None = None,
    diagnostic: bool = False,
) -> None:
    """Run the deterministic pipeline from a YAML config.

    The pipeline loads the config, generates the mesh, builds and runs the model, analyses
    the results, and writes ``timeseries.csv`` and ``stats.json``.

    ``diagnostic=True`` additionally writes every plot and runs the
    spatial, temporal and analytical verification studies into ``<workspace>/diagnostic/``.
    That costs several extra model runs, see :mod:`ATESTool.diagnostic`. The normal
    single-run artifacts are written first and are unaffected; a diagnostic stage that
    fails is only reported, never raised.

    """
    work = Path(workspace) if workspace is not None else Path.cwd() / _DEFAULT_WORKSPACE
    work.mkdir(parents=True, exist_ok=True)

    logger.info("loading config: %s", config_path)
    config = load_config(config_path)

    logger.info("generating mesh")
    mesh = generate_mesh(config)
    logger.info("building model")
    model_paths = build_model(config, mesh, work)
    logger.info("running model")
    raw = run_model(model_paths)
    logger.info("analysing results")
    result = analyse(model_paths, raw, config, mesh)
    result.timeseries.to_csv(work / "timeseries.csv")
    (work / "stats.json").write_text(json.dumps(result.stats, indent=2))
    if config.analysis.observation_points:
        result.observation_timeseries(config.analysis.observation_points, mesh).to_csv(
            work / "observations.csv"
        )
    logger.info("done: outputs in %s", work)

    if diagnostic:
        # Imported here so the default path never pays for matplotlib.
        from ATESTool.diagnostic import run_diagnostic

        run_diagnostic(config, mesh, result, model_paths.schedule, work)


def cli(argv: list[str] | None = None) -> None:
    """Console entry point for the ``ATESTool`` command."""
    parser = argparse.ArgumentParser(
        prog="ATESTool", description="Config-driven ATES groundwater modelling on MODFLOW 6."
    )
    parser.add_argument("config", help="path to the YAML config file")
    parser.add_argument("--workspace", default=None, help="output directory")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "also write every plot and run the spatial/temporal/analytical verification "
            "studies (costs several extra model runs)"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    run_from_config(
        args.config,
        workspace=args.workspace,
        diagnostic=args.diagnostic,
    )


if __name__ == "__main__":
    cli()

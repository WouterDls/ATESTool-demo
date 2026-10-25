"""Small shared runner for the three public examples."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from ATESTool.analysis import analyse
from ATESTool.config import Config, Doublet, load_config
from ATESTool.mesh import generate_mesh
from ATESTool.model import build_model, run_model

ROOT = Path(__file__).resolve().parent
AMBIENT_TEMPERATURE_C = 12.0


def doublet(
    base: Config,
    name: str,
    warm_well: tuple[float, float],
    cold_well: tuple[float, float],
    flow_rate: float = 200.0,
) -> Doublet:
    """Return one doublet using the shared example's thermal properties."""
    item = copy.deepcopy(base.model.doublets[0])
    item.name = name
    item.warm_well = warm_well
    item.cold_well = cold_well
    item.flow_rate = flow_rate
    return item


def run_comparison(
    cases: Mapping[str, list[Doublet]], output_dir: Path
) -> pd.DataFrame:
    """Run all cases on one union mesh and write extraction histories and a summary."""
    base = load_config(ROOT / "base.yaml")
    mesh_config = copy.deepcopy(base)
    mesh_config.model.doublets = [item for items in cases.values() for item in items]
    mesh = generate_mesh(mesh_config)

    histories: dict[str, pd.Series] = {}
    rows: list[dict[str, float | str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, items in cases.items():
        config = copy.deepcopy(base)
        config.model.doublets = copy.deepcopy(items)
        paths = build_model(config, mesh, output_dir / name, budget_save=None)
        result = analyse(paths, run_model(paths), config, mesh)
        temperature = result.timeseries["temperature"]
        histories[name] = temperature.rename(name)
        rows.append(
            {
                "case": name,
                "min_extraction_temperature_c": float(temperature.min()),
                "max_extraction_temperature_c": float(temperature.max()),
                "max_abs_change_from_ambient_c": float(
                    (temperature - AMBIENT_TEMPERATURE_C).abs().max()
                ),
            }
        )

    pd.concat(histories.values(), axis=1).to_csv(output_dir / "extraction_temperature.csv")
    summary = pd.DataFrame(rows).set_index("case")
    summary.to_csv(output_dir / "summary.csv")
    return summary


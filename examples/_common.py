"""Shared simulation, plotting, and reporting helpers for the public examples."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ATESTool.analysis import SimulationResult, analyse
from ATESTool.analysis.report import write_html_report
from ATESTool.config import Config, Doublet, load_config
from ATESTool.mesh import VertexGrid, generate_mesh
from ATESTool.model import build_model, run_model

ROOT = Path(__file__).resolve().parent


@dataclass
class Case:
    """Physical state to run on the comparison's shared mesh."""

    doublets: list[Doublet]
    extraction_flow_rate: float


@dataclass
class Comparison:
    """Results and inputs retained for plots and cross-state calculations."""

    results: dict[str, SimulationResult]
    cases: Mapping[str, Case]
    base: Config
    mesh: VertexGrid
    output_dir: Path


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


def run_comparison(cases: Mapping[str, Case], output_dir: Path) -> Comparison:
    """Run all physical states on one mesh and preserve their analysed results."""
    base = load_config(ROOT / "base.yaml")
    mesh_config = copy.deepcopy(base)
    mesh_config.model.doublets = [
        copy.deepcopy(item) for case in cases.values() for item in case.doublets
    ]
    mesh = generate_mesh(mesh_config)

    results: dict[str, SimulationResult] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, case in cases.items():
        config = copy.deepcopy(base)
        config.model.doublets = copy.deepcopy(case.doublets)
        config.model.extraction_well.flow_rate = case.extraction_flow_rate
        case_dir = output_dir / "states" / name
        paths = build_model(config, mesh, case_dir, budget_save=None)
        result = analyse(paths, run_model(paths), config, mesh)
        results[name] = result

        result.timeseries.to_csv(case_dir / "timeseries.csv")
        result.efficiency.to_csv(case_dir / "recovery.csv")
        result.energy_balance.to_csv(case_dir / "energy.csv", index=False)
        (case_dir / "stats.json").write_text(
            json.dumps(result.stats, indent=2), encoding="utf-8"
        )

    return Comparison(results, cases, base, mesh, output_dir)


def temperature_difference(
    active: SimulationResult, reference: SimulationResult, name: str
) -> pd.Series:
    """Return active minus reference extraction temperature on active saved times."""
    active_series = active.timeseries["temperature"]
    reference_series = reference.timeseries["temperature"]
    times = active_series.index.to_numpy(dtype=float)
    reference_at_times = np.interp(
        times,
        reference_series.index.to_numpy(dtype=float),
        reference_series.to_numpy(dtype=float),
    )
    return pd.Series(
        active_series.to_numpy(dtype=float) - reference_at_times,
        index=active_series.index,
        name=name,
    )


def recovery_comparison(
    with_extraction: SimulationResult, without_extraction: SimulationResult
) -> pd.DataFrame:
    """Match per-well seasonal recovery from the combined and ATES-only states."""
    left = with_extraction.energy_balance[["season", "well", "efficiency"]].rename(
        columns={"efficiency": "with_groundwater_extraction_percent"}
    )
    right = without_extraction.energy_balance[["season", "well", "efficiency"]].rename(
        columns={"efficiency": "without_groundwater_extraction_percent"}
    )
    comparison = left.merge(right, on=["season", "well"], validate="one_to_one")
    comparison["difference_percentage_points"] = (
        comparison["with_groundwater_extraction_percent"]
        - comparison["without_groundwater_extraction_percent"]
    )
    return comparison


def plot_layout(
    ax: Axes,
    doublets: Sequence[Doublet],
    extraction_coordinates: tuple[float, float],
    title: str,
) -> None:
    """Draw the physical well layout and regional-flow direction."""
    ex, ey = extraction_coordinates
    ax.scatter(ex, ey, marker="*", s=130, color="black", label="Extraction", zorder=3)
    for index, item in enumerate(doublets):
        label_warm = "Warm well" if index == 0 else None
        label_cold = "Cold well" if index == 0 else None
        ax.scatter(*item.warm_well, marker="^", s=75, color="tab:red", label=label_warm)
        ax.scatter(*item.cold_well, marker="v", s=75, color="tab:blue", label=label_cold)
        ax.plot(
            [item.warm_well[0], item.cold_well[0]],
            [item.warm_well[1], item.cold_well[1]],
            color="0.65",
            linewidth=1,
        )
    ax.text(
        0.88,
        0.90,
        "regional flow",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="0.35",
    )
    ax.annotate(
        "",
        xy=(0.88, 0.10),
        xytext=(0.88, 0.86),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "0.35"},
    )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)


def save_figure(fig: Figure, path: Path) -> Path:
    """Save one review-ready PNG and release its matplotlib resources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    comparison: Comparison,
    title: str,
    stats: dict[str, float],
    summary: str,
    figures: Sequence[Path],
) -> Path:
    """Create a portable HTML report from the comparison outputs."""
    return write_html_report(
        comparison.output_dir / "report.html",
        stats=stats,
        summary_text=summary,
        plot_paths=figures,
        title=title,
    )

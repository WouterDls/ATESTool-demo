"""Example 1: quantify interaction between ATES and extraction in both directions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (
    ROOT,
    Case,
    doublet,
    plot_layout,
    recovery_comparison,
    run_comparison,
    save_figure,
    temperature_difference,
    write_report,
)
from ATESTool.config import load_config


def main() -> None:
    base = load_config(ROOT / "base.yaml")
    active = doublet(base, "case1", (0.0, 200.0), (0.0, 300.0))
    inactive = doublet(base, "case1", (0.0, 200.0), (0.0, 300.0), 0.0)
    extraction_rate = base.model.extraction_well.flow_rate
    cases = {
        "combined": Case([active], extraction_rate),
        "extraction_only": Case([inactive], extraction_rate),
        "ates_only": Case([active], 0.0),
    }
    output = ROOT / "output" / "e1_material_interference"
    comparison = run_comparison(cases, output)

    delta_temperature = temperature_difference(
        comparison.results["combined"],
        comparison.results["extraction_only"],
        "ates_induced_temperature_change_c",
    )
    recovery = recovery_comparison(
        comparison.results["combined"], comparison.results["ates_only"]
    )
    delta_temperature.to_csv(output / "extraction_temperature_difference.csv")
    recovery.to_csv(output / "recovery_comparison.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_layout(
        axes[0, 0],
        [active],
        base.model.extraction_well.coordinates,
        "(a) Case 1 input layout",
    )
    axes[0, 0].legend(loc="best", fontsize="small")
    axes[0, 1].plot(delta_temperature.index / 365.0, delta_temperature, color="tab:orange")
    axes[0, 1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0, 1].set(
        title="(b) ATES influence on extraction",
        xlabel="time [years]",
        ylabel="temperature change [degC]",
    )
    axes[0, 1].grid(alpha=0.2)

    for ax, suffix, title, panel in (
        (axes[1, 0], "_warm", "Warm-well recovery", "(c)"),
        (axes[1, 1], "_cold", "Cold-well recovery", "(d)"),
    ):
        rows = recovery[recovery["well"].str.endswith(suffix)]
        ax.plot(
            rows["season"],
            rows["with_groundwater_extraction_percent"],
            marker="o",
            color="tab:orange",
            label="With groundwater extraction",
        )
        ax.plot(
            rows["season"],
            rows["without_groundwater_extraction_percent"],
            marker="s",
            color="tab:green",
            label="Without groundwater extraction",
        )
        ax.set(
            title=f"{panel} {title}",
            xlabel="discharge season",
            ylabel="recovery efficiency [%]",
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize="small")

    figure_path = save_figure(figure, output / "figures" / "case1_reciprocal.png")
    stats = {
        "maximum_ates_induced_warming_c": float(delta_temperature.max()),
        "maximum_ates_induced_cooling_c": float(delta_temperature.min()),
        "mean_recovery_with_extraction_percent": float(
            recovery["with_groundwater_extraction_percent"].mean()
        ),
        "mean_recovery_without_extraction_percent": float(
            recovery["without_groundwater_extraction_percent"].mean()
        ),
        "mean_recovery_difference_percentage_points": float(
            recovery["difference_percentage_points"].mean()
        ),
    }
    summary = (
        "State A combines ATES and groundwater extraction; state B retains extraction "
        "but disables ATES; state C retains ATES but disables groundwater extraction.\n"
        "A - B measures the ATES influence on extracted-water temperature.\n"
        "Recovery(A) - Recovery(C) measures the groundwater-extraction influence on ATES."
    )
    report = write_report(
        comparison, "ATESTool Example 1: reciprocal interaction", stats, summary, [figure_path]
    )
    print(f"report: {report}")
    for name, value in stats.items():
        print(f"{name}: {value:.6g}")


if __name__ == "__main__":
    main()

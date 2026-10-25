"""Example 2: compare ATES layouts using a matched no-ATES extraction reference."""

from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (
    ROOT,
    Case,
    doublet,
    plot_layout,
    run_comparison,
    save_figure,
    temperature_difference,
    write_report,
)
from ATESTool.config import load_config


def main() -> None:
    base = load_config(ROOT / "base.yaml")
    layouts = {
        "near_downstream": doublet(base, "near", (0.0, 100.0), (0.0, 200.0)),
        "far_downstream": doublet(base, "far", (0.0, 450.0), (0.0, 550.0)),
        "transverse": doublet(base, "transverse", (-50.0, 250.0), (50.0, 250.0)),
        "reversed_longitudinal": doublet(base, "reversed", (0.0, 300.0), (0.0, 200.0)),
    }
    inactive = doublet(base, "reference", (0.0, 200.0), (0.0, 300.0), 0.0)
    extraction_rate = base.model.extraction_well.flow_rate
    cases = {"without_ates": Case([inactive], extraction_rate)}
    cases.update({name: Case([item], extraction_rate) for name, item in layouts.items()})

    output = ROOT / "output" / "e2_siting_sensitivity"
    comparison = run_comparison(cases, output)
    reference = comparison.results["without_ates"]
    differences = {
        name: temperature_difference(comparison.results[name], reference, name)
        for name in layouts
    }
    histories = pd.concat(differences.values(), axis=1)
    histories.to_csv(output / "extraction_temperature_differences.csv")

    rows = []
    for name, series in differences.items():
        final_year = series[series.index >= float(series.index.max()) - 365.0]
        rows.append(
            {
                "layout": name,
                "maximum_warming_c": float(series.max()),
                "maximum_cooling_c": float(series.min()),
                "maximum_absolute_change_c": float(series.abs().max()),
                "final_year_mean_change_c": float(final_year.mean()),
            }
        )
    summary_table = pd.DataFrame(rows).set_index("layout")
    summary_table.to_csv(output / "summary.csv")

    figure = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=[1.0, 1.35])
    display_names = {
        "near_downstream": "Near longitudinal",
        "far_downstream": "Far longitudinal",
        "transverse": "Transverse",
        "reversed_longitudinal": "Reversed longitudinal",
    }
    for column, (name, item) in enumerate(layouts.items()):
        ax = figure.add_subplot(grid[0, column])
        plot_layout(
            ax,
            [item],
            base.model.extraction_well.coordinates,
            display_names[name],
        )
        if column == 0:
            ax.legend(loc="best", fontsize="x-small")

    ax_response = figure.add_subplot(grid[1, :])
    for name, series in differences.items():
        ax_response.plot(series.index / 365.0, series, label=display_names[name])
    ax_response.axhline(0.0, color="0.5", linewidth=0.8)
    ax_response.set(
        title="ATES-induced extraction-temperature change",
        xlabel="time [years]",
        ylabel="temperature change [degC]",
    )
    ax_response.grid(alpha=0.2)
    ax_response.legend(loc="best", ncol=2)

    figure_path = save_figure(figure, output / "figures" / "case2_layouts.png")
    stats = {
        "largest_warming_across_layouts_c": float(summary_table["maximum_warming_c"].max()),
        "largest_cooling_across_layouts_c": float(summary_table["maximum_cooling_c"].min()),
    }
    summary = (
        "Every layout uses the same aquifer, operating rates, groundwater-extraction "
        "state, and shared mesh. Each curve is the active-layout extraction temperature "
        "minus the matched state without ATES operation. Detailed values are in summary.csv."
    )
    report = write_report(
        comparison, "ATESTool Example 2: layout sensitivity", stats, summary, [figure_path]
    )
    print(summary_table)
    print(f"report: {report}")


if __name__ == "__main__":
    main()

# Quickstart

```python
from ATESTool import run_from_config

# Single deterministic run + compact HTML report (no extra model runs)
run_from_config("examples/base.yaml", workspace="output", report=True)


# Add every plot + the spatial/temporal/analytical verification studies
# (costs ~12 extra model runs; deterministic runs only) -> ./output/diagnostic/
run_from_config("examples/base.yaml", diagnostic=True, workspace="output")
```

The compact report is written to `output/report.html`, with its PNG figures under
`output/plots/`. Omit `report=True` when only the CSV and JSON outputs are needed.
The report includes a conceptual view of the active wells and aquifer layers alongside
the final hydraulic head contours. In a terminal, a progress bar shows the percentage
of simulated time completed and the elapsed solver time. Adaptive runs update the
percentage at stress-period boundaries. Redirected output gets one completion line.

When multiple temperature snapshots are available, the report includes a thermal plume
player. Use Play/Pause or the time slider to inspect the saved fields. The map and
temperature-change scale stay fixed throughout playback.

Follow {doc}`installation` to set up the Python 3.12 environment, the mesh backend,
and MODFLOW 6 before running a model.

See {doc}`config_reference` for input meanings, units, and configuration options.

## Run the examples

With the ATESTool environment active, run each physical state directly from its YAML file:

```bash
ATESTool examples/single_run.yaml --workspace examples/output/e1_single_run --report

ATESTool examples/case1/combined.yaml --workspace examples/output/case1/combined --report
ATESTool examples/case1/extraction_only.yaml --workspace examples/output/case1/extraction_only --report
ATESTool examples/case1/ates_only.yaml --workspace examples/output/case1/ates_only --report

ATESTool examples/case2/without_ates.yaml --workspace examples/output/case2/without_ates --report
ATESTool examples/case2/near_upstream.yaml --workspace examples/output/case2/near_upstream --report
ATESTool examples/case2/far_upstream.yaml --workspace examples/output/case2/far_upstream --report
ATESTool examples/case2/transverse.yaml --workspace examples/output/case2/transverse --report
ATESTool examples/case2/reversed_longitudinal.yaml --workspace examples/output/case2/reversed_longitudinal --report
```

The first command introduces one model and report. Case 1 evaluates both directions of
ATES--groundwater-extraction interaction. Case 2 compares four layouts against a common
no-ATES reference. Every workspace contains `report.html`, `timeseries.csv`, `energy.csv`,
`recovery.csv`, and `stats.json`. Copy the CSV values into the yellow sheets of the supplied
Case 1 or Case 2 `.xlsx` template; visible formulas and editable charts perform the aggregate
analysis. A populated `_solution.xlsx` is included beside each empty template as a worked
reference.
See the examples [README](https://github.com/WouterDls/ATESTool-demo/tree/main/examples) for
the state definitions and output locations.

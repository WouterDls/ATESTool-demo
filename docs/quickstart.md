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

See {doc}`installation` for setting up the two external dependencies (Triangle,
MODFLOW 6) before running a model.

## Run the SoftwareX examples

The repository contains two matched-scenario examples:

```bash
python examples/e1_material_interference.py
python examples/e2_siting_sensitivity.py
```

Example 1 reports interaction in both directions: the effect of ATES on groundwater-
extraction temperature and the effect of groundwater extraction on ATES recovery. Example
2 compares four ATES layouts against the same no-ATES extraction reference. Each command
writes compact CSV tables, a review-ready figure, and a self-contained `report.html` below
`examples/output/`. See the examples [README](https://github.com/WouterDls/ATESTool-demo/tree/main/examples)
for the state definitions and output locations.

# Running the examples

End users run only YAML files. No example-specific Python script is required. Run every
command from the repository root after installing ATESTool with `pip install -e ".[mesh]"`.
Each command uses a separate workspace and creates `report.html`, `timeseries.csv`,
`energy.csv`, `recovery.csv`, `stats.json`, the MODFLOW files, and report figures.

## Example 1 — single model run

```bash
ATESTool examples/single_run.yaml \
  --workspace examples/output/e1_single_run \
  --report
```

Open `examples/output/e1_single_run/report.html`.

## Case 1 — reciprocal interaction

Run the three matched states:

```bash
ATESTool examples/case1/combined.yaml \
  --workspace examples/output/case1/combined --report
ATESTool examples/case1/extraction_only.yaml \
  --workspace examples/output/case1/extraction_only --report
ATESTool examples/case1/ates_only.yaml \
  --workspace examples/output/case1/ates_only --report
```

Each folder has its own report. To reproduce the aggregate analysis in Excel, open
`examples/case1/case1_analysis_template.xlsx` and follow its `Guide` sheet. Copy each
complete CSV, including its header row, and paste **values only** at cell A1 of the matching
yellow sheet:

- `combined/timeseries.csv` → `Combined time`
- `extraction_only/timeseries.csv` → `Extraction time`
- `combined/energy.csv` → `Combined energy`
- `ates_only/energy.csv` → `ATES energy`

The workbook formulas compare the extraction-well temperature and ATES recovery in both
directions. Save the populated workbook under a new name so the clean template remains
reusable. `case1_analysis_solution.xlsx` is a completed copy made from the supplied runs;
use it to check the expected paste layout and calculated results.

## Case 2 — siting sensitivity

Run the common reference and four ATES layouts:

```bash
ATESTool examples/case2/without_ates.yaml \
  --workspace examples/output/case2/without_ates --report
ATESTool examples/case2/near_downstream.yaml \
  --workspace examples/output/case2/near_downstream --report
ATESTool examples/case2/far_downstream.yaml \
  --workspace examples/output/case2/far_downstream --report
ATESTool examples/case2/transverse.yaml \
  --workspace examples/output/case2/transverse --report
ATESTool examples/case2/reversed_longitudinal.yaml \
  --workspace examples/output/case2/reversed_longitudinal --report
```

Open `examples/case2/case2_analysis_template.xlsx` and follow its `Guide` sheet. Paste each
run's complete `timeseries.csv`, including its header row, as values at A1 of the matching
yellow source sheet. The formulas subtract the `Without ATES` reference and update the
comparison chart and summary automatically. `case2_analysis_solution.xlsx` contains the
worked result, while `case2_analysis_template.xlsx` remains empty for reuse.

The YAML files within each comparison case use identical mesh settings and a shared list of
refinement points. This keeps the meshes aligned even when a well is inactive in one state.
The 20-year simulations are substantial; runtime depends on the mesh backend, MODFLOW 6
build, and computer.

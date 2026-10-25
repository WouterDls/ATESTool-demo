# Running examples

The two examples reproduce the application questions retained for the SoftwareX
demonstration. They share the synthetic aquifer definition in `base.yaml`; every physical
state within an example is evaluated on one shared mesh. Generated model files, compact CSV
tables, a figure, and a self-contained HTML report are written below `examples/output/`.

From the repository root, after installing ATESTool, Triangle, and MODFLOW 6 as described
in `INSTALL.md`, run:

```bash
python examples/e1_material_interference.py
python examples/e2_siting_sensitivity.py
```

## Example 1 — reciprocal interaction

Example 1 evaluates three matched states:

- ATES and groundwater extraction active;
- groundwater extraction active without ATES operation;
- ATES active without groundwater extraction.

It reports both directions of interaction: the ATES-induced temperature change at the
groundwater-extraction well, and the groundwater-extraction-induced change in warm- and
cold-well recovery. Open `examples/output/e1_material_interference/report.html` after the
run.

## Example 2 — layout sensitivity

Example 2 compares near longitudinal, far longitudinal, transverse, and reversed
longitudinal ATES layouts while holding the aquifer and operating rates fixed. Every curve
is reported relative to the matched groundwater-extraction state without ATES operation.
Open `examples/output/e2_siting_sensitivity/report.html` after the run.

The complete examples are substantial 20-year simulations. Runtime depends on the mesh
backend, MODFLOW 6 build, and computer. The focused reports do not run the package's
additional spatial, temporal, and domain diagnostic campaigns.

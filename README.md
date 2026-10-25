# ATESTool

ATESTool is a config-driven tool for modelling **ATES (Aquifer Thermal Energy
Storage)** systems and their groundwater interaction. From a single YAML file it
generates an unstructured mesh, builds and runs a **MODFLOW 6** flow (GWF) +
energy-transport (GWE) model through [FloPy](https://github.com/modflowpy/flopy), and
post-processes the results. The YAML config is the only interface: users never call the
mesh, model, or analysis modules directly.

## Installation

Two external pieces of software are **not** installed by `pip`; install them
first (full details in [`INSTALL.md`](INSTALL.md)):

1. **Triangle** mesh generator: via the optional `triangle` PyPI package (a Python
   wrapper around J.R. Shewchuk's Triangle C library):

   ```bash
   pip install triangle          # or: pip install -e ".[mesh]"
   ```

   ATESTool triangulates the domain, then builds the **Voronoi dual** and uses the
   Voronoi polygons as MODFLOW 6 DISV cells (their generators satisfy the CVFD
   orthogonality requirement; wells are generators, so each sits at a cell centre).
   If `triangle` is absent, ATESTool falls back to `scipy.spatial.Delaunay` raw
   triangles (no per-well refinement, no Voronoi dual) with a warning. ⚠️ Triangle is
   **restricted for commercial use**: verify its licence for your context.

2. **MODFLOW 6** binary on your `PATH` (or pointed to by `$MF6_BIN`), needed only
   to *run* models. **GWE requires mf6 ≥ 6.5**, and on Apple Silicon you need an
   **arm64** build.

Then install ATESTool:

```bash
pip install -e .            # runtime
pip install -e ".[dev]"     # + mypy
```

Requires Python ≥ 3.10.

## Quickstart

```python
from ATESTool import run_from_config

# Single deterministic run -> CSV, JSON, figures, and a self-contained HTML report
run_from_config("examples/base.yaml", workspace="output", report=True)

```

Or from the command line:

```bash
ATESTool examples/base.yaml --workspace output
ATESTool examples/base.yaml --workspace output --report
ATESTool my_config.yaml --diagnostic --workspace output   # + plots & verification studies
```

See [`examples/base.yaml`](examples/base.yaml) for a complete single-layer configuration.
The [`examples`](examples/README.md) directory contains two runnable SoftwareX
demonstrations: reciprocal ATES--groundwater-extraction interaction and ATES-layout
sensitivity. Each example writes CSV results, a review-ready figure, and a self-contained
HTML report.

## Time stepping (temporal resolution)

Just as the mesh is auto-refined in space and gated on the cell-**Péclet** number,
the time step is gated on its temporal twin, the **Courant** number `Co = v·Δt/Δx`:
near a well the thermal front must not jump multiple cells per step, or it smears.
Because MODFLOW 6 GWE is implicit (unconditionally stable) this is an *accuracy*
advisory, not a stability limit, but it matters for long 3–25 year runs, where a
hand-picked step that was fine on a coarse mesh now under-resolves the refined one.

Add an optional `time_stepping` block under `model.simulation` to pick the strategy:

```yaml
model:
  simulation:
    n_years: 10
    season_length_days: 182.5
    timesteps_per_season: 10      # used directly by "fixed"; a base/fallback otherwise
    time_stepping:
      mode: courant               # "fixed" (default) | "courant" | "ats"
      target_courant: 1.0         # courant: Co target (also the ats max-step ceiling)
      tsmult: 1.2                 # courant: within-season step growth (auto-1.0 if infeasible)
      ats_dtadj: 2.0              # ats: solver-driven grow/shrink factor
      ats_dtfailadj: 5.0          # ats: step-cut factor on a failed solve
```

- **`fixed`** (default when the block is omitted): uses `timesteps_per_season`
  verbatim; the Courant gate still logs an advisory if that under-resolves the front.
- **`courant`**: derives the steps-per-season from `target_courant`, so accuracy is
  independent of run length. `tsmult` grades the steps (small first step after each
  seasonal flow reversal) when the Courant step is large enough; otherwise it falls
  back to uniform steps automatically.
- **`ats`**: MODFLOW 6 Adaptive Time Stepping: the solver grows/shrinks the step by
  convergence, capped at the same Courant step so the two strategies share one
  accuracy ceiling. Run a case both ways to compare cost vs. accuracy.

## Diagnostic mode (verification bundle)

`--diagnostic` (or `run_from_config(..., diagnostic=True)`) keeps the normal run and its
normal outputs, then adds everything you would otherwise assemble by hand:

- **every plot** for the run: timeseries, efficiency, doublet temperatures, the model
  overview dashboard, and a temperature/head/drawdown snapshot trio per
  `analysis.times_of_interest`;
- the **spatial** convergence study: reruns the model on progressively finer meshes and
  reports whether the outputs have stopped moving;
- the **temporal** convergence study: reruns it at tightening Courant targets, with the
  observed order of accuracy, a Richardson Δt→0 extrapolation, and the GWE mass-balance
  health check;
- the **analytical** benchmark: the modelled thermal front against the closed-form
  energy-balance radius `R_th(t) ∝ √t`, sampled across the first injection season.

Everything lands under `<workspace>/diagnostic/` (plots as PNG, results as both CSV and
JSON, plus a human-readable `summary.txt`); the normal artifacts stay where they were.

Two things to know before using it:

- **It is expensive.** The studies run the full pipeline once per level, which with the
  defaults costs roughly **12 baseline-run-equivalents** (the finest spatial level has ~4×
  the cells; the finest temporal level ~4× the steps). Cost scales linearly with
  `simulation.n_years`, so try it on a short config first. The estimate is logged before
  anything starts.

A stage that fails is recorded in `summary.txt` and the rest continue; a diverging
temporal study does not throw away a spatial study that already cost you five model runs.

Note that the spatial study pins `fixed` time stepping: under `courant`/`ats` the step is
derived from the cell size, so refining the mesh would refine the time step too and the
result would describe both axes at once. The override is recorded in
`spatial_convergence.json`.

## Module & concept overview

| Module | Responsibility |
|---|---|
| `ATESTool.config` | Load and validate deterministic YAML configuration |
| `ATESTool.mesh` | `generate_mesh` → `VertexGrid` (Triangle→Voronoi DISV / scipy fallback) |
| `ATESTool.model` | GWF + GWE setup (`build_model`), execution (`run_model`), `WellSchedule` |
| `ATESTool.analysis` | Parse output behind `SimulationResult`; timeseries/snapshot plots |
| `ATESTool.validation` | Spatial, temporal, and domain convergence studies; health checks; analytical benchmarks; their plots |
| `ATESTool.diagnostic` | Opt-in verification bundle: every plot + all three convergence/benchmark studies → `DiagnosticReport` |
| `ATESTool.run` | `run_from_config` pipeline + `ATESTool` CLI |

| First-class term | What it is |
|---|---|
| `VertexGrid` (`mesh`) | The sole mesh representation; emits FloPy DISV args. Nothing downstream sees raw Triangle output. |
| `WellSchedule` (`model`) | Owns all seasonal sign-flip / warm↔cold phase-reversal logic; GWF & GWE consume this table, not seasons. |
| `SimulationResult` (`analysis`) | The single output-parsing seam: `.timeseries`, `.snapshot(t)`, `.stats`. |
| `Config` (`config`) | All-scalar configuration accepted by mesh, model, analysis, and diagnostics. |

## Validation

```bash
mypy ATESTool          # checks the package's static types
python -m compileall -q ATESTool examples
```

## Documentation

The rendered user and API documentation is available at
[wouterdls.github.io/ATESTool-demo](https://wouterdls.github.io/ATESTool-demo/).

The documentation is built automatically with GitHub Pages. To build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html   # or: (cd docs && make html)
```

Open `docs/_build/html/index.html` in a browser.

A PDF can be built the same way (requires a LaTeX distribution, e.g. TeX Live or
MacTeX, providing `xelatex`/`latexmk` on `PATH`):

```bash
cd docs && make latexpdf
```

Output: `docs/_build/latex/ATESTool.pdf`.

## Licence

ATESTool is released under the MIT License: see [LICENSE.md](LICENSE.md).
Note that the optional Triangle mesh generator carries its own licence with
**commercial-use restrictions**; review it before distributing or using
ATESTool commercially.

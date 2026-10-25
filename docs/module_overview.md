# Module overview

```{list-table}
:header-rows: 1
:widths: 25 75

* - Module
  - Responsibility
* - `ATESTool.config`
  - Load and validate deterministic, scalar-valued YAML configuration
* - `ATESTool.mesh`
  - `generate_mesh` -> `VertexGrid` (Triangle-Voronoi DISV / scipy fallback)
* - `ATESTool.model`
  - GWF + GWE setup (`build_model`), execution (`run_model`), `WellSchedule`
* - `ATESTool.analysis`
  - Parse output behind `SimulationResult`; timeseries/snapshot plots
* - `ATESTool.validation`
  - Spatial, temporal, and domain convergence studies; health checks; analytical benchmarks; their plots
* - `ATESTool.diagnostic`
  - Optional verification bundle: every plot + all three studies -> `DiagnosticReport`
* - `ATESTool.run`
  - `run_from_config` pipeline + `ATESTool` CLI
```

```{toctree}
:hidden:
:maxdepth: 2

installation
config_reference
cli
```

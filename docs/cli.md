# Command-line interface

Installing ATESTool (see {doc}`installation`) provides the `ATESTool` console command,
a thin wrapper around {func}`~ATESTool.run.run_from_config`:

```bash
ATESTool <config> [--workspace WORKSPACE] [--diagnostic]
```

`<config>` is the only required argument: the path to a deterministic YAML config file
(see {doc}`config_reference`).

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--workspace WORKSPACE` | `atestool_output` | Output directory. Created if it does not exist. |
| `--diagnostic` | off | Write every plot and run the spatial, temporal, and domain-size convergence studies plus the analytical benchmark into `<workspace>/diagnostic/`. This costs several extra model runs; see {doc}`api/ATESTool.diagnostic`. |

## Outputs

The run writes these artifacts directly to `<workspace>/`:

- `timeseries.csv`: the extraction well's timeseries.
- `stats.json`: scalar summary statistics, including peak temperature, drawdown,
  breakthrough time, efficiency, energy balance, ambient drift, and boundary influence.
- `observations.csv`: written when `analysis.observation_points` is non-empty.
- `diagnostic/`: written only with `--diagnostic`; contains plots, convergence studies,
  the analytical benchmark, machine-readable results, and an HTML report.

## Python API equivalent

```python
from ATESTool import run_from_config

run_from_config("examples/base.yaml", workspace="output", diagnostic=True)
```

See {func}`ATESTool.run.run_from_config` for the full parameter list.

# Quickstart

```python
from ATESTool import run_from_config

# Single deterministic run -> writes timeseries.csv + stats.json to ./output
run_from_config("examples/base.yaml", workspace="output")


# Add every plot + the spatial/temporal/analytical verification studies
# (costs ~12 extra model runs; deterministic runs only) -> ./output/diagnostic/
run_from_config("examples/base.yaml", diagnostic=True, workspace="output")
```

See {doc}`installation` for setting up the two external dependencies (Triangle,
MODFLOW 6) before running a model.

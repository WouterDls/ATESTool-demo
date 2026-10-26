# Config reference

Every ATESTool run starts from one deterministic YAML file. Numeric fields accept scalar
numbers only. `load_config()` validates the structure and returns a {class}`ATESTool.config.Config`
consumed directly by the mesh, model, analysis, and diagnostic modules.

`examples/base.yaml` is the annotated, runnable reference configuration.

## YAML scientific notation

YAML requires an explicit exponent sign for a value to parse as a number:

```yaml
specific_storage: 4.18e6     # wrong: parsed as text by PyYAML
specific_storage: 4.18e+6    # correct
specific_storage: 4180000.0  # also correct
```

ATESTool raises `ConfigError` with the field path when a numeric field is not a scalar.

## `mesh`

The domain is centred on the origin and spans `[-width/2, width/2]` by
`[-height/2, height/2]`.

| Field | Type / unit | Default | Notes |
|---|---|---|---|
| `width`, `height` | number, m | required | Full domain extents. |
| `refinement_rings` | list of `{radius, max_cell_size}`, m | required | Applied around every well; may be empty. |
| `thermal_refinement` | `manual` or `auto` | `manual` | `auto` adds a thermal-radius ring around each doublet well. |
| `background_max_cell_size` | number, m | derived | Far-field target cell size; omitted means `min(width, height) / 20`. |
| `refinement_points` | list of `[x, y]`, m | `[]` | Optional mesh-only refinement centres. Use the same list in matched scenarios to keep their meshes comparable without adding inactive model wells. |

## `model.aquifer.layers`

`layers` is a non-empty, top-down list. Every layer requires:

- `top`, `bottom` [m]
- `hydraulic_conductivity` [m/day]
- `porosity` [-] and `specific_storage` [1/m]
- `initial_temperature` [degC]
- `longitudinal_dispersivity`, `transverse_dispersivity` [m]
- `thermal_conductivity_water`, `thermal_conductivity_solid` [W/m/K]
- `heat_capacity_water`, `heat_capacity_solid` [J/kg/K]
- `density_water`, `density_solid` [kg/m³]

Wells fully penetrate the aquifer and their flow is split across layers by transmissivity.
Two-dimensional reporting uses the layer with the highest transmissivity weight.

## `model.chd_heads`

`north`, `south`, `east`, and `west` are required constant-head boundary values in metres.
Optional `mode` is `sides` (default, one constant head per side) or `linear`
(a regional head plane, with the four values interpreted as edge-midpoint heads).
Linear mode requires `north + south = east + west` and evaluates the plane at each
boundary cell centre. Domain convergence uses linear mode with a fixed regional gradient.
Each boundary cell also carries its layer's initial temperature so inflow enters at ambient
temperature.

## `model.recharge`

`recharge` is an optional areal flux in m/day and defaults to `0.0`. A positive value adds
a top-layer GWF RCH package and a matching GWE source at the top layer's ambient temperature.

## `model.doublets`

At least one doublet is required. Each entry contains:

| Field | Type / unit | Notes |
|---|---|---|
| `name` | string | Used in outputs; should be unique. |
| `warm_well`, `cold_well` | `[x, y]`, m | Well coordinates. |
| `flow_rate` | number, m³/day | Balanced seasonal rate for the pair. |
| `warm_injection_temperature` | number, degC | Warm-season injected temperature. |
| `cold_injection_temperature` | number, degC | Cold-season injected temperature. |

For equal-length seasons, annual energy balance occurs when the sum of the two injection
temperatures equals twice the reporting layer's ambient temperature.

## `model.extraction_well`

`coordinates` is an `[x, y]` pair in metres and `flow_rate` is in m³/day. The well is always
treated as extraction; the sign is fixed internally.

## `model.simulation`

| Field | Type / unit | Notes |
|---|---|---|
| `n_years` | number, years | Simulation duration. |
| `season_length_days` | number, days | Length of each charge/discharge season. |
| `timesteps_per_season` | number | Used directly in fixed mode. |
| `time_stepping` | mapping | Optional settings below. |

The optional `time_stepping` mapping accepts:

| Field | Default | Notes |
|---|---|---|
| `mode` | `fixed` | `fixed`, `courant`, or `ats`. |
| `target_courant` | `1.0` | Courant target and ATS maximum-step ceiling. |
| `tsmult` | `1.2` for Courant, otherwise `1.0` | Geometric step growth. |
| `ats_dtadj` | `2.0` | ATS grow/shrink factor. |
| `ats_dtfailadj` | `5.0` | ATS step-cut factor after failure. |

`fixed` uses the configured step count. `courant` derives a step count that resolves the
near-well thermal front. `ats` lets MODFLOW 6 adapt the step while respecting the same
Courant ceiling.

## `analysis`

| Field | Type / unit | Default | Notes |
|---|---|---|---|
| `output_variables` | list of `temperature` and/or `head` | required | Fields requested for analysis. |
| `times_of_interest` | list of numbers, days | required | Snapshot times. |
| `observation_points` | list of `{name, x, y}` | `[]` | Optional named sampling points written to `observations.csv`. |

## Known limitations

- The schedule contains two seasons per year.
- `chd_heads` is static for the whole run.

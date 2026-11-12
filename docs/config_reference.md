# YAML input guide

Every ATESTool run starts from one deterministic YAML file. Numeric fields accept scalar
numbers only. `load_config()` validates the structure and returns a {class}`ATESTool.config.Config`
consumed directly by the mesh, model, analysis, and diagnostic modules.

Start with `examples/single_run.yaml` for a five-year example with one ATES
doublet and one extraction well. Make a copy and change one group of settings at
a time. Keep the original run in its own workspace so you can compare the reports.

| To change | Edit |
|---|---|
| Where the wells sit and how much water they move | `model.doublets` and `model.extraction_well`. |
| Aquifer properties or the direction of regional flow | `model.aquifer.layers` and `model.chd_heads`. |
| Cell sizes near the wells and farther away | `mesh.refinement_rings` and `mesh.background_max_cell_size`. |
| Run duration or timestep size | `model.simulation`. Review `analysis.times_of_interest` if you shorten the run. |

Coordinates are in metres relative to the model centre. Positive `x` is east
and positive `y` is north. A change in boundary heads, well positions or
pumping rates can also change the flow near the wells, so check the layout and
results in the report after each change.

## YAML scientific notation

YAML requires an explicit exponent sign for a value to parse as a number:

```yaml
heat_capacity_water: 4.18e3   # wrong: parsed as text by PyYAML
heat_capacity_water: 4.18e+3  # correct
heat_capacity_water: 4180.0   # also correct
```

ATESTool raises `ConfigError` with the field path when a numeric field is not a scalar.

## `mesh`

The domain is centred on the origin and spans `[-width/2, width/2]` by
`[-height/2, height/2]`.

| Field | Type / unit | Default | Notes |
|---|---|---|---|
| `width`, `height` | number, m | required | Full domain extents. |
| `refinement_rings` | list of `{radius, max_cell_size}`, m | required | Applied around every well and any refinement points. Each radius is measured from its centre; smaller target cell sizes increase local resolution. Use `[]` for no manual rings. |
| `thermal_refinement` | `manual` or `auto` | `manual` | `auto` adds a thermal-radius ring around each doublet well. |
| `background_max_cell_size` | number, m | derived | Far-field target cell size; omitted means `min(width, height) / 20`. |
| `refinement_points` | list of `[x, y]`, m | `[]` | Optional mesh-only refinement centres. Use the same list in matched scenarios to keep their meshes comparable without adding inactive model wells. |

## `model.aquifer.layers`

`layers` is a non-empty, top-down list. Each layer requires every field below.

| Field | Unit | Meaning |
|---|---|---|
| `top`, `bottom` | m | Elevations of the top and bottom of the layer, measured from the same datum. |
| `hydraulic_conductivity` | m/day | How readily groundwater flows through the layer. |
| `porosity` | fraction | Fraction of the layer volume occupied by pore water. Enter `0.30` for 30%. |
| `specific_storage` | 1/m | Water released per unit aquifer volume per unit decline in hydraulic head. |
| `initial_temperature` | °C | Starting groundwater temperature in the layer. Regional inflow uses this temperature too. |
| `longitudinal_dispersivity`, `transverse_dispersivity` | m | Mechanical spreading along and across the groundwater flow direction, respectively. |
| `thermal_conductivity_water`, `thermal_conductivity_solid` | W/(m·K) | Heat conduction properties of the water and solid matrix. |
| `heat_capacity_water`, `heat_capacity_solid` | J/(kg·K) | Energy needed to warm a kilogram of water or solid matrix by one degree. |
| `density_water`, `density_solid` | kg/m³ | Mass per unit volume of water and solid matrix. |

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

The introductory example sets north to 10.5 m and south to 9.5 m, with east and
west both at 10.0 m. This drives regional flow from north to south. If you
change the heads in `linear` mode, keep the opposite-side sums equal.

## `model.recharge`

`recharge` is an optional areal flux in m/day and defaults to `0.0`. A positive value adds
a top-layer GWF RCH package and a matching GWE source at the top layer's ambient temperature.

## `model.doublets`

At least one doublet is required. Each entry contains:

| Field | Type / unit | Notes |
|---|---|---|
| `name` | string | Used in outputs; should be unique. |
| `warm_well`, `cold_well` | `[x, y]`, m | Well coordinates. |
| `flow_rate` | number, m³/day | Enter a positive pumping magnitude per well. One well injects and the other extracts at this rate. Set to `0` for an inactive doublet in a comparison. |
| `warm_injection_temperature` | number, degC | Warm-season injected temperature. |
| `cold_injection_temperature` | number, degC | Cold-season injected temperature. |

The first season injects at the warm well and extracts at the cold well.
The following season reverses the roles.

For equal-length seasons, annual energy balance occurs when the sum of the two injection
temperatures equals twice the reporting layer's ambient temperature.

## `model.extraction_well`

`coordinates` is an `[x, y]` pair in metres and `flow_rate` is in m³/day. The well is always
treated as extraction. Enter a positive pumping magnitude or `0` to turn it off.

## `model.simulation`

| Field | Type / unit | Notes |
|---|---|---|
| `n_years` | number, years | Use a whole number of years. Each year contains two seasons. |
| `season_length_days` | number, days | Length of each charge/discharge season. |
| `timesteps_per_season` | number | Steps per season in fixed mode. This also provides the base count for ATS and a fallback for Courant mode. |
| `time_stepping` | mapping | Optional settings below. |

The optional `time_stepping` mapping accepts:

| Field | Default | Notes |
|---|---|---|
| `mode` | `fixed` | `fixed`, `courant`, or `ats`. |
| `target_courant` | `1.0` | Courant target and ATS maximum-step ceiling. |
| `tsmult` | `1.2` for Courant, otherwise `1.0` | Geometric step growth. |
| `ats_dtadj` | `2.0` | ATS grow/shrink factor. |
| `ats_dtfailadj` | `5.0` | ATS step-cut factor after failure. |

- **Fixed** uses the configured step count. With `tsmult: 1.0`, steps within
  a season are equal. The workshop example uses 48 steps per season.
- **Courant** estimates a maximum step from the near-well flow and cell sizes and
  chooses a step count to meet that target. If geometric growth would exceed the
  ceiling, ATESTool can use uniform steps.
- **ATS** lets MODFLOW 6 adapt steps based on solver convergence, with the same
  estimated Courant ceiling.

For a timestep comparison, keep the mesh and duration the same and use
`time_stepping: {mode: courant, target_courant: 1.0, tsmult: 1.0}` under
`model.simulation`. Run in a separate workspace and compare extraction
temperatures and seasonal recovery. The Courant target is a resolution check.
Mesh and timestep sensitivity determine how much the results depend on the
discretization.

## `analysis`

| Field | Type / unit | Default | Notes |
|---|---|---|---|
| `output_variables` | list of `temperature` and/or `head` | required | Use `[temperature, head]` as in the examples. The current runner saves both fields for the standard results and report. |
| `times_of_interest` | list of numbers, days | required | Times for spatial plots in diagnostic runs. Keep them within the simulation period or use `[]` for the final saved time. The HTML animation samples saved temperatures independently. |
| `observation_points` | list of `{name, x, y}` | `[]` | Optional named locations in metres in the model coordinate system. ATESTool samples the temperature of the nearest cell and writes `observations.csv`. |

## Known limitations

- The schedule contains two seasons per year.
- `chd_heads` is static for the whole run.

# ATESTool

ATESTool is a config-driven tool for modelling **ATES (Aquifer Thermal Energy Storage)**
systems, including interaction between multiple systems within a shared groundwater
extraction area. From a single YAML config-file it generates an unstructured mesh, builds
and runs a **MODFLOW 6** flow (GWF) + energy-transport (GWE) model through
[FloPy](https://github.com/modflowpy/flopy), and post-processes the results. The YAML config is the only interface:
users never call the `mesh`, `model`, or `analysis` modules directly; see {doc}`api` for
what each module exposes.

```{toctree}
:hidden:
:maxdepth: 2

quickstart
module_overview
meshing_primer
api
```

## Abbreviations

| Abbreviation | Meaning |
|---|---|
| ATES | Aquifer Thermal Energy Storage |
| GWF | groundwater flow (the MODFLOW 6 flow model) |
| GWE | groundwater energy transport (the MODFLOW 6 heat-transport model) |
| DISV | MODFLOW 6's vertex (unstructured) grid discretisation package |
| CVFD | control-volume finite-difference (the numerical scheme DISV relies on) |
| CHD | constant-head (the MODFLOW 6 boundary package used for the domain edges) |
| IMS | Iterative Model Solution (MODFLOW 6's numerical solver) |
| TDIS | Temporal Discretisation (the MODFLOW 6 package that defines stress periods and time steps) |
| ATS | Adaptive Time Stepping (a MODFLOW 6 `time_stepping.mode`) |
| CFL | Courant-Friedrichs-Lewy (the classical stability condition for explicit numerical schemes) |
| CLI | Command-Line Interface |
| GJ | gigajoule (10^9 joules) |

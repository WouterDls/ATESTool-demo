# Meshing & verification rationale

This page explains why the numbers this tool produces can be trusted, and collects the
verification scripts (`ATESTool.mesh.quality`, `ATESTool.model.timestepping`,
`ATESTool.validation.convergence` / `temporal_convergence` / `analytical`) in one place.

## 1. Why Voronoi, not raw Delaunay triangles

MODFLOW 6's DISV grid needs the line between two cell centres to cross their shared
face at a right angle, through its midpoint; a raw Delaunay triangle's centroid usually
doesn't sit there, but a Voronoi cell's generator point always does.

```{image} _static/voronoi_vs_delaunay.png
:alt: Delaunay triangle centroid versus Voronoi cell generator, showing which one sits on the perpendicular bisector of the shared face
:width: 100%
```

MODFLOW 6's DISV grid uses a **control-volume finite-difference (CVFD)** scheme, and
that perpendicular-bisector property is exactly what its inter-cell conductance formula
assumes. Using triangle centroids as DISV cell centres therefore biases the numerics:
conductances are computed for a geometry the actual cells don't have.

The fix is the standard one: build the **Voronoi diagram** dual to the Delaunay
triangulation and use the Voronoi *cells* as the DISV cells, with the Voronoi
*generators* (the well/seed points fed into the triangulation) as the cell centres.
`ATESTool.mesh.generator.generate_mesh` builds a constrained-Delaunay triangulation
with `triangle` (wells and graded refinement seed points as nodes), takes its Voronoi
dual, and clips it to the domain; the scipy fallback (when `triangle` isn't installed)
is raw triangles and does **not** have this property, which is why it logs a warning
and skips refinement-ring support rather than pretending to be equivalent.

**Measured outcome:** mesh orthogonality of 0-6 degrees and a flow/transport
mass-balance discrepancy of 0.0% on the minimal case and minimal config. That's the
empirical confirmation that the construction actually delivers the CVFD property it's
built for, not just a theoretical argument.

### Graded near-well sizing

Refinement rings and the `thermal_refinement="auto"` ring size cells more finely near
each well and relax outward. The relaxation uses a fixed **expansion ratio**
(`_EXPANSION_RATIO = 1.4` in `mesh/generator.py`) between successive grading steps.
That value is a deliberate, near-optimal middle ground: **1.3-1.5** grades smoothly
without adding too many cells, while a ratio above roughly 2 introduces large
conductance contrasts between adjacent cells (a big cell next to a tiny one), which
hurts IMS solver conditioning independent of any accuracy question.

## 2. The quality gate

`ATESTool.mesh.quality.assess_quality` computes four per-mesh metrics, each with a
*warn* threshold and (for two of them) a stricter *raise* threshold:

| Metric | Warn | Raise | Kind |
|---|---|---|---|
| Min interior angle | < 20° | < 5° | hard |
| Max orthogonality deviation | > 30° | > 45° | hard |
| Max aspect ratio | > 10 | n/a | advisory only |
| Max bisection error | > 0.30 (fraction of face length) | n/a | advisory only |

This isn't a contradiction with §1's perpendicular-bisector claim: that property holds
exactly for an *unclipped* Voronoi cell. Clipping the diagram to the domain boundary
(and ordinary floating-point rounding) perturbs boundary cells away from that ideal,
which is why measured orthogonality deviation is small but non-zero (0-6 degrees on
the minimal case, see above) rather than exactly zero, and why the gate checks it
rather than assuming it.

Only **orthogonality** and **minimum angle** gate hard (raise `MeshQualityError`,
refusing to build the model). Both are direct proxies for CVFD correctness itself
(§1): near-zero orthogonality deviation is what makes the conductance formula valid,
and a near-degenerate angle breaks the cell's geometry outright. Aspect ratio and
bisection error are demoted to advisories because a legitimately graded mesh (coarse
far field, fine near the wells) *will* have a handful of high-aspect transition cells
and small faces at the domain boundary or at fine/coarse transitions; those carry
negligible flow and don't threaten correctness, so failing the build over them would
punish exactly the refinement behaviour the mesh is supposed to have.

The hard raise applies only to the Voronoi path. The scipy fallback is already a
degraded, un-refined raw-triangle mesh (logged loudly at generation time, §1), so
gating it further would just be redundant noise; it warns instead.

## 3. Cell Péclet: near-well transport resolution

The quality gate above verifies *geometry*; it does not tell you whether the mesh is
*fine enough* to resolve advection-dominated heat transport. `assess_peclet` adds that
check: the grid (cell) Péclet number

```
Pe = Δx / α_L
```

`Pe` is dimensionless. Small `Pe` means the cell is small relative to the dispersion
length scale, so the front is resolved within a cell; large `Pe` means the cell is
coarse relative to that scale, so the front smears. `Pe` compares the local cell size
to the longitudinal dispersivity (`α_L`, the length scale over which mechanical mixing
spreads a solute front along the flow direction; the advection-dominated, conservative
bound, since ignoring thermal diffusion only makes `Pe` larger, i.e. the check
stricter).

```{image} _static/peclet_resolution.png
:alt: Well-resolved (low Péclet) versus poorly-resolved (high Péclet) thermal front
:width: 100%
```

`Pe ≲ 2` keeps numerical dispersion from smearing the thermal front; `Pe > 4` is
flagged as poor. This is checked **only within the refinement rings around each
well** (the region the thermal plume actually lives in), because the coarse far field
legitimately, and harmlessly, exceeds the threshold. Like the quality gate's advisory
metrics, this is a warning, not a hard limit: a high-Péclet mesh still runs and still
conserves mass (MODFLOW 6's own inflow/outflow/storage bookkeeping still balances), it
just numerically smears the front more than a finer mesh would.

## 4. Courant number: an accuracy limit, not a stability one

`ATESTool.model.timestepping` is the **temporal twin** of the cell-Péclet check: where
Péclet asks "is the mesh fine enough in space", the Courant number

```
Co = v * Δt / Δx
```

asks the same question in time, using the same near-well seepage velocity (the average
pore-water velocity, `v = Q / (2π r b θ)`, distinct from the Darcy/specific-discharge
velocity) that dominates transport near a well.

**This is an accuracy *advisory*, not a stability limit**, and that distinction matters
enough to repeat: MODFLOW 6's GWE solver is **implicit** (IMS) (it solves the governing
equations at the end of each time step, which is what buys unconditional stability),
hence *unconditionally stable* (it will not produce invalid/blown-up numbers regardless
of step size): a large Courant number does not blow the simulation up. What it does is
smear the thermal front in time, exactly as a coarse Péclet smears it in space. Treating
Courant as a hard CFL-style stability ceiling (the classical meaning for *explicit*
schemes) would be the wrong mental model here; `gate_courant` warns, it never raises.

Three `time_stepping.mode` strategies build on this:

- **`fixed`**: `timesteps_per_season` verbatim; not recommended, kept because it's the
  easiest mode to understand. It will still run and the gate will warn when the step
  under-resolves the front.
- **`courant`**: the step count is derived from a target Courant number, so accuracy
  stays roughly constant regardless of `n_years`/`season_length_days`.
- **`ats`**: MODFLOW 6 Adaptive Time Stepping, with its maximum step capped at the
  Courant step so `courant` and `ats` share one accuracy metric and remain comparable.

## 5. Thermal radius as a meshing target

The ATES thermal radius `R_th`, the radius reached by the thermal front from one
season's radial injection, from an energy balance between the heat carried by the
injected water and the heat stored in the swept cylinder of aquifer
(`ATESTool.mesh.thermal.thermal_front_radius`), is the natural length scale for
"is the mesh fine enough to see the front move". The guideline used throughout this
tool is **10-20 cells across `R_th`** (`THERMAL_FRONT_DIVISIONS = 18.0`), evaluated
over a **front band** of `0.8`-`1.2 * R_th` (`THERMAL_FRONT_RADIUS_FACTOR = 1.2`), wide
enough to catch the front whichever side of the nominal radius it's actually on at the
check time.

`mesh.thermal` is deliberately dependency-free (floats and numpy only) so both
`mesh.generator` (which sizes the `thermal_refinement="auto"` ring against `R_th`) and
`validation.analytical` (which uses the same formula as the closed-form benchmark, §6)
import the *same* definition rather than risking two formulas drifting apart. The
`auto` refinement mode adds a ring sized to each ATES doublet well's own `R_th` on top
of the configured `refinement_rings`; the standalone extraction well is never given
one, since it has no injection temperature and therefore no thermal radius to resolve.

## 6. Verification: convergence and the analytical benchmark

Sections 1-5 establish that the mesh and time step are *well-formed* and *plausibly
fine enough*. Neither proves the model is *accurate*: a geometrically pristine,
low-Péclet, low-Courant mesh can still be wrong if it simply isn't fine enough for the
physics at hand. Two independent checks close that gap, and this section is where those
accuracy tests live.

### 6.1 Grid and temporal convergence

`ATESTool.validation.convergence.run_convergence` reruns the full pipeline at
progressively finer meshes (`generate_mesh`'s `refinement_factor`) and reports how key
outputs (an off-well observation head, thermal breakthrough, peak temperature) change
between the two finest levels. `ATESTool.validation.temporal_convergence` does the same
along the *time* axis instead, tightening the Courant target at a **fixed** mesh. Both
apply relative-change tolerances with absolute floors (5% / 0.01 m for hydraulic
metrics; 10% with metric-specific floors for transport). Spatial drawdown is now
sampled at one fixed off-well coordinate; extraction-cell drawdown is a diagnostic.
Only the temporal study reports observed order and Richardson extrapolation; the
spatial verdict measures stabilization between its two finest levels.

Spatial studies pin fixed uniform time stepping internally, while temporal studies
hold the mesh fixed. A stable spatial result still needs a separate time-step check.
Domain studies preserve background resolution targets and a regional head plane,
and compare pumped heads against an otherwise identical no-well reference. They
can still fail when the outer boundary controls pumping drawdown.

The metric definitions and convergence workflow are documented in the API reference.

### 6.2 Analytical benchmarks

`ATESTool.validation.analytical` supplies closed-form solutions that bound a run's
*absolute* error against known physics: the one check neither the quality gate nor a
convergence study can give, since both only measure *relative* change between mesh/step
levels, never distance from ground truth:

- `thiem_drawdown`: steady-state radial flow to/from a well (Thiem, 1906).
- `theis_drawdown`: transient radial flow (Theis, 1935).
- `thermal_front_radius`: the ATES thermal radius (§5 for the energy-balance
  derivation), from an energy balance with thermal retardation.
- `thermal_front_trajectory`: that radius over a series of times, `R_th(t) ∝ √t` (§5
  for the energy-balance derivation); the **transient** benchmark, since the front's
  arrival *over time* is controlled by the time step, not just the mesh, so it
  validates the temporal discretisation too, not only space.

These are idealised (single well, homogeneous confined aquifer, radial symmetry,
distant boundaries); compare them to a model only where those assumptions
approximately hold: near a well, early in time, away from other wells and boundaries.
They are a bound, not a full-field truth reference.

## 7. Mass balance proves conservation, not accuracy

This is the one sentence that undersells everything above if forgotten: a
mass-balance (or energy-balance) discrepancy near 0% proves that MODFLOW 6's own
bookkeeping closes for a simulation, e.g. that inflow, outflow, and storage change add
up. It says **nothing** about whether the modelled heads or temperatures are close to
the true physical answer. A badly under-resolved mesh or time step can still conserve
mass/energy essentially perfectly while smearing the thermal front by tens of percent
(§4's fixed-vs-resolved recovery-efficiency comparison had a discrepancy
indistinguishable from the well-resolved run). **Accuracy is what the convergence
studies (§6.1) and the analytical benchmark (§6.2) establish; mass balance only rules
out a broken run.**

Concretely: `DISCREPANCY_TOL = 1.0` (percent) is the shared tolerance
(`ATESTool.validation.health`) used by the temporal/domain convergence studies and the
deterministic run health check (`RunHealth.ok`) to flag a *broken* run, not to grade
its accuracy. The GWF IMS solver's `dvclose` tolerances were also tightened below the
MODERATE defaults specifically to keep the flow discrepancy under 0.1% on unstructured
DISV grids with strong well sinks/sources; the untightened default left a ~1.5%
discrepancy that came purely from solver slack, not from the mesh, a reminder that a
mass-balance number can also reflect solver settings unrelated to discretisation
quality.

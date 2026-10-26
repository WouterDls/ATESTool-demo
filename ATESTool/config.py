"""Deterministic YAML configuration schema for ATESTool."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

__all__ = [
    "AnalysisConfig", "AquiferConfig", "AquiferLayer", "ChdHeads", "Config",
    "ConfigError", "Doublet", "ExtractionWell", "MeshConfig",
    "ModelConfig", "ObservationPoint", "RefinementRing", "SimulationConfig",
    "TimeSteppingConfig", "load_config",
]


class ConfigError(ValueError):
    """Raised when a config file is missing required fields or is malformed."""


@dataclass
class RefinementRing:
    radius: float
    max_cell_size: float


@dataclass
class MeshConfig:
    width: float
    height: float
    refinement_rings: list[RefinementRing]
    thermal_refinement: str = "manual"
    background_max_cell_size: float | None = None
    refinement_points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class AquiferLayer:
    top: float
    bottom: float
    hydraulic_conductivity: float
    porosity: float
    specific_storage: float
    initial_temperature: float
    longitudinal_dispersivity: float
    transverse_dispersivity: float
    thermal_conductivity_water: float
    thermal_conductivity_solid: float
    heat_capacity_water: float
    density_water: float
    heat_capacity_solid: float
    density_solid: float


@dataclass
class AquiferConfig:
    layers: list[AquiferLayer]


@dataclass
class ChdHeads:
    north: float
    south: float
    east: float
    west: float
    mode: str = "sides"

    @property
    def reference_head(self) -> float:
        return (self.north + self.south + self.east + self.west) / 4.0

    def regional_head(self, x: float, y: float, width: float, height: float) -> float:
        """Planar regional head; side values denote edge-midpoint heads in linear mode."""
        return (self.reference_head + (self.east - self.west) * x / width
                + (self.north - self.south) * y / height)

    def validate_linear(self) -> None:
        """Four edge-midpoint heads must describe one plane, not conflicting datums."""
        import math

        if not math.isclose(self.north + self.south, self.east + self.west,
                            rel_tol=0.0, abs_tol=1e-8):
            raise ConfigError("linear regional heads require north + south = east + west")


@dataclass
class Doublet:
    name: str
    warm_well: tuple[float, float]
    cold_well: tuple[float, float]
    flow_rate: float
    warm_injection_temperature: float
    cold_injection_temperature: float


@dataclass
class ExtractionWell:
    coordinates: tuple[float, float]
    flow_rate: float


@dataclass
class TimeSteppingConfig:
    mode: str
    target_courant: float
    tsmult: float
    ats_dtadj: float
    ats_dtfailadj: float


@dataclass
class SimulationConfig:
    n_years: float
    season_length_days: float
    timesteps_per_season: float
    time_stepping: TimeSteppingConfig | None = None


@dataclass
class ModelConfig:
    aquifer: AquiferConfig
    chd_heads: ChdHeads
    doublets: list[Doublet]
    extraction_well: ExtractionWell
    simulation: SimulationConfig
    recharge: float


@dataclass
class ObservationPoint:
    """A named point sampled into ``observations.csv``."""

    name: str
    x: float
    y: float


@dataclass
class AnalysisConfig:
    output_variables: list[str]
    times_of_interest: list[float]
    observation_points: list[ObservationPoint] = field(default_factory=list)


@dataclass
class Config:
    """The complete deterministic ATESTool configuration."""

    mesh: MeshConfig
    model: ModelConfig
    analysis: AnalysisConfig


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _require(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{_join(path, key)}: required field is missing")
    return mapping[key]


def _section(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected a mapping/section, got {type(value).__name__}")
    return value


def _scalar(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                pass
            else:
                raise ConfigError(
                    f"{path}: value {value!r} was read as text, not a number. In YAML, "
                    "scientific notation needs an explicit exponent sign (e.g. 4.18e+6, "
                    "not 4.18e6), or use a plain decimal."
                )
        raise ConfigError(f"{path}: expected a number, got {value!r}")
    return float(value)


def _leaf(mapping: Mapping[str, Any], key: str, path: str) -> float:
    return _scalar(_require(mapping, key, path), _join(path, key))


def _leaf_or(mapping: Mapping[str, Any], key: str, path: str, default: float) -> float:
    return default if key not in mapping else _scalar(mapping[key], _join(path, key))


def _coords(mapping: Mapping[str, Any], key: str, path: str) -> tuple[float, float]:
    raw = _require(mapping, key, path)
    full = _join(path, key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ConfigError(f"{full}: expected a coordinate pair [x, y]")
    return (_scalar(raw[0], f"{full}[0]"), _scalar(raw[1], f"{full}[1]"))


_THERMAL_REFINEMENT_MODES: Final = ("manual", "auto")
_TIME_STEPPING_MODES: Final = ("fixed", "courant", "ats")


def _parse_mesh(d: Mapping[str, Any]) -> MeshConfig:
    rings_raw = _require(d, "refinement_rings", "mesh")
    if not isinstance(rings_raw, list):
        raise ConfigError("mesh.refinement_rings: expected a list")
    rings: list[RefinementRing] = []
    for i, ring in enumerate(rings_raw):
        path = f"mesh.refinement_rings[{i}]"
        values = _section(ring, path)
        rings.append(RefinementRing(_leaf(values, "radius", path), _leaf(values, "max_cell_size", path)))
    thermal_refinement = d.get("thermal_refinement", "manual")
    if thermal_refinement not in _THERMAL_REFINEMENT_MODES:
        raise ConfigError(
            f"mesh.thermal_refinement: expected one of {list(_THERMAL_REFINEMENT_MODES)}, "
            f"got {thermal_refinement!r}"
        )
    background = (
        _scalar(d["background_max_cell_size"], "mesh.background_max_cell_size")
        if "background_max_cell_size" in d else None
    )
    points_raw = d.get("refinement_points", [])
    if not isinstance(points_raw, list):
        raise ConfigError("mesh.refinement_points: expected a list")
    refinement_points = [
        _coords({"point": point}, "point", f"mesh.refinement_points[{i}]")
        for i, point in enumerate(points_raw)
    ]
    return MeshConfig(
        _leaf(d, "width", "mesh"), _leaf(d, "height", "mesh"), rings,
        str(thermal_refinement), background, refinement_points,
    )


def _parse_time_stepping(sim: Mapping[str, Any]) -> TimeSteppingConfig | None:
    if "time_stepping" not in sim:
        return None
    path = "model.simulation.time_stepping"
    values = _section(sim["time_stepping"], path)
    mode = values.get("mode", "fixed")
    if mode not in _TIME_STEPPING_MODES:
        raise ConfigError(f"{path}.mode: expected one of {list(_TIME_STEPPING_MODES)}, got {mode!r}")
    return TimeSteppingConfig(
        str(mode),
        _leaf_or(values, "target_courant", path, 1.0),
        _leaf_or(values, "tsmult", path, 1.2 if mode == "courant" else 1.0),
        _leaf_or(values, "ats_dtadj", path, 2.0),
        _leaf_or(values, "ats_dtfailadj", path, 5.0),
    )


def _parse_model(d: Mapping[str, Any]) -> ModelConfig:
    aq = _section(_require(d, "aquifer", "model"), "model.aquifer")
    layers_raw = _require(aq, "layers", "model.aquifer")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ConfigError("model.aquifer.layers: expected a non-empty list")
    layers: list[AquiferLayer] = []
    for i, layer in enumerate(layers_raw):
        path = f"model.aquifer.layers[{i}]"
        values = _section(layer, path)
        layers.append(AquiferLayer(*(
            _leaf(values, key, path) for key in (
                "top", "bottom", "hydraulic_conductivity", "porosity", "specific_storage",
                "initial_temperature", "longitudinal_dispersivity", "transverse_dispersivity",
                "thermal_conductivity_water", "thermal_conductivity_solid", "heat_capacity_water",
                "density_water", "heat_capacity_solid", "density_solid",
            )
        )))

    chd_raw = _section(_require(d, "chd_heads", "model"), "model.chd_heads")
    chd = ChdHeads(
        north=_leaf(chd_raw, "north", "model.chd_heads"),
        south=_leaf(chd_raw, "south", "model.chd_heads"),
        east=_leaf(chd_raw, "east", "model.chd_heads"),
        west=_leaf(chd_raw, "west", "model.chd_heads"),
    )
    chd.mode = str(chd_raw.get("mode", "sides"))
    if chd.mode not in ("sides", "linear"):
        raise ConfigError("model.chd_heads.mode: expected 'sides' or 'linear'")
    if chd.mode == "linear":
        chd.validate_linear()

    doublets_raw = _require(d, "doublets", "model")
    if not isinstance(doublets_raw, list) or not doublets_raw:
        raise ConfigError("model.doublets: expected a non-empty list")
    doublets: list[Doublet] = []
    for i, doublet in enumerate(doublets_raw):
        path = f"model.doublets[{i}]"
        values = _section(doublet, path)
        name = _require(values, "name", path)
        if not isinstance(name, str):
            raise ConfigError(f"{path}.name: expected a string")
        doublets.append(Doublet(
            name, _coords(values, "warm_well", path), _coords(values, "cold_well", path),
            _leaf(values, "flow_rate", path), _leaf(values, "warm_injection_temperature", path),
            _leaf(values, "cold_injection_temperature", path),
        ))

    ew = _section(_require(d, "extraction_well", "model"), "model.extraction_well")
    sim = _section(_require(d, "simulation", "model"), "model.simulation")
    return ModelConfig(
        AquiferConfig(layers), chd, doublets,
        ExtractionWell(_coords(ew, "coordinates", "model.extraction_well"), _leaf(ew, "flow_rate", "model.extraction_well")),
        SimulationConfig(
            _leaf(sim, "n_years", "model.simulation"),
            _leaf(sim, "season_length_days", "model.simulation"),
            _leaf(sim, "timesteps_per_season", "model.simulation"),
            _parse_time_stepping(sim),
        ),
        _scalar(d["recharge"], "model.recharge") if "recharge" in d else 0.0,
    )


def _parse_observation_points(d: Mapping[str, Any]) -> list[ObservationPoint]:
    raw = d.get("observation_points", [])
    if not isinstance(raw, list):
        raise ConfigError("analysis.observation_points: expected a list")
    points: list[ObservationPoint] = []
    seen: set[str] = set()
    for i, point in enumerate(raw):
        path = f"analysis.observation_points[{i}]"
        values = _section(point, path)
        name = _require(values, "name", path)
        if not isinstance(name, str):
            raise ConfigError(f"{path}.name: expected a string")
        if name in seen:
            raise ConfigError(f"{path}.name: duplicate observation point name {name!r}")
        seen.add(name)
        points.append(ObservationPoint(name, _leaf(values, "x", path), _leaf(values, "y", path)))
    return points


def _parse_analysis(d: Mapping[str, Any]) -> AnalysisConfig:
    variables = _require(d, "output_variables", "analysis")
    if not isinstance(variables, list) or not all(isinstance(x, str) for x in variables):
        raise ConfigError("analysis.output_variables: expected a list of strings")
    times = _require(d, "times_of_interest", "analysis")
    if not isinstance(times, list):
        raise ConfigError("analysis.times_of_interest: expected a list")
    return AnalysisConfig(
        list(variables),
        [_scalar(value, f"analysis.times_of_interest[{i}]") for i, value in enumerate(times)],
        _parse_observation_points(d),
    )


def load_config(path: str | Path) -> Config:
    """Load and validate a deterministic YAML configuration."""

    source = Path(path)
    try:
        text = source.read_text()
    except OSError as exc:
        raise ConfigError(f"could not read config file '{source}': {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"'{source}' is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    return Config(
        _parse_mesh(_section(_require(raw, "mesh", ""), "mesh")),
        _parse_model(_section(_require(raw, "model", ""), "model")),
        _parse_analysis(_section(_require(raw, "analysis", ""), "analysis")),
    )

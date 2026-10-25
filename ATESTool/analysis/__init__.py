"""Output parsing and plotting.

``analyse`` parses a run into a :class:`SimulationResult`, the single seam exposing
``.timeseries``, ``.snapshot(t)`` and ``.stats``. Example, after building and running a
model with :func:`~ATESTool.model.build_model` / :func:`~ATESTool.model.run_model`::

    result = analyse(model_paths, raw_output, config, mesh=mesh)
    result.timeseries.head()  # DataFrame indexed by time: temperature, head
    result.stats
    # {'peak_temperature': ..., 'max_drawdown': ..., 'breakthrough_time': ...,
    #  'mean_recovery_efficiency': ..., 'net_energy_imbalance': ...,
    #  'ambient_drift': ..., 'boundary_temperature_influence': ...,
    #  'boundary_head_influence': ...}

``run_from_config`` (in :mod:`ATESTool.run`) calls ``analyse`` for you and writes
``.timeseries``/``.stats`` to ``timeseries.csv``/``stats.json``.
"""

from ATESTool.analysis.results import SimulationResult, analyse

__all__ = ["analyse", "SimulationResult"]

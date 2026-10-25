"""Numerical integration conventions for MODFLOW end-of-step outputs."""

from __future__ import annotations

import numpy as np


def integrate_end_step(
    times: np.ndarray,
    values: np.ndarray,
    *,
    start: float = 0.0,
    end: float | None = None,
) -> float:
    """Integrate values reported at the end of each model time step.

    MODFLOW writes the temperature used by its implicit source/sink calculation at
    the *end* of a time step. The matching discrete integral is therefore a
    right-endpoint sum, ``sum(value_i * dt_i)``. Treating these samples as ordinary
    point observations and applying ``trapezoid`` drops the first interval when the
    initial state is not present in the output and smooths across stress changes.

    ``start``/``end`` may select one stress period. A sample exactly at ``start`` is
    the end of the preceding interval and is not counted in the selected period.
    """
    sample_times = np.asarray(times, dtype=float)
    sample_values = np.asarray(values, dtype=float)
    if sample_times.ndim != 1 or sample_values.ndim != 1 or sample_times.size != sample_values.size:
        raise ValueError("times and values must be one-dimensional arrays of equal length")
    if sample_times.size and np.any(np.diff(sample_times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    stop = float(sample_times[-1]) if end is None and sample_times.size else end
    if stop is None or float(stop) <= float(start):
        return 0.0
    selected = (sample_times > float(start)) & (sample_times <= float(stop))
    selected_times = sample_times[selected]
    if selected_times.size == 0:
        return 0.0
    interval_starts = np.concatenate(([float(start)], selected_times[:-1]))
    return float(np.sum(sample_values[selected] * (selected_times - interval_starts)))

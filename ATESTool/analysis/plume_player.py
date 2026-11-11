"""Self-contained HTML player for saved thermal plume snapshots."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex

from ATESTool.analysis.plots import _zoom_limits

if TYPE_CHECKING:
    from ATESTool.analysis.results import SimulationResult
    from ATESTool.config import Config
    from ATESTool.mesh import VertexGrid
    from ATESTool.model.schedule import WellSchedule


def plume_player(
    sim: SimulationResult, mesh: VertexGrid, schedule: WellSchedule, config: Config
) -> str:
    """Embed a bounded sample of temperature fields and draw the mesh in the browser."""
    times = np.asarray(sim.times, dtype=float)
    if len(times) < 2:
        return ""
    season = config.model.simulation.season_length_days
    n_boundaries = int(times[-1] / season)
    boundaries = np.linspace(1, n_boundaries, min(n_boundaries, 40)).round() * season
    targets = np.r_[np.linspace(times[0], times[-1], 40), boundaries]
    indices = np.unique([np.abs(times - target).argmin() for target in targets])

    (xmin, xmax), (ymin, ymax) = _zoom_limits(sim, mesh, schedule)
    vertices = mesh.vertices
    polygons = []
    cells = []
    scale = 600 / max(xmax - xmin, ymax - ymin)
    width, height = round((xmax - xmin) * scale), round((ymax - ymin) * scale)

    def point(x: float, y: float) -> tuple[float, float]:
        return round((x - xmin) * scale, 2), round((ymax - y) * scale, 2)

    for index, cell in enumerate(mesh.cell_vertices):
        xy = vertices[cell]
        if (xy[:, 0].max() < xmin or xy[:, 0].min() > xmax
                or xy[:, 1].max() < ymin or xy[:, 1].min() > ymax):
            continue
        polygons.append("M" + "L".join(f"{px},{py}" for px, py in map(lambda p: point(*p), xy)) + "Z")
        cells.append(index)
    if not cells:
        return ""
    delta = sim.temperature[np.ix_(indices, cells)] - sim.initial_temperature
    finite = np.isfinite(delta)
    limit = max(0.1, float(np.max(np.abs(delta[finite]), initial=0)))
    codes = np.where(finite, np.rint(127 * (np.where(finite, delta, 0) / limit + 1)), 255)
    palette = [to_hex(colormaps["RdBu_r"](i / 254)) for i in range(255)] + ["#999999"]
    wells = []
    extraction = config.model.extraction_well
    if extraction.flow_rate:
        wells.append((*point(*extraction.coordinates), "#222222"))
    for doublet in config.model.doublets:
        if doublet.flow_rate:
            wells.append((*point(*doublet.warm_well), "#b2182b"))
            wells.append((*point(*doublet.cold_well), "#2166ac"))
    data = json.dumps({
        "polygons": polygons, "palette": palette, "wells": wells,
        "times": [f"{time:,.1f}" for time in times[indices]],
        "frames": base64.b64encode(codes.astype(np.uint8).tobytes()).decode("ascii"),
    }, separators=(",", ":"))
    return f"""
<section id="plume-player">
  <h2>Thermal plume animation</h2>
  <button type="button">Play</button>
  <label>Saved time <input type="range" min="0" max="{len(indices) - 1}" value="0"></label>
  <output></output>
  <figure>
    <canvas width="{width}" height="{height}" style="max-width:100%;height:auto"
      aria-label="Well-centred temperature change map, north up"></canvas>
    <figcaption>North is up. Black: extraction. Red: warm well. Blue: cold well.</figcaption>
  </figure>
  <div style="width:min(22rem,100%);height:0.8rem;background:linear-gradient(
    to right,{','.join(palette[:255:32])},{palette[254]})"></div>
  <p>−{limit:.2g} °C to +{limit:.2g} °C relative to initial aquifer temperature.
    The colour scale stays fixed across frames.</p>
  <noscript>Enable JavaScript to play the animation.</noscript>
  <script type="application/json">{data}</script>
</section>
<script>
(() => {{
  const player = document.querySelector('#plume-player');
  const data = JSON.parse(player.querySelector('script').textContent);
  const canvas = player.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const paths = data.polygons.map(p => new Path2D(p));
  const values = atob(data.frames);
  const slider = player.querySelector('input');
  const button = player.querySelector('button');
  let timer = null;
  function draw() {{
    const frame = Number(slider.value);
    const offset = frame * paths.length;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    paths.forEach((path, i) => {{
      ctx.fillStyle = data.palette[values.charCodeAt(offset + i)];
      ctx.fill(path);
    }});
    for (const [x, y, colour] of data.wells) {{
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = colour;
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.stroke();
    }}
    player.querySelector('output').value = 'Day ' + data.times[frame];
  }}
  function pause() {{
    clearInterval(timer);
    timer = null;
    button.textContent = 'Play';
  }}
  slider.addEventListener('input', () => {{ pause(); draw(); }});
  button.addEventListener('click', () => {{
    if (timer !== null) {{ pause(); return; }}
    if (slider.value === slider.max) {{ slider.value = '0'; draw(); }}
    button.textContent = 'Pause';
    timer = setInterval(() => {{
      slider.value = Number(slider.value) + 1;
      draw();
      if (slider.value === slider.max) pause();
    }}, 160);
  }});
  draw();
}})();
</script>
"""

# Running examples

The examples reproduce the three application questions used for the SoftwareX
demonstration. They share one synthetic aquifer definition in `base.yaml` and write
their generated model files and CSV results below `examples/output/`.

From the repository root, after installing ATESTool, Triangle, and MODFLOW 6 as
described in `INSTALL.md`, run:

```bash
python examples/e1_material_interference.py
python examples/e2_siting_sensitivity.py
python examples/e3_accumulation.py
```

- **E1 — material interference:** compares a no-ATES baseline with one downstream
  doublet.
- **E2 — siting sensitivity:** compares near, far, transverse, and reversed
  longitudinal layouts while keeping the physical controls fixed.
- **E3 — accumulation:** runs two individual doublets and their combined layout, then
  reports the superposition residual.

Each example uses a common mesh for every case in its comparison. The complete examples
are intentionally substantial 20-year simulations; runtime depends on the mesh backend,
MODFLOW 6 build, and computer.

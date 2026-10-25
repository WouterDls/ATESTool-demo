# Installing ATESTool

ATESTool needs two external pieces of software that are **not** installed by
`pip install`: a **Triangle** mesh generator and a **MODFLOW 6** binary. Install
those first, then install the Python package.

---

## 1. Triangle mesh backend (optional but recommended)

The mesh generator uses [Triangle](https://www.cs.cmu.edu/~quake/triangle.html)
(J.R. Shewchuk). ATESTool depends on the **`triangle` PyPI package**, a Python wrapper
around Triangle's C code:

```bash
pip install triangle          # or:  pip install -e ".[mesh]"
```

If `triangle` is **not** installed, ATESTool falls back to
`scipy.spatial.Delaunay` and logs a warning, **per-well refinement rings are not
supported in that fallback**, so the mesh will differ.

```{warning}
**Licensing:** Triangle is free for private, research, and instructional use,
but **commercial use is restricted** (separate permission from the author is
required). Verify the licence terms before using ATESTool in a commercial
deliverable.
```

---

## 2. Install ATESTool

```bash
pip install -e .            # runtime
pip install -e ".[dev]"     # + pytest, pytest-cov, mypy (for development)
```

Requires Python ≥ 3.10.

---

## 3. MODFLOW 6 binary (required to *run* models)

Building model input files needs only FloPy, but **running** a simulation needs a
MODFLOW 6 executable on your `PATH`.

```{warning}
**Version & architecture matter.**
- The **GWE** (groundwater energy transport) model used by ATESTool requires
  **MODFLOW 6 ≥ 6.5.0**. Earlier builds (e.g. 6.4.4) do **not** include GWE.
- On **Apple Silicon (arm64)** you need an **arm64** build. An x86_64 build will
  fail with `bad CPU type in executable` unless Rosetta 2 is installed.
```

Get a suitable build with FloPy's downloader (it fetches the correct architecture):

```bash
pip install flopy
get-modflow :python   # or: get-modflow /path/to/bin
```

or download from the [USGS MODFLOW 6 releases](https://github.com/MODFLOW-USGS/modflow6/releases).

Tests that run the solver are marked `@pytest.mark.integration` and **skip cleanly
when no binary is found**, so the default `pytest` run stays green without it.
Run them explicitly with `pytest -m integration`.

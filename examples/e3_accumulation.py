"""E3: test whether interference from two doublets is approximately additive."""

import pandas as pd

from _common import ROOT, doublet, run_comparison
from ATESTool.config import load_config


def main() -> None:
    base = load_config(ROOT / "base.yaml")
    a = doublet(base, "a", (-175.0, 250.0), (-75.0, 250.0))
    b = doublet(base, "b", (175.0, 250.0), (75.0, 250.0))
    output = ROOT / "output" / "e3_accumulation"
    cases = {
        "no_ates": [doublet(base, "baseline", (0.0, 200.0), (0.0, 300.0), 0.0)],
        "doublet_a": [a],
        "doublet_b": [b],
        "doublets_a_and_b": [a, b],
    }
    summary = run_comparison(cases, output)

    temperatures = pd.read_csv(output / "extraction_temperature.csv", index_col=0)
    baseline = temperatures["no_ates"]
    residual = (
        temperatures["doublets_a_and_b"]
        - baseline
        - (temperatures["doublet_a"] - baseline)
        - (temperatures["doublet_b"] - baseline)
    )
    residual.rename("superposition_residual_c").to_csv(
        output / "superposition_residual.csv"
    )
    print(summary)
    print(f"maximum absolute superposition residual: {residual.abs().max():.6g} degC")


if __name__ == "__main__":
    main()


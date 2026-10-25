"""E1: quantify thermal interference from one downstream ATES doublet."""

from _common import ROOT, doublet, run_comparison
from ATESTool.config import load_config


def main() -> None:
    base = load_config(ROOT / "base.yaml")
    cases = {
        "no_ates": [doublet(base, "baseline", (0.0, 200.0), (0.0, 300.0), 0.0)],
        "downstream_doublet": [
            doublet(base, "downstream", (0.0, 200.0), (0.0, 300.0))
        ],
    }
    summary = run_comparison(cases, ROOT / "output" / "e1_material_interference")
    print(summary)


if __name__ == "__main__":
    main()


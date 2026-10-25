"""E2: compare distance and orientation while holding physical controls fixed."""

from _common import ROOT, doublet, run_comparison
from ATESTool.config import load_config


def main() -> None:
    base = load_config(ROOT / "base.yaml")
    cases = {
        "no_ates": [doublet(base, "baseline", (0.0, 200.0), (0.0, 300.0), 0.0)],
        "near_downstream": [doublet(base, "near", (0.0, 100.0), (0.0, 200.0))],
        "far_downstream": [doublet(base, "far", (0.0, 450.0), (0.0, 550.0))],
        "transverse": [doublet(base, "transverse", (-50.0, 250.0), (50.0, 250.0))],
        "warm_upstream": [doublet(base, "upstream", (0.0, 300.0), (0.0, 200.0))],
    }
    summary = run_comparison(cases, ROOT / "output" / "e2_siting_sensitivity")
    print(summary)


if __name__ == "__main__":
    main()


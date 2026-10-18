"""Stage 4: the area-law test.

Combines the inspiral posteriors (initial area A_i = A_1 + A_2) with the ringdown
posteriors (final area A_f) for all six inspiral-ringdown combinations, and
reports P(Delta A > 0) and the equivalent one-sided Gaussian significance.
P(Delta A > 0) is evaluated exactly over all pairs of samples.
Runs in seconds from the data shipped in data/.

Writes data/derived/area_law_significance.csv and data/derived/area_summary.csv.

The inspiral cuts and ringdown runs that are combined are set in utils/config.py
(INSPIRAL_CUTS_TM and RINGDOWN_REFERENCE_RUNS).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make utils/ importable
from utils import area, config


def plain(label):
    """Strip the LaTeX markup from a plot label, for the CSV tables."""
    return label.replace("$", "").replace("\\,", " ").replace("_", "").replace("{", "").replace("}", "")


def main():
    """Compute the areas and P(Delta A > 0) for every combination, then print and save the tables."""
    a_insp = {cut: area.inspiral_area(cut) for cut in config.INSPIRAL_CUTS_TM}
    a_ring = {(nm, st): area.ringdown_area(nm, st) for nm, st, _ in config.RINGDOWN_REFERENCE_RUNS}

    # one row per (inspiral cut, ringdown run) pair
    rows = []
    for cut, (nm, st, label) in area.combinations():
        a_i, a_f = a_insp[cut], a_ring[(nm, st)]
        p = area.prob_increase(a_i, a_f)
        d = area.fractional_change_draws(a_i, a_f)
        rows.append({
            "inspiral": f"t< = {cut} T_M", "ringdown": plain(label),
            "P(dA>0)": p, "sigma": area.sigma_from_prob(p),
            "dA_median": np.median(d), "dA_p05": np.percentile(d, 5), "dA_p95": np.percentile(d, 95),
        })
    table = pd.DataFrame(rows)

    # 5/50/95% of each area posterior
    summary = pd.DataFrame(
        [{"area": f"A_i, t< = {cut} T_M", **dict(zip(("p05", "median", "p95"), np.percentile(a, [5, 50, 95])))}
         for cut, a in a_insp.items()]
        + [{"area": f"A_f, {plain(label)}",
            **dict(zip(("p05", "median", "p95"), np.percentile(a_ring[(nm, st)], [5, 50, 95])))}
           for nm, st, label in config.RINGDOWN_REFERENCE_RUNS])

    config.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.DERIVED_DIR / "area_law_significance.csv", index=False, float_format="%.4f")
    summary.to_csv(config.DERIVED_DIR / "area_summary.csv", index=False, float_format="%.5g")
    with pd.option_context("display.width", 200):
        print("Horizon areas [Msun^2]:")
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print("\nArea-law test:")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()

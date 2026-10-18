"""Stage 2b: decode TDInf samples into physical parameters.

The raw samples (data/posteriors/raw/*_samples_raw.txt) are in TDInf's
unconstrained sampling coordinates: each bounded parameter is a logistic
transform of its prior range, so decoding needs the prior bounds of the
run that produced them (those in pipeline/inspiral/run_tdinf.sh):

    total mass            (58, 82) Msun
    mass ratio            (0.25, 1.0)
    inclination           (0, pi)       [sampled as cos(inclination) in (-1, 1)]
    luminosity distance   (200, 700) Mpc
    spin magnitudes       (0, 0.9)

The phase is sampled as a 2D vector (phase_x, phase_y), and each spin as a
magnitude plus an unnormalised direction vector (c_x, c_y, c_z).

Writes data/posteriors/{inspiral_40,inspiral_90,full_signal}_samples.dat.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make utils/ importable
from utils import config

RAW_DIR = config.POSTERIOR_DIR / "raw"

# column order of the raw files (TDInf's sampled keys)
RAW_HEADER = [
    "total_mass", "mass_ratio", "luminosity_distance", "cos_inclination",
    "spin1_magnitude", "spin2_magnitude", "phase_x", "phase_y",
    "c1_x", "c1_y", "c1_z", "c2_x", "c2_y", "c2_z",
]

# Prior bounds of the TDInf runs. If you change the priors in pipeline/inspiral/run_tdinf.sh,
# change them here too, or the decoded samples will be wrong.
PRIOR_BOUNDS = {
    "total_mass": (58.0, 82.0),
    "mass_ratio": (0.25, 1.0),
    "luminosity_distance": (200.0, 700.0),
    "spin1_magnitude": (0.0, 0.9),
    "spin2_magnitude": (0.0, 0.9),
}

RUNS = ("inspiral_40", "inspiral_90", "full_signal")  # decodes data/posteriors/raw/<run>_samples_raw.txt

EXPORT_COLUMNS = [
    "total_mass", "mass_ratio", "luminosity_distance", "inclination", "phase",
    "spin1_x", "spin1_y", "spin1_z", "spin2_x", "spin2_y", "spin2_z",
    "spin1_magnitude", "spin2_magnitude",
]


def inv_logit(y, xmin, xmax):
    """Map a logistic coordinate y back to the interval (xmin, xmax)."""
    ey = np.exp(np.clip(y, -700, 700))
    return (ey * xmax + xmin) / (1.0 + ey)


def normalize_spin_components(magnitude, c_x, c_y, c_z):
    """Scale the direction vector (c_x, c_y, c_z) to length magnitude; return the spin components."""
    norm = np.sqrt(c_x**2 + c_y**2 + c_z**2)
    chi = np.divide(magnitude, norm, out=np.zeros_like(magnitude, dtype=float), where=norm > 0)
    return c_x * chi, c_y * chi, c_z * chi


def decode(raw):
    """Convert a DataFrame of raw TDInf samples into physical parameters."""
    out = pd.DataFrame({
        "total_mass": inv_logit(raw["total_mass"], *PRIOR_BOUNDS["total_mass"]),
        "mass_ratio": inv_logit(raw["mass_ratio"], *PRIOR_BOUNDS["mass_ratio"]),
        "luminosity_distance": inv_logit(raw["luminosity_distance"], *PRIOR_BOUNDS["luminosity_distance"]),
        "inclination": np.arccos(np.clip(inv_logit(raw["cos_inclination"], -1.0, 1.0), -1.0, 1.0)),
        "phase": np.arctan2(raw["phase_y"], raw["phase_x"]),
        "spin1_magnitude": inv_logit(raw["spin1_magnitude"], *PRIOR_BOUNDS["spin1_magnitude"]),
        "spin2_magnitude": inv_logit(raw["spin2_magnitude"], *PRIOR_BOUNDS["spin2_magnitude"]),
    })
    out["spin1_x"], out["spin1_y"], out["spin1_z"] = normalize_spin_components(
        out["spin1_magnitude"].to_numpy(), raw["c1_x"].to_numpy(), raw["c1_y"].to_numpy(), raw["c1_z"].to_numpy())
    out["spin2_x"], out["spin2_y"], out["spin2_z"] = normalize_spin_components(
        out["spin2_magnitude"].to_numpy(), raw["c2_x"].to_numpy(), raw["c2_y"].to_numpy(), raw["c2_z"].to_numpy())
    return out


def main():
    """Decode every run in RUNS and write its physical samples to data/posteriors/."""
    for label in RUNS:
        path = RAW_DIR / f"{label}_samples_raw.txt"
        raw = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_HEADER)
        phys = decode(raw)
        out_path = config.POSTERIOR_DIR / f"{label}_samples.dat"
        phys[EXPORT_COLUMNS].to_csv(out_path, sep=" ", index=False, float_format="%.16e")
        print(f"{label}: wrote {out_path.name} ({len(phys)} samples, "
              f"median M = {phys.total_mass.median():.2f} Msun, median q = {phys.mass_ratio.median():.3f})")


if __name__ == "__main__":
    main()

"""Paths, constants and analysis choices shared by the pipeline and figure scripts.

The analysis choices at the bottom (inspiral cuts, ringdown start times and the
ringdown runs used in the area-law test) are read by the figure scripts and by
pipeline/area_law_test.py. They can be changed here, but every combination must
exist in data/: the inspiral cuts need matching files in data/posteriors/, and the
ringdown runs must be among the fits in data/ringdown/ringdown_posteriors.h5.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
INJECTION_DIR = DATA / "injection"
POSTERIOR_DIR = DATA / "posteriors"
RINGDOWN_DIR = DATA / "ringdown"
DERIVED_DIR = DATA / "derived"
FIGURE_DIR = REPO / "figures"
EXTERNAL_DIR = REPO / "external"

# Files that must be downloaded separately (see README / external/README.md)
GRCHOMBO_WAVEFORM = EXTERNAL_DIR / "GRChombo_BBSsol02_A17A17q100d17p000_Res40.h5"
SEOBNR_ROM = EXTERNAL_DIR / "SEOBNRv4ROM_v3.0.hdf5"
NRSUR_FILE = EXTERNAL_DIR / "NRSur7dq4_v1.0.h5"

# Injection products
METADATA_CSV = INJECTION_DIR / "injection_metadata.csv"
STRAIN = {ifo: INJECTION_DIR / f"{ifo}_injection_data.h5" for ifo in ("H1", "L1")}
CLEAN_STRAIN = {ifo: INJECTION_DIR / f"nonoise_{ifo}_injection_data.h5" for ifo in ("H1", "L1")}
PSD = {ifo: INJECTION_DIR / f"{ifo}_PSD.dat" for ifo in ("H1", "L1")}

# Posterior samples (physical parameters)
INSPIRAL_SAMPLES = {
    40: POSTERIOR_DIR / "inspiral_40_samples.dat",
    90: POSTERIOR_DIR / "inspiral_90_samples.dat",
}
FULL_SIGNAL_SAMPLES = POSTERIOR_DIR / "full_signal_samples.dat"
RINGDOWN_POSTERIORS = RINGDOWN_DIR / "ringdown_posteriors.h5"
FULL_SIGNAL_RECONSTRUCTION = DERIVED_DIR / "full_signal_reconstruction.npz"

# Source
TOTAL_MASS = 71.5            # Msun, total mass of the injected BS binary
LUMINOSITY_DISTANCE = 425.0  # Mpc

# Time units, following Abac et al. (2025) (GW250114 area-law paper)
T_M = 0.354e-3    # s, mass-rescaled time unit of the total initial mass
T_MF = 0.337e-3   # s, mass-rescaled time unit of the remnant mass

# Analysis segments (see the module docstring before changing these)
INSPIRAL_CUTS_TM = (40, 90)          # t_< = t_peak - (40 or 90) T_M
RINGDOWN_STARTS_TMF = (6.0, 10.5)    # t_> = t_peak + (6 or 10.5) T_Mf used in the area test
RINGDOWN_REFERENCE_RUNS = [          # (number of modes, t_> in T_Mf, label)
    (1, 10.5, r"$t_>$ = 10.5 $T_{M_f}$, 1 mode"),
    (1, 6.0, r"$t_>$ = 6 $T_{M_f}$, 1 mode"),
    (2, 6.0, r"$t_>$ = 6 $T_{M_f}$, 2 modes"),
]
F_HIGH = 270.0  # Hz, upper analysis frequency, stored in the injection metadata


def require(path, hint=""):
    """Return path if it exists, otherwise raise FileNotFoundError with an optional hint."""
    path = Path(path)
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\n  -> {hint}"
        raise FileNotFoundError(msg)
    return path

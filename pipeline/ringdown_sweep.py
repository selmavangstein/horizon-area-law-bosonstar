"""Stage 3: ringdown start-time sweep with the `ringdown` package.

Fits Kerr QNMs (220, or 220 + 221) to 0.2 s of H1/L1 data starting at
t_> = t_peak + start x T_Mf, for each start time below. NOT needed to make the
figures: the results are shipped compactly in data/ringdown/ringdown_posteriors.h5.
The sweep takes ~3 h on a laptop (33 fits); it is resumable.

Writes output/ringdown/fit_result_modes_<n>_start_<t>.nc (start 10.5 -> "105"), then:
  python pipeline/export_ringdown_posteriors.py output/ringdown

Settings you can change are the constants below: the start times of the sweep, the
length of data analysed, the number of chains and samples, and the priors. Fits whose
output file already exists are skipped, so delete output/ringdown/ to rerun them
with new settings.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"  # one thread per chain; must be set before numpy/jax load

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from jax import config as jax_config
jax_config.update("jax_enable_x64", False)

import numpyro
numpyro.set_host_device_count(4)  # run the NUM_CHAINS chains in parallel on 4 CPU devices
numpyro.set_platform("cpu")

import ringdown as rd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make utils/ importable
from utils import config, data

OUT_DIR = config.REPO / "output" / "ringdown"

# Start times t_> - t_peak in T_Mf: both models up to 15 T_Mf, the 220 model alone to 20 T_Mf
START_OFFSETS_BOTH = [3, 4, 5, 6, 7, 8, 9, 10, 10.5, 11, 12, 13, 14, 15]
START_OFFSETS_1MODE_ONLY = [16, 17, 18, 19, 20]

T_MF = config.T_MF   # mass-rescaled time unit of the remnant (Abac et al. 2025)
DURATION = 0.2       # s of data analysed after t_>
NUM_CHAINS = 4       # NumPyro chains per fit
NUM_SAMPLES = 2000   # posterior samples per chain

# Priors of the QNM model
A_SCALE_MAX = 5e-20              # scale of the amplitude prior
M_MIN, M_MAX = 50.0, 85.0        # remnant mass, detector frame [Msun]
CHI_MIN, CHI_MAX = 0.255, 0.95   # remnant spin


def load_inputs():
    """Noisy H1/L1 strain from data/injection/ (as dicts of times and strain) and the injection metadata."""
    h1, l1 = ({"times": t, "strain": s} for t, s in (data.strain("H1"), data.strain("L1")))
    md = pd.read_csv(config.METADATA_CSV)
    return h1, l1, md


def run_one(h1, l1, md, num_modes, start_offset, output_file):
    """Fit the 220 (num_modes=1) or 220+221 (num_modes=2) model from t_> = start_offset T_Mf.

    The result is written as netCDF to output_file, and its path is returned.
    """
    # modes as (prograde, spin weight, l, m, overtone n)
    if num_modes == 1:
        modes = [(1, -2, 2, 2, 0)]
    elif num_modes == 2:
        modes = [(1, -2, 2, 2, 0), (1, -2, 2, 2, 1)]
    else:
        raise ValueError("num_modes must be 1 or 2")

    fit = rd.Fit(modes=modes)

    ra = md["ra"].values[0]
    dec = md["dec"].values[0]
    psi = md["psi"].values[0]
    h1_arrival_time = md["h1_arrival_time"].values[0]
    f_min = md["f_min"].values[0]

    fit.add_data(pd.Series(h1["strain"], index=h1["times"]), ifo="H1")
    fit.add_data(pd.Series(l1["strain"], index=l1["times"]), ifo="L1")

    # t_> is measured from the peak time in H1; the sky position is fixed to the injected one
    fit.set_target(
        t0=h1_arrival_time + start_offset * T_MF,
        ra=ra, dec=dec, psi=psi,
        duration=DURATION,
        reference_ifo="H1",
    )

    fit.condition_data(ds=1, f_min=f_min)
    fit.compute_acfs(f_min=f_min)

    fit.update_model(a_scale_max=A_SCALE_MAX, m_min=M_MIN, m_max=M_MAX,
                     chi_min=CHI_MIN, chi_max=CHI_MAX)

    fit.run(num_chains=NUM_CHAINS, num_samples=NUM_SAMPLES)

    output = Path(output_file).with_suffix(".nc")
    output.parent.mkdir(parents=True, exist_ok=True)
    fit.result.to_netcdf(output)
    return output


def main():
    """Run every (num_modes, start time) fit in turn; return 1 if any fit failed."""
    parameter_sets = (
        [{"num_modes": nm, "analysis_start_time": so}
         for nm in (1, 2) for so in START_OFFSETS_BOTH]
        + [{"num_modes": 1, "analysis_start_time": so}
           for so in START_OFFSETS_1MODE_ONLY]
    )
    parameter_sets.sort(key=lambda p: (p["num_modes"], p["analysis_start_time"]))

    h1, l1, md = load_inputs()
    print(f"Input dir : {config.INJECTION_DIR}", flush=True)
    print(f"Output dir: {OUT_DIR}", flush=True)
    print(f"Total configurations: {len(parameter_sets)}", flush=True)

    failures = []
    for i, params in enumerate(parameter_sets, start=1):
        nm = params["num_modes"]
        so = params["analysis_start_time"]
        tag = str(so).replace(".", "")
        output_file = OUT_DIR / f"fit_result_modes_{nm}_start_{tag}.nc"

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"[{i}/{len(parameter_sets)}] skip (exists): {output_file.name}", flush=True)
            continue

        print(f"[{i}/{len(parameter_sets)}] modes={nm} start={so} t_Mf -> {output_file.name}", flush=True)
        t_start = time.time()
        try:
            run_one(h1, l1, md, nm, so, output_file)
            print(f"    done in {time.time() - t_start:.0f}s", flush=True)
        except Exception:
            failures.append((nm, so))
            print(f"    FAILED after {time.time() - t_start:.0f}s", flush=True)
            traceback.print_exc()

    print(f"\nSweep finished. failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

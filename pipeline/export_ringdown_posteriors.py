"""Compress the ringdown start-time sweep into data/ringdown/ringdown_posteriors.h5.

The full `ringdown` result files (one netCDF per fit, ~1.6 GB in total, mostly
reconstructed waveforms) are archived on Zenodo. This keeps only the scalar
posterior variables listed in KEEP (remnant mass and spin; mode amplitudes,
frequencies and damping rates), flattened over chains, which is all the figures need.

Usage:
  python pipeline/export_ringdown_posteriors.py <directory of fit_result_modes_*_start_*.nc>
"""
import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from ringdown.result import Result

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make utils/ importable
from utils import config
from utils.data import ringdown_key

# remnant mass/spin, and per-mode amplitude, frequency and damping rate
KEEP = ("m", "chi", "a", "f", "g")
FILE_RE = re.compile(r"fit_result_modes_(\d+)_start_(\d+)\.nc$")


def parse_name(name):
    """Parse (num_modes, start in T_Mf) from a result file name, or None if it does not match.

    The start time is written without the decimal point, e.g.
    fit_result_modes_1_start_105.nc -> (1, 10.5).
    """
    match = FILE_RE.search(name)
    if not match:
        return None
    start = int(match.group(2))
    return int(match.group(1)), start / 10.0 if start >= 100 else float(start)


def main():
    """Read every fit in fit_dir and write one HDF5 group per fit to --out."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fit_dir", type=Path)
    p.add_argument("--out", type=Path, default=config.RINGDOWN_POSTERIORS)
    args = p.parse_args()

    files = sorted((parse_name(f.name), f) for f in args.fit_dir.glob("*.nc") if parse_name(f.name))
    if not files:
        raise FileNotFoundError(f"no fit_result_modes_*_start_*.nc files in {args.fit_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as out:
        out.attrs["description"] = "Scalar ringdown posteriors (chains flattened) for the boson-star injection"
        out.attrs["t_mf_seconds"] = config.T_MF
        for (num_modes, start), path in files:
            post = Result.from_netcdf(path).posterior
            g = out.create_group(ringdown_key(num_modes, start))
            g.attrs["num_modes"] = num_modes
            g.attrs["start_tmf"] = start
            g.attrs["source_file"] = path.name
            g.attrs["modes"] = [m.decode() if isinstance(m, bytes) else str(m) for m in post["mode"].values]
            for name in KEEP:
                var = post[name]
                arr = var.values
                # flatten (chain, draw[, mode]) into (sample[, mode])
                arr = arr.reshape(-1, arr.shape[-1]) if "mode" in var.dims else arr.reshape(-1)
                g.create_dataset(name, data=arr.astype(np.float32), compression="gzip")
            print(f"{path.name}: {g.attrs['num_modes']} mode(s), t> = {start:g} T_Mf, "
                  f"{g['m'].size} samples")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

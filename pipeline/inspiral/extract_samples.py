"""Thin an emcee backend from run_tdinf.sh into posterior samples (sampling coordinates).

The production chain (after main.py's unsaved 2e4-step burn-in) is thinned by 1e4,
giving 58 x 64 = 3712 samples for a 6e5-step run. The output columns are TDInf's
unconstrained sampling coordinates; convert them with pipeline/decode_tdinf_samples.py.

Usage:
  python pipeline/inspiral/extract_samples.py <run>_results.h5 <out>_samples_raw.txt [--thin 10000]
"""
import argparse

import emcee
import numpy as np


def main():
    """Read the emcee backend, thin the chain and save the flattened samples as text."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("backend", help="emcee HDF5 backend written by run_tdinf.sh")
    p.add_argument("out", help="output text file, one sample per row")
    p.add_argument("--thin", type=int, default=10000, help="keep every n-th step of each walker")
    p.add_argument("--discard", type=int, default=0,
                   help="steps to drop from the start of the saved chain (the burn-in is already not saved)")
    args = p.parse_args()

    backend = emcee.backends.HDFBackend(args.backend, read_only=True)
    samples = backend.get_chain(discard=args.discard, thin=args.thin, flat=True)
    np.savetxt(args.out, samples)
    print(f"{backend.iteration} steps x {backend.shape[0]} walkers -> {len(samples)} samples in {args.out}")


if __name__ == "__main__":
    main()

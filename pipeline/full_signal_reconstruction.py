"""Waveform reconstruction for the full-signal (inspiral-merger-ringdown) analysis.

Evaluates NRSur7dq4, projected onto H1 and L1, for posterior draws from the
full-signal TDInf run. The data are conditioned exactly as in the inference, and
the script saves the posterior median and the 5, 25, 75 and 95% percentiles alongside the conditioned
zero-noise injection. The output is data/derived/full_signal_reconstruction.npz,
which makefig5.py reads.

Requires tdinf and NRSur7dq4_v1.0.h5 (see README). Takes a few minutes for 1000 draws.

Usage:
  python pipeline/full_signal_reconstruction.py [--n-draws 1000] [--seed 12345] [--out path.npz]

--n-draws sets how many posterior samples are turned into waveforms (more draws give
smoother percentile bands, at a cost of roughly 0.3 s each). The run settings below
(T_CUT_TM, PRIOR_ARGS and the options in build_args) must match those of the
full_signal run in pipeline/inspiral/run_tdinf.sh.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))                            # make utils/ importable
sys.path.insert(0, str(REPO / "pipeline" / "inspiral"))  # and analysis_utils
from utils import config

# NRSur7dq4 must be on LAL_DATA_PATH before tdinf imports LALSimulation
config.require(config.NRSUR_FILE, "download NRSur7dq4_v1.0.h5 into external/ (see README)")
os.environ["LAL_DATA_PATH"] = ":".join(filter(None, [str(config.EXTERNAL_DIR), os.environ.get("LAL_DATA_PATH")]))

import tdinf.utils as utils
from analysis_utils import create_run_sampler_arg_parser, get_conditioned_time_and_data, initialize_kwargs

IFOS = ("H1", "L1")
# Settings of the full-signal TDInf run: a cut beyond the end of the signal
T_CUT_TM = 115.0
PRIOR_ARGS = [
    "--total-mass-prior-bounds", "58", "82",
    "--mass-ratio-prior-bounds", "0.25", "1.0",
    "--inclination-prior-bounds", "0", str(np.pi),
    "--luminosity-distance-prior-bounds", "200", "700",
]
# posterior columns passed to the waveform model
SAMPLE_COLUMNS = ["total_mass", "mass_ratio", "luminosity_distance", "inclination", "phase",
                  "spin1_x", "spin1_y", "spin1_z", "spin2_x", "spin2_y", "spin2_z"]


def build_args(strain_paths):
    """Command-line arguments of the full-signal TDInf run, parsed with main.py's parser."""
    return create_run_sampler_arg_parser().parse_args([
        "--output-h5", "unused.h5", "--mode", "pre", "--Tcut-tm", str(T_CUT_TM),
        "--ifos", *IFOS, "--prefix", "full_signal_reconstruction",
        "--metadata", str(config.METADATA_CSV),
        "--h1-data", str(strain_paths["H1"]), "--l1-data", str(strain_paths["L1"]),
        "--h1-psd", str(config.PSD["H1"]), "--l1-psd", str(config.PSD["L1"]),
        "--data-dir", str(config.DATA),
        *PRIOR_ARGS,
        "--approx", "NRSur7dq4", "--sampling-rate", "2048", "--data-sampling-rate", "4096.0",
        "--max-padding-fraction", "0.2",
    ])


def reference_parameters(args):
    """Injection metadata in TDInf's naming, filling in f_low, f_ref and f22_start on args as main.py does."""
    meta = pd.read_csv(args.metadata).iloc[0].to_dict()
    meta["right_ascension"] = meta["ra"]
    meta["declination"] = meta["dec"]
    meta["polarization"] = meta["psi"]
    for key in ["spin1_x", "spin1_y", "spin1_z", "spin2_x", "spin2_y", "spin2_z"]:
        meta.setdefault(key, 0.0)
    if args.flow is None:
        args.flow = meta["f_min"]
    if args.fref is None or args.fref <= 0:
        args.fref = meta["f_ref"] if meta["f_ref"] > 0 else args.flow
    if args.f22_start is None:
        args.f22_start = meta["f22_start"]
    return meta


def conditioned(args, meta, strain_paths):
    """Condition and crop the strain in strain_paths exactly as in the inference.

    Returns the waveform manager and the per-detector time and data dictionaries.
    """
    kwargs = initialize_kwargs(args, meta)
    manager = utils.WaveformManager(list(IFOS), vary_time=False, vary_skypos=False, **kwargs)
    time_dict, data_dict, _ = get_conditioned_time_and_data(
        args, wf_manager=manager, reference_parameters=meta,
        data_paths={ifo: str(p) for ifo, p in strain_paths.items()},
        psd_paths={ifo: str(p) for ifo, p in config.PSD.items()},
        verbose=False,
    )
    return manager, time_dict, data_dict


def projected_waveform(manager, time_dict, meta, row):
    """NRSur7dq4 strain in each detector for one posterior sample, at the injected sky position and time."""
    x = {k: float(row[k]) for k in SAMPLE_COLUMNS}
    x.update(right_ascension=float(meta["right_ascension"]), declination=float(meta["declination"]),
             polarization=float(meta["polarization"]), geocenter_time=float(meta["geocenter_time"]))
    wf = manager.get_projected_waveform(x, list(IFOS), time_dict, f22_start=float(meta["f22_start"]),
                                        f_ref=float(meta["f_ref"]), window=True)
    return {ifo: np.asarray(wf[ifo], dtype=float) for ifo in IFOS}


def main():
    """Draw posterior samples, evaluate their waveforms and save the percentile bands."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", type=Path, default=config.FULL_SIGNAL_RECONSTRUCTION)
    args_cli = p.parse_args()

    samples = pd.read_csv(config.FULL_SIGNAL_SAMPLES, sep=r"\s+")
    rng = np.random.default_rng(args_cli.seed)
    n = min(args_cli.n_draws, len(samples))
    draws = samples.iloc[rng.choice(len(samples), size=n, replace=False)].reset_index(drop=True)

    # condition the noisy and the zero-noise data in the same way
    args = build_args(config.STRAIN)
    meta = reference_parameters(args)
    manager, time_dict, noisy = conditioned(args, meta, config.STRAIN)
    _, _, clean = conditioned(args, meta, config.CLEAN_STRAIN)

    # waveform of every draw, then percentiles across draws at each time
    stacks = {ifo: np.empty((n, len(time_dict[ifo]))) for ifo in IFOS}
    for i, (_, row) in enumerate(draws.iterrows()):
        wf = projected_waveform(manager, time_dict, meta, row)
        for ifo in IFOS:
            stacks[ifo][i] = wf[ifo]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} waveforms", flush=True)

    out = {"ifos": np.asarray(IFOS), "n_draws": np.asarray(n)}
    for ifo in IFOS:
        q05, q25, q50, q75, q95 = np.percentile(stacks[ifo], [5, 25, 50, 75, 95], axis=0)
        out[f"{ifo}_t_rel"] = np.asarray(time_dict[ifo]) - float(meta[f"{ifo.lower()}_arrival_time"])
        out[f"{ifo}_data"] = np.asarray(noisy[ifo], dtype=float)
        out[f"{ifo}_clean"] = np.asarray(clean[ifo], dtype=float)
        for name, q in (("q05", q05), ("q25", q25), ("q50", q50), ("q75", q75), ("q95", q95)):
            out[f"{ifo}_{name}"] = q
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args_cli.out, **out)
    print(f"wrote {args_cli.out}")


if __name__ == "__main__":
    main()

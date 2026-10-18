"""Loaders for the data products shipped in data/."""
import h5py
import numpy as np
import pandas as pd

from . import config


def metadata():
    """Injection parameters and times (data/injection/injection_metadata.csv) as a pandas Series."""
    return pd.read_csv(config.METADATA_CSV).iloc[0]


def strain(ifo, clean=False):
    """Detector strain (GPS times, strain) for the noisy or zero-noise injection."""
    path = (config.CLEAN_STRAIN if clean else config.STRAIN)[ifo]
    with h5py.File(path, "r") as f:
        return f["times"][:], f["strain"][:]


def psd(ifo):
    """One-sided PSD as (frequencies, PSD)."""
    arr = np.loadtxt(config.PSD[ifo])
    return arr[:, 0], arr[:, 1]


def posterior(path):
    """Physical posterior samples from a .dat file, with component and chirp masses added.

    The samples hold the detector-frame total mass M and mass ratio q = m_2/m_1 <= 1,
    so m_1 = M/(1+q) and m_2 = qM.
    """
    s = pd.read_csv(path, sep=r"\s+")
    s["mass_1"] = s["total_mass"] / (1.0 + s["mass_ratio"])
    s["mass_2"] = s["mass_ratio"] * s["mass_1"]
    s["chirp_mass"] = (s["mass_1"] * s["mass_2"]) ** 0.6 / s["total_mass"] ** 0.2
    return s


def inspiral_posterior(cut_tm):
    """Posterior of the inspiral analysis with t_< = t_peak - cut_tm T_M."""
    return posterior(config.INSPIRAL_SAMPLES[cut_tm])


def full_signal_posterior():
    """Posterior of the full-signal (inspiral-merger-ringdown) analysis."""
    return posterior(config.FULL_SIGNAL_SAMPLES)


def ringdown_runs():
    """List of (num_modes, start_tmf) available in the ringdown posterior file."""
    with h5py.File(config.RINGDOWN_POSTERIORS, "r") as f:
        return sorted((f[k].attrs["num_modes"], f[k].attrs["start_tmf"]) for k in f)


def ringdown_posterior(num_modes, start_tmf):
    """Dict of flattened posterior arrays (m, chi, a, f, g, ...) for one ringdown fit."""
    with h5py.File(config.RINGDOWN_POSTERIORS, "r") as f:
        g = f[ringdown_key(num_modes, start_tmf)]
        return {k: g[k][:] for k in g}


def ringdown_key(num_modes, start_tmf):
    """Group name of a ringdown fit in ringdown_posteriors.h5, e.g. (1, 10.5) -> 'modes_1_start_10.5'."""
    return f"modes_{int(num_modes)}_start_{float(start_tmf):g}"

"""Source-frame waveforms and whitening used by the figures."""
import os

import h5py
import numpy as np

from . import config


def _set_lal_data_path():
    """Prepend external/ to LAL_DATA_PATH so LALSimulation finds the waveform data files."""
    paths = [str(config.EXTERNAL_DIR), *filter(None, os.environ.get("LAL_DATA_PATH", "").split(":"))]
    os.environ["LAL_DATA_PATH"] = ":".join(dict.fromkeys(paths))


def _peak_aligned(hp, hc, delta_t):
    """Return (t, hp, hc) with t = 0 at the peak of the amplitude sqrt(hp^2 + hc^2)."""
    amp = np.hypot(hp, hc)
    t = (np.arange(hp.size) - np.argmax(amp)) * delta_t
    return t, hp, hc


def boson_star_hphc(inclination, distance_mpc=config.LUMINOSITY_DISTANCE,
                    total_mass=config.TOTAL_MASS, delta_t=1 / 4096.0):
    """h_+, h_x of the GRChombo A17 binary (all l=2 modes), with t=0 at peak amplitude.

    Generated exactly as in pipeline/injection.py, before tapering. Needs the
    GRChombo file and SEOBNRv4ROM_v3.0.hdf5 in external/.
    """
    config.require(config.GRCHOMBO_WAVEFORM, "see external/README.md for how to obtain the NR waveform")
    config.require(config.SEOBNR_ROM, "LALSimulation's NR_hdf5 approximant needs SEOBNRv4ROM; see README")
    _set_lal_data_path()
    import lal
    import lalsimulation as lalsim

    path = str(config.GRCHOMBO_WAVEFORM)
    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertNumRelData(params, path)
    modes = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateAllModesAtL(modes, 2)
    lalsim.SimInspiralWaveformParamsInsertModeArray(params, modes)

    # mass ratio and starting frequency are stored in the NR file
    with h5py.File(path, "r") as f:
        m1, m2 = f.attrs["mass1"], f.attrs["mass2"]
        f_lower = f.attrs["f_lower_at_1MSUN"] / total_mass
    spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(0.0, total_mass, path)
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        m1 / (m1 + m2) * total_mass * lal.MSUN_SI, m2 / (m1 + m2) * total_mass * lal.MSUN_SI,
        *spins, distance_mpc * 1e6 * lal.PC_SI, inclination, 0.0, 0.0, 0.0, 0.0,
        delta_t, f_lower, f_lower, params, lalsim.NR_hdf5)
    return _peak_aligned(hp.data.data.copy(), hc.data.data.copy(), delta_t)


def nrsur_hphc(inclination, total_mass=config.TOTAL_MASS, mass_ratio=1.0,
               distance_mpc=config.LUMINOSITY_DISTANCE, f_low=20.0, delta_t=1 / 4096.0):
    """h_+, h_x of a non-spinning binary BH from NRSur7dq4, with t=0 at peak amplitude."""
    config.require(config.NRSUR_FILE, "download NRSur7dq4_v1.0.h5 into external/ (see README)")
    _set_lal_data_path()
    import lal
    import lalsimulation as lalsim

    m1 = total_mass / (1 + mass_ratio)
    params = lal.CreateDict()
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        m1 * lal.MSUN_SI, mass_ratio * m1 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0,
        distance_mpc * 1e6 * lal.PC_SI, inclination, 0.0, 0.0, 0.0, 0.0,
        delta_t, f_low, f_low, params, lalsim.SimInspiralGetApproximantFromString("NRSur7dq4"))
    return _peak_aligned(hp.data.data.copy(), hc.data.data.copy(), delta_t)


def whiten(strain, psd_freqs, psd, sample_rate, f_band=(None, None), alpha=0.1):
    """Whiten a time series by the (one-sided) PSD, normalised to unit noise variance.

    The series is Tukey-windowed, divided by sqrt(PSD * fs / 2) in the frequency
    domain and band-limited to f_band, so stationary Gaussian noise with this PSD
    comes out with (approximately) unit variance per sample.
    """
    from scipy.signal.windows import tukey

    n = strain.size
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    asd = np.sqrt(np.interp(freqs, psd_freqs, psd) * sample_rate / 2.0)
    spec = np.fft.rfft(strain * tukey(n, alpha)) / asd
    lo, hi = f_band
    mask = np.ones_like(freqs, dtype=bool)
    if lo is not None:
        mask &= freqs >= lo
    if hi is not None:
        mask &= freqs <= hi
    spec[~mask] = 0.0
    return np.fft.irfft(spec, n)

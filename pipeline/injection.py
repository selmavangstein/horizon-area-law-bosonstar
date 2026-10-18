"""Stage 1: inject the GRChombo boson-star waveform into simulated H1/L1 noise.

Steps:
  1. generate h_+, h_x from the NR file (all l=2 modes) with LALSimulation's NR_hdf5
     approximant, scaled to M = 71.5 Msun;
  2. taper the start (cosine window that ends safely below the instantaneous GW
     frequency, which sets f_low = 23.4 Hz) and the last cycles of the ringdown;
  3. zero-pad by 4x the signal length on each side and roll the peak to t = 0;
  4. project onto H1/L1 with bilby at the GW250114-like extrinsic parameters and
     add Gaussian noise from the aLIGO_ZERO_DET_high_P PSD (plus a zero-noise copy).

Outputs (default: output/injection/): {H1,L1}_injection_data.h5,
nonoise_{H1,L1}_injection_data.h5, injection_metadata.csv, {H1,L1}_PSD.dat.
The files shipped in data/injection/ were produced by this procedure; rerunning it
reproduces the zero-noise strain exactly but draws a different noise realisation.

Needs external/GRChombo_..._Res40.h5 and external/SEOBNRv4ROM_v3.0.hdf5 (LALSimulation
calls SEOBNRv4ROM internally when generating NR_hdf5 waveforms).

Usage:
  python pipeline/injection.py [--outdir output/injection]

The source parameters, tapering choices and noise seed are set as constants below.
Note that total mass and distance come from utils/config.py, which the figures and
the analyses also read.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import root_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make utils/ importable
from utils import config

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Injection parameters (Table 2 of the supplement of Abac et al. 2025, except D_L)
LUMINOSITY_DISTANCE = config.LUMINOSITY_DISTANCE  # Mpc, chosen for network SNR ~ 80
TOTAL_MASS = config.TOTAL_MASS                    # Msun
INCLINATION = 0.78                # rad
PSI = 1.329                       # polarisation angle, rad
RA = 2.333                        # rad
DEC = 0.190                       # rad
GEOCENT_TIME = 1420878141.235932  # GPS time of the geocentre peak
PHI_REF = 0.0                     # reference phase passed to LALSimulation
SAMPLING_FREQUENCY = 4096.0       # Hz
F_HIGH = config.F_HIGH

# Tapering, padding and noise settings
BEGINNING_TAPER_SAFETY = 2   # taper frequency 1/(t1-t0) is (1+safety) below f_GW(t1)
END_TAPER_PEAKS = 10         # end taper starts this many h_+ peaks before the end
PAD_FACTOR = 4               # zero-padding on each side, in units of the signal length
SEED = 88170235              # seed of the noise realisation


def generate_hphc(inclination, f_lower, params, masses_si, spins, distance_si, delta_t):
    """h_+, h_x (LAL REAL8TimeSeries) of the NR waveform in params, via the NR_hdf5 approximant."""
    import lalsimulation as lalsim
    return lalsim.SimInspiralChooseTDWaveform(
        *masses_si, *spins, distance_si, inclination, PHI_REF, 0.0, 0.0, 0.0,
        delta_t, f_lower, f_lower, params, lalsim.NR_hdf5)


def cosine_window(times, t0, t1, direction="on"):
    """Half-cosine taper rising from 0 to 1 over [t0, t1] ("on"), or falling from 1 to 0 ("off")."""
    taper = np.ones_like(times)
    mask = (times >= t0) & (times <= t1)
    x = (times[mask] - t0) / (t1 - t0)
    taper[mask] = 0.5 * (1 - np.cos(np.pi * x))
    if direction == "off":
        taper[mask] = 1 - taper[mask]
        taper[times < t0] = 1.0
        taper[times > t1] = 0.0
    else:
        taper[times < t0] = 0.0
        taper[times > t1] = 1.0
    return taper


def find_peaks(x):
    """Indices of the local maxima of x."""
    d = np.diff(x)
    return np.where((d[:-1] > 0) & (d[1:] < 0))[0] + 1


def main():
    """Generate, taper, pad and inject the waveform, then write the strain, PSDs and metadata."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=config.REPO / "output" / "injection")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    config.require(config.GRCHOMBO_WAVEFORM, "see external/README.md")
    config.require(config.SEOBNR_ROM, "see external/README.md")
    os.environ["LAL_DATA_PATH"] = ":".join(filter(None, [str(config.EXTERNAL_DIR), os.environ.get("LAL_DATA_PATH")]))
    import bilby
    import lal
    import lalsimulation as lalsim

    nr_file = str(config.GRCHOMBO_WAVEFORM)
    with h5py.File(nr_file, "r") as f:
        m1_code, m2_code = f.attrs["mass1"], f.attrs["mass2"]
        f_lower = f.attrs["f_lower_at_1MSUN"] / TOTAL_MASS  # start of the NR waveform
    mass_1 = m1_code * TOTAL_MASS / (m1_code + m2_code)
    mass_2 = m2_code * TOTAL_MASS / (m1_code + m2_code)
    masses_si = (mass_1 * lal.MSUN_SI, mass_2 * lal.MSUN_SI)
    spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(0.0, TOTAL_MASS, nr_file)
    distance_si = LUMINOSITY_DISTANCE * 1e6 * lal.PC_SI
    delta_t = 1.0 / SAMPLING_FREQUENCY

    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertNumRelData(params, nr_file)
    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateAllModesAtL(mode_array, 2)
    lalsim.SimInspiralWaveformParamsInsertModeArray(params, mode_array)

    # start taper, computed from the face-on waveform: it ends at the time t1 where
    # 1/t1 is (1 + safety) times below the instantaneous GW frequency
    hp, hc = generate_hphc(0.0, f_lower, params, masses_si, spins, distance_si, delta_t)
    times = np.arange(hp.data.length) * delta_t
    phase = UnivariateSpline(times, np.unwrap(np.arctan2(hc.data.data, hp.data.data)), k=3, s=0)
    omega = phase.derivative()
    t1 = root_scalar(lambda t: (1 + BEGINNING_TAPER_SAFETY) / t - omega(t) / (2 * np.pi),
                     x0=0.1, bracket=[0.01, 0.3]).root
    f_min = float(omega(t1) / (2 * np.pi))  # lowest trustworthy frequency after tapering
    print(f"start taper ends at t = {t1:.3f} s, f_min = {f_min:.2f} Hz")

    # waveform at the injected inclination
    hp, hc = generate_hphc(INCLINATION, f_lower, params, masses_si, spins, distance_si, delta_t)
    times = np.arange(hp.data.length) * delta_t
    taper = cosine_window(times, 0.0, t1, "on")
    t_end_start = times[find_peaks(hp.data.data)[-END_TAPER_PEAKS]]
    taper *= cosine_window(times, t_end_start, times[-1], "off")
    hp_t, hc_t = hp.data.data * taper, hc.data.data * taper

    # zero-pad and roll the amplitude peak to t = 0
    n_pad = int(len(times) * PAD_FACTOR)
    hp_pad = np.pad(hp_t, n_pad)
    hc_pad = np.pad(hc_t, n_pad)
    times_pad = np.arange(hp_pad.size) * delta_t
    peak_id = np.argmax(np.hypot(hp_pad, hc_pad))
    peak = times_pad[peak_id]
    phase_merger = np.arctan2(-hc_pad[peak_id], hp_pad[peak_id]) + np.pi
    hp_roll, hc_roll = np.roll(hp_pad, -peak_id), np.roll(hc_pad, -peak_id)
    duration = hp_pad.size / SAMPLING_FREQUENCY
    start_time = GEOCENT_TIME - peak

    def nr_source(time):
        """Time-domain source model for bilby: the padded, peak-centred NR polarisations."""
        return {"plus": np.interp(time, times_pad, hp_roll, left=0.0, right=0.0),
                "cross": np.interp(time, times_pad, hc_roll, left=0.0, right=0.0)}

    # NB: the paper's noise realisation came from bilby's unseeded generator and is not
    # reproducible; the zero-noise signal is reproduced exactly. Seed bilby so reruns repeat.
    np.random.seed(SEED)
    bilby.core.utils.random.seed(SEED)
    injection_parameters = dict(
        mass_1=mass_1, mass_2=mass_2, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
        luminosity_distance=LUMINOSITY_DISTANCE, theta_jn=INCLINATION, psi=PSI, phase=phase_merger,
        geocent_time=GEOCENT_TIME, ra=RA, dec=DEC)
    waveform = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=SAMPLING_FREQUENCY, time_domain_source_model=nr_source,
        start_time=0)

    # project onto the detectors, once without noise and once with Gaussian noise from the PSD
    ifos_clean = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos_clean.set_strain_data_from_zero_noise(sampling_frequency=SAMPLING_FREQUENCY, duration=duration,
                                               start_time=start_time)
    ifos_clean.inject_signal(waveform_generator=waveform, parameters=injection_parameters, raise_error=False)
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=SAMPLING_FREQUENCY,
                                                       duration=duration, start_time=start_time)
    ifos.inject_signal(waveform_generator=waveform, parameters=injection_parameters, raise_error=False)

    # metadata read by the analyses and figures
    meta = {
        "total_mass": TOTAL_MASS, "mass_ratio": mass_1 / mass_2, "luminosity_distance": LUMINOSITY_DISTANCE,
        "ra": RA, "dec": DEC, "psi": PSI, "duration": duration, "f_min": f_min, "f_max": F_HIGH,
        "f_ref": f_lower, "f22_start": f_lower, "inclination": INCLINATION, "phase": phase_merger,
        "geocenter_time": GEOCENT_TIME,
    }
    for ifo, ifo_clean in zip(ifos, ifos_clean):
        name = ifo.name
        meta[f"{name.lower()}_arrival_time"] = GEOCENT_TIME + ifo.time_delay_from_geocenter(RA, DEC, GEOCENT_TIME)
        meta[f"{name.lower()}_peak_time"] = ifo_clean.time_array[np.argmax(np.abs(ifo_clean.time_domain_strain))]
        for prefix, det in (("", ifo), ("nonoise_", ifo_clean)):
            with h5py.File(args.outdir / f"{prefix}{name}_injection_data.h5", "w") as f:
                f.create_dataset("times", data=det.time_array)
                f.create_dataset("strain", data=det.time_domain_strain)
        psd = ifo.power_spectral_density_array.copy()
        psd[np.isinf(psd)] = 0.25  # replace the infinite values outside the PSD's band
        np.savetxt(args.outdir / f"{name}_PSD.dat", np.column_stack((ifo.frequency_array, psd)))
    meta.update(sampling_frequency=SAMPLING_FREQUENCY, start_time=ifos[0].time_array[0],
                pad_time=n_pad * delta_t)
    pd.DataFrame([meta]).to_csv(args.outdir / "injection_metadata.csv", index=False)

    snrs = [ifo.meta_data["optimal_SNR"] for ifo in ifos]
    print("optimal SNR: " + ", ".join(f"{ifo.name} {s:.2f}" for ifo, s in zip(ifos, snrs))
          + f", network {np.sqrt(np.sum(np.square(snrs))):.2f}")
    print(f"wrote injection products to {args.outdir}")


if __name__ == "__main__":
    main()

# Data

Everything needed to make the figures is in this directory (about 16 MB).
All of it is for the boson-star injection described in the paper.

## `injection/`
Output of `pipeline/injection.py`.

| File | Contents |
|---|---|
| `{H1,L1}_injection_data.h5` | detector strain with Gaussian noise; datasets `times` (GPS s) and `strain` |
| `nonoise_{H1,L1}_injection_data.h5` | the same signal with zero noise |
| `{H1,L1}_PSD.dat` | one-sided PSD (`aLIGO_ZERO_DET_high_P`): columns are frequency [Hz] and PSD [1/Hz] |
| `injection_metadata.csv` | injection parameters, arrival and peak times, f_min (23.4 Hz) and SNR inputs |

## `posteriors/`
TDInf + emcee posterior samples, NRSur7dq4, 3712 samples each.

| File | Analysis |
|---|---|
| `inspiral_40_samples.dat` | inspiral, data truncated at t_< = t_peak − 40 T_M |
| `inspiral_90_samples.dat` | inspiral, data truncated at t_< = t_peak − 90 T_M |
| `full_signal_samples.dat` | full inspiral–merger–ringdown signal |

The files are whitespace-separated with a header row. The columns are:
- `total_mass` [M☉] and `mass_ratio` (m2/m1 ≤ 1)
- `luminosity_distance` [Mpc], `inclination` [rad] and `phase` [rad]
- dimensionless spin components `spin{1,2}_{x,y,z}`
- spin magnitudes `spin{1,2}_magnitude`

`raw/*_samples_raw.txt` contain the same samples in TDInf's unconstrained sampling
coordinates, as written by the sampler. `pipeline/decode_tdinf_samples.py`
converts them into the files above.

## `ringdown/ringdown_posteriors.h5`
Scalar posteriors from the `ringdown` start-time sweep. There are 33 fits:
- the 220 model at t_> = 3–20 T_Mf
- the 220+221 model at t_> = 3–15 T_Mf

Each fit has 4 chains × 2000 samples, flattened. Each group is named
`modes_<n>_start_<t>` and contains:
- `m` (remnant mass, M☉) and `chi` (remnant spin)
- `a` (mode amplitudes), `f` (frequencies, Hz) and `g` (damping rates, 1/s), each with shape (samples, modes)

Produced by `pipeline/export_ringdown_posteriors.py`.

## `derived/`
- `full_signal_reconstruction.npz`: waveform reconstruction for Fig. S1, made by `pipeline/full_signal_reconstruction.py`
- `area_law_significance.csv` and `area_summary.csv`: area-law test results, made by `pipeline/area_law_test.py`

## Not included (Zenodo)
The full sampler outputs are archived on Zenodo at **[DOI: 10.5281/zenodo.XXXXXXX — to be added]**:
- the emcee backends of the TDInf runs, including full chains for convergence checks
- the complete `ringdown` netCDF result files (~1.6 GB), which include reconstructed waveforms

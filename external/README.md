# External files

These files are not redistributed here. Put them in this directory (`external/`).
The scripts add it to `LAL_DATA_PATH` automatically.

| File | Needed by | Where to get it |
|---|---|---|
| `SEOBNRv4ROM_v3.0.hdf5` (333 MB) | `pipeline/injection.py`, `makefig1.py` (boson-star waveform) | [LALSuite waveform data, Zenodo DOI 10.5281/zenodo.14999310](https://zenodo.org/records/14999310) |
| `NRSur7dq4_v1.0.h5` (17.5 MB) | `pipeline/inspiral/run_tdinf.sh`, `pipeline/full_signal_reconstruction.py`, `makefig1.py` (black-hole waveform) | [LALSuite waveform data, Zenodo DOI 10.5281/zenodo.14999310](https://zenodo.org/records/14999310) |
| `GRChombo_BBSsol02_A17A17q100d17p000_Res40.h5` | `pipeline/injection.py`, `makefig1.py` | The GRChombo boson-star simulation catalogue of Evstafyeva et al., [PRL 133, 131401 (2024)](https://doi.org/10.1103/PhysRevLett.133.131401). Contact the authors of that paper. |

```bash
cd external
wget -O SEOBNRv4ROM_v3.0.hdf5 "https://zenodo.org/records/14999310/files/SEOBNRv4ROM_v3.0.hdf5?download=1"
wget -O NRSur7dq4_v1.0.h5    "https://zenodo.org/records/14999310/files/NRSur7dq4_v1.0.h5?download=1"
```

Why SEOBNRv4ROM? The injection uses LALSimulation's `NR_hdf5` approximant. When
generating the waveform, this approximant internally calls SEOBNRv4ROM to estimate
the waveform duration, and it fails if the ROM file is not on `LAL_DATA_PATH`.
The SEOB model is not used anywhere else in the analysis.

These file names apply to LALSuite >= 7.25. Older LALSuite versions instead look
for the `lalsuite-extra` names (e.g. `NRSur7dq4.h5`).

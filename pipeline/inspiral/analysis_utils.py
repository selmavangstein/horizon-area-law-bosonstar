"""Setup for the TDInf analyses: command-line options, data conditioning and walker initialisation.

Used by main.py and by pipeline/full_signal_reconstruction.py, which conditions the
data in exactly the same way.
"""
import argparse

import emcee
import numpy as np
import tdinf.utils as utils

# T_M in seconds, the time unit of --Tcut-tm (same value as utils/config.py: T_M)
TM_UNIT_SECONDS = 0.000354


def create_run_sampler_arg_parser():
    """Command-line options of main.py.

    The values used in the paper are set in run_tdinf.sh; the defaults here are
    generic and are not the paper's settings.
    """
    p = argparse.ArgumentParser(description="TDInf + emcee analysis of injected data (see main.py).")

    # Path to where to save data
    p.add_argument('-o', '--output-h5',  required=True, 
                   help='h5 filename for emcee backend')

    # Which data to analyse: before the cutoff ("pre"), after it ("post"), or both ("full")
    p.add_argument('-m', '--mode', required=True,
                   help='Options: "pre", "post", or "full"')

    # Cutoff time relative to the geocentre peak, in seconds or in units of T_M
    p.add_argument('-ts', '--Tcut-seconds', type=float, default=None,
                  help='Cutoff time in seconds from peak.')
    p.add_argument('--Tcut-tm', type=float, default=None,
                  help='Cutoff time in T_M units (T_M = 0.354 ms), e.g. -40 for t_peak - 40 T_M. '
                       'Converted to seconds internally.')
    # Optional start time to control analysis window
    p.add_argument('--Tstart', type=float, default=None,
                  help='Analysis start time offset from start_time in metadata (seconds). '\
                       'If not provided, will be calculated from metadata (pad_time).')
    p.add_argument('--metadata', type=str, default=None,
                  help='Path to injection metadata CSV file (contains all injection/extrinsic parameters).')
    p.add_argument('--data-dir', type=str, default='./data/',
                  help='Directory containing IFO data and PSD files. '
                       'Expects H1_injections.h5, L1_injections.h5, H1_PSD.dat, L1_PSD.dat by default.')
    p.add_argument('--h1-data', type=str, default=None,
                  help='Path to H1 data file (HDF5)')
    p.add_argument('--l1-data', type=str, default=None,
                  help='Path to L1 data file (HDF5)')
    p.add_argument('--h1-psd', type=str, default=None,
                  help='Path to H1 PSD file.')
    p.add_argument('--l1-psd', type=str, default=None,
                  help='Path to L1 PSD file.')
    # Optional args for waveform/data settings
    p.add_argument('--approx', default='NRSur7dq4', 
                  help='Waveform approximate string. Default: NRSur7dq4')
    p.add_argument('--no-spins', action='store_true',
                  help='Disable spin sampling by using a spinless parameterization. Keeps waveform approximant but fixes spins to zero.')
    p.add_argument('--sampling-rate', type=int, default=2048,
                   help='Sampling rate [Hz] the data are downsampled to for the analysis.')
    p.add_argument('--flow', type=float, default=None,
                   help="Lower frequency bound for data conditioning and likelihood function (ACF). ")
    p.add_argument('--fmax', type=float, default=None,
                   help="Upper frequency bound for data conditioning and likelihood function (ACF). "
                        "Default: None")
    p.add_argument('--f22-start', type=float, default=None,
                   help="frequency at which to start generating 22 mode for waveforms.")
    p.add_argument('--fref', type=float, default=None, 
                   help='Reference frequency.')

    # Optional args sampler settings
    p.add_argument('--nwalkers', type=int, default=512,
                  help='Number of walkers for emcee to run with. Default: 512')
    p.add_argument('--nsteps', type=int, default=50000,
                   help='Number of steps for emcee to run. Default: 50000')
    p.add_argument('--ncpu', type=int, default=4,
                   help='Number of CPUs for emcee to run on. Default: 4')
    p.add_argument('--prefix', type=str, default='tdinf_run',
                  help='Prefix for output files.')

    # Do we want to run with only the prior?
    p.add_argument('--only-prior', action='store_true',
                   help = 'Include flag if you want to run with only the prior.')

    # Do we want to sample in time and/or sky position?
    p.add_argument('--vary-time', action='store_true',
                   help = 'Include flag if you want to sample over time parameter.')
    p.add_argument('--vary-skypos', action='store_true',
                   help = 'Include flag if you want to sample over sky position parameters '
                           '(right ascension, declination, and polarization).')

    # Set up prior bounds
    p.add_argument('--total-mass-prior-bounds', type=float, nargs=2, default=[100, 400],
                   help="detector frame total mass bounds (in solar masses) for total_mass prior."
                        "Default: [100, 400]")
    p.add_argument('--mass-ratio-prior-bounds', type=float, nargs=2, default=[0.25, 1],
                   help="mass ratio bounds for mass ratio prior."
                        "Default: [0.25, 1]")
    p.add_argument('--luminosity-distance-prior-bounds', type=float, nargs=2, default=[100, 10000],
                   help="luminosity_distance bounds for luminosity_distance prior."
                        "Default: [100, 10000]")
    p.add_argument('--spin-magnitude-prior-bounds', type=float, nargs=2, default=[-0.99, 0.99],
                   help="Spin magnitude bounds for spin magnitude prior."
                        "Default: [-0.99,0.99]")
    p.add_argument('--inclination-prior-bounds', type=float, nargs=2, default=[0, np.pi],
                     help="Inclination bounds for inclination prior (radians). Default: [0, pi]")
    p.add_argument('--time-prior-sigma', type=float, default=0.01, 
                   help="Standard deviation of time prior [s.]"
                        "Default: 0.01 s")
    p.add_argument('--ifos', nargs='+', default=['H1', 'L1'],
                  help='List of detector IFOs (e.g., H1 L1). Default: [H1, L1]')
    p.add_argument('--data-sampling-rate', type=float, default=4096.0,
                   help="Sampling rate of input data (default: 4096.0 Hz)")
    p.add_argument('--max-padding-fraction', type=float, default=0.5,
                   help="Maximum fraction of analyzed signal that can be padding (default: 0.5 = 50%%). "
                        "If padding would exceed this fraction, Tstart is adjusted to cut it off.")

    # Debugging tools, run before sampling (see main.py)
    p.add_argument('--find-likelihood-slice', action='store_true',
                   help="Save a 1D log-likelihood slice in inclination as a PNG.")
    p.add_argument('--create-hphc', action='store_true',
                   help="Save h_+ and h_x at the reference parameters as CSV.")
    p.add_argument('--output-waveform', action='store_true',
                   help="Save the projected reference waveform in each detector as CSV.")
    p.add_argument('--one-d-optimizer', action='store_true',
                   help="Maximise the likelihood over inclination, with the other parameters at their reference values.")
    p.add_argument('--full-d-optimizer', action='store_true',
                   help="Run a local MAP and maximum-likelihood search in all sampled parameters.")
    p.add_argument('--run-emcee', action='store_true',
                   help="Run the emcee sampler. Without this flag, main.py stops after the debugging tools.")

    p.add_argument('--verbose', action='store_true',
                  help = 'Include flag if you want code to print out additional information as it '
                         'runs. Helpful for debugging.')

    # Reference parameters (used by --walkerinit reference and the debugging tools)
    p.add_argument('--override-total-mass', type=float, default=None,
                   help='Override total_mass from metadata (solar masses)')
    p.add_argument('--override-mass-ratio', type=float, default=None,
                   help='Override mass_ratio from metadata (dimensionless, q = m2/m1 <= 1)')
    p.add_argument('--override-distance', type=float, default=None,
                   help='Override luminosity_distance from metadata (Mpc)')
    p.add_argument('--override-inclination', type=float, default=None,
                   help='Override inclination from metadata (radians)')
    p.add_argument('--override-phase', type=float, default=None,
                   help='Override phase from metadata (radians)')

    # Walker initialisation (see the docstring of main.py)
    p.add_argument('--walkerinit', choices=['reference', 'posterior', 'flat', 'resume'], default='reference',
                   help='Walker init mode: reference (around injected parameters), '
                        'posterior (from a prior run), flat (from prior support), '
                        'or resume (continue the run in --output-h5).')
    p.add_argument('--init-from-backend', type=str, default=None,
                   help='Path to previous emcee HDFBackend (.h5) when --walkerinit=posterior.')

    return p


def _logit_from_uniform(u, lo, hi):
    """Map values in (lo, hi) to logistic coordinates via the inverse-logit map."""
    u = np.asarray(u, dtype=float)
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"Invalid prior limits for logistic transform: {lo}, {hi}")

    eps = 1e-12
    u = np.clip(u, lo + eps, hi - eps)
    return np.log((u - lo) / (hi - u))


def initialize_kwargs(args, reference_parameters):
    """Initialize kwargs dictionary for waveform manager and likelihood."""
    # Check that a cutoff time is given
    assert args.Tcut_seconds is not None or args.Tcut_tm is not None, "must give a cutoff time"

    # Check that the given mode is allowed
    run_mode = args.mode
    assert run_mode in ['full', 'pre', 'post'], f"mode must be 'full', 'pre', or 'post'. given mode = '{run_mode}'."

    # put all arguments into a dict
    kwargs = {
        'mtot_lim': args.total_mass_prior_bounds,
        'q_lim': args.mass_ratio_prior_bounds,
        'chi_lim': args.spin_magnitude_prior_bounds,
        'dist_lim': args.luminosity_distance_prior_bounds,
        'incl_lim': args.inclination_prior_bounds,
        'sigma_time': args.time_prior_sigma,
        'approx': args.approx,
        'force_no_spins': args.no_spins,
        'f_ref': args.fref,
        'f_low': args.flow,
        'f22_start': args.f22_start,
        'delta_t': 1 / args.sampling_rate,
        'right_ascension': reference_parameters['ra'],
        'declination': reference_parameters['dec'],
        'polarization': reference_parameters['psi'],
        'geocenter_time': reference_parameters['geocenter_time'],
        'verbose': args.verbose,
    }
    return kwargs


def _load_psds(psd_path_dict, ifos):
    """Load each detector's PSD file as an array of (frequency, PSD) rows."""
    pe_psds = {}
    for ifo in ifos:
        pe_psds[ifo] = np.genfromtxt(psd_path_dict[ifo], dtype=float)
    return pe_psds


def _regularize_time_arrays(raw_time_dict, ifos, expected_fs, verbose=False):
    """Rebuild the time arrays as exactly uniform grids if their sampling rate deviates from expected_fs.

    Guards against rounding errors in the stored GPS times.
    """
    fs_loaded = 1.0 / (raw_time_dict["H1"][1] - raw_time_dict["H1"][0])
    if abs(fs_loaded - expected_fs) <= 0.01:
        return raw_time_dict

    print(f"WARNING: loaded sampling frequency {fs_loaded:.10g} Hz deviates from expected {expected_fs} Hz; "
          "rebuilding uniform time arrays.")
    for ifo in ifos:
        time_loaded = raw_time_dict[ifo]
        raw_time_dict[ifo] = np.linspace(time_loaded[0], time_loaded[-1], len(time_loaded))
    return raw_time_dict


def _compute_effective_tstart(args, reference_parameters, time_dict, ifos, verbose=False):
    """GPS start time of the analysed data.

    With --Tstart, this is the start of the conditioned data plus Tstart. Otherwise
    the zero padding before the signal (pad_time in the metadata) is skipped, except
    for enough padding to make up --max-padding-fraction of the analysed data.
    """
    earliest_conditioned = min(time_dict[ifo][0] for ifo in ifos)

    if args.Tstart is not None:
        return earliest_conditioned + args.Tstart

    pad_time = reference_parameters.get("pad_time", 0)
    latest_time = max(time_dict[ifo][-1] for ifo in ifos)
    total_available_duration = latest_time - earliest_conditioned
    actual_signal_duration = total_available_duration - 2 * pad_time

    if args.max_padding_fraction < 1.0:
        max_padding_included = args.max_padding_fraction * actual_signal_duration / (1 - args.max_padding_fraction)
    else:
        max_padding_included = pad_time

    padding_included = min(max_padding_included, pad_time)
    padding_to_skip = pad_time - padding_included
    tstart_effective = earliest_conditioned + padding_to_skip

    if verbose:
        print(f"Including {padding_included:.4f} s of the {pad_time:.4f} s of zero padding before the signal")

    return tstart_effective


def _get_analysis_window(time_array, idx_cutoff, tstart_effective, run_mode, ifo=None, verbose=False):
    """Start and end indices of the analysed data, and the number of samples before and after the cutoff."""
    idx_start = np.searchsorted(time_array, tstart_effective, side="left")
    if idx_start >= idx_cutoff:
        print(f"WARNING: {ifo or ''} Tstart is after the cutoff (idx_start={idx_start} >= idx_cutoff={idx_cutoff}); "
              "starting one sample before the cutoff.")
        idx_start = max(0, idx_cutoff - 1)

    idx_end = len(time_array) - 1

    if run_mode == "full":
        npre = idx_cutoff - idx_start
        npost = idx_end - idx_cutoff
    elif run_mode == "pre":
        npre = idx_cutoff - idx_start
        npost = 1
    elif run_mode == "post":
        idx_start = idx_cutoff
        npre = 0
        npost = idx_end - idx_cutoff
    else:
        raise NotImplementedError(f"Run mode {run_mode} is not defined, please use one of pre post or full")

    return idx_start, idx_end, max(0, npre), max(0, npost)


def get_conditioned_time_and_data(args, wf_manager, reference_parameters, data_paths, psd_paths, initial_run_dir='', verbose=False):
    """Load, condition and crop the data according to the analysis mode and cutoff time.

    The data are downsampled to --sampling-rate and high-passed at 3/4 of f_low, then
    cut at the cutoff time, which is projected from the geocentre to each detector.

    Returns:
        time_dict: Dictionary of time arrays per IFO
        data_dict: Dictionary of strain data per IFO
        pe_psds: Dictionary of PSDs per IFO
    """
    # Check that a cutoff time is given
    assert args.Tcut_seconds is not None or args.Tcut_tm is not None, "must give a cutoff time"

    # Check that the given mode is allowed
    run_mode = args.mode
    assert run_mode in ['full', 'pre', 'post'], f"mode must be 'full', 'pre', or 'post'. given mode = '{run_mode}'."

    ifos = args.ifos
    data_path_dict = {ifo: data_paths[ifo] for ifo in ifos}
    psd_path_dict = {ifo: psd_paths[ifo] for ifo in ifos}

    # geocentre peak time of the injection
    tpeak_geocent = reference_parameters['geocenter_time']

    pe_psds = _load_psds(psd_path_dict, ifos)
    raw_time_dict, raw_data_dict = utils.load_raw_data(ifos=ifos, path_dict=data_path_dict, verbose=verbose)

    # Convert Tcut from user input (seconds or tM) to seconds
    tcut_seconds_input = args.Tcut_seconds
    if tcut_seconds_input is None and args.Tcut_tm is not None:
        tcut_seconds_input = args.Tcut_tm * TM_UNIT_SECONDS

    tcut_geocent = tpeak_geocent + tcut_seconds_input
    if verbose:
        print(f"Cutoff time: {tcut_geocent} ({tcut_seconds_input:+.6f} s from the peak)")

    # cutoff time in each detector
    tcut_dict, _ = utils.get_tgps_and_ap_dicts(tcut_geocent, ifos,
                                               reference_parameters['ra'],
                                               reference_parameters['dec'],
                                               reference_parameters['psi'])

    raw_time_dict = _regularize_time_arrays(
        raw_time_dict,
        ifos,
        expected_fs=args.data_sampling_rate,
        verbose=verbose,
    )

    time_dict, data_dict, icut_dict = utils.condition(raw_time_dict, raw_data_dict, tcut_dict,
                                                      args.sampling_rate, f_min=args.flow * 3 / 4,
                                                      f_max=args.fmax, verbose=verbose)

    dt = time_dict[ifos[0]][1] - time_dict[ifos[0]][0]
    tstart_effective = _compute_effective_tstart(
        args,
        reference_parameters,
        time_dict,
        ifos,
        verbose=verbose,
    )

    # analysis window in each detector; all detectors use the same number of samples
    start_idx_dict = {}
    npre_dict = {}
    npost_dict = {}
    for ifo, idx_cutoff in icut_dict.items():
        idx_start, idx_end, npre, npost = _get_analysis_window(
            time_dict[ifo],
            idx_cutoff,
            tstart_effective,
            run_mode,
            ifo=ifo,
            verbose=verbose,
        )
        start_idx_dict[ifo] = idx_start
        npre_dict[ifo] = npre
        npost_dict[ifo] = npost

    npre = min(npre_dict.values())
    npost = min(npost_dict.values())
    nanalyze = npre + npost
    tanalyze = nanalyze * dt
    if verbose:
        print('Will analyze {:.3f} s of data at {:.1f} Hz (Npre={}, Npost={})\n'.format(tanalyze, 1 / dt, npre, npost))
    assert tanalyze > 0, "Analysis window is empty. Check your cutoff time and analysis mode settings."

    for ifo, idx_cutoff in icut_dict.items():
        idx_start = start_idx_dict[ifo]
        time_dict[ifo] = time_dict[ifo][idx_start:idx_cutoff + npost]
        data_dict[ifo] = data_dict[ifo][idx_start:idx_cutoff + npost]

    return time_dict, data_dict, pe_psds

def init_walkers_from_backend_gaussian(
    likelihood_manager,
    backend_path: str,
    nwalkers: int,
    discard_frac: float = 0.5,
    thin: int = 1,
    inflate: float = 1.25,
    seed: int = 1234,
):
    """Draw initial walkers from a Gaussian fitted to the samples of an earlier run.

    The first discard_frac of the earlier chain is discarded and the rest thinned by
    `thin`. The Gaussian's covariance is widened by `inflate`, and draws outside the
    prior bounds or with a non-finite posterior are redrawn. The earlier run must
    use the same sampled parameters. Returns p0 with shape (nwalkers, ndim).
    """
    print("Initializing walkers based on posterior samples from backend:", backend_path)
    be = emcee.backends.HDFBackend(backend_path, read_only=True)
    discard = int(np.ceil(be.get_chain().shape[0] * discard_frac))
    chain = be.get_chain(discard=discard, thin=thin, flat=True)
    logp  = be.get_log_prob(discard=discard, thin=thin, flat=True)

    m = np.isfinite(logp)
    X_all = np.asarray(chain[m], float)
    lp_all = np.asarray(logp[m], float)
    print(
        f"Init sample source: discard={discard}, thin={thin}, "
        f"finite logp kept {int(m.sum())}/{len(logp)} samples"
    )

    if X_all.ndim != 2 or X_all.shape[0] == 0:
        raise ValueError("No finite posterior samples available after discard/thin.")

    mu  = X_all.mean(axis=0)
    cov = np.cov(X_all, rowvar=False)
    cov = (inflate ** 2) * cov + 1e-12 * np.eye(cov.shape[0])  # mild regularization

    rng = np.random.default_rng(seed)
    p0 = rng.multivariate_normal(mu, cov, size=nwalkers)

    def _inside_prior_bounds(x):
        """True if every bounded parameter of sample x lies inside its prior range."""
        try:
            x_phys = likelihood_manager.samp_to_phys(x)
        except Exception:
            return False

        for param in likelihood_manager.log_prior.logistic_parameters:
            try:
                val = x_phys[param.physical_name]
                # TrigLogisticParameter limits are defined on sin/cos-transformed values.
                if hasattr(param, "trig_function"):
                    val = param.trig_function(val)
            except Exception:
                return False

            lo, hi = param.limit
            if (not np.isfinite(val)) or (val < lo) or (val > hi):
                return False

        # Optional runtime override for inclination bounds (in radians).
        custom_inc = getattr(likelihood_manager, "custom_inclination_prior_bounds", None)
        if custom_inc is not None:
            inc = x_phys.get("inclination", np.nan)
            lo, hi = custom_inc
            if (not np.isfinite(inc)) or (inc < lo) or (inc > hi):
                return False

        return True

    def _safe_log_posterior(x):
        """Log posterior of x, or -inf if it cannot be evaluated."""
        try:
            return likelihood_manager.get_log_posterior(x)
        except Exception:
            return -np.inf

    def _is_valid_walker(x):
        """True if x is inside the prior bounds and has a finite posterior."""
        return _inside_prior_bounds(x) and np.isfinite(_safe_log_posterior(x))

    # Single rejection loop: each walker must satisfy both prior bounds and posterior validity.
    valid = np.array([_is_valid_walker(x) for x in p0], dtype=bool)
    bad = ~valid
    while np.any(bad):
        nb = int(bad.sum())
        p0[bad] = rng.multivariate_normal(mu, cov, size=nb)
        valid[bad] = np.array([_is_valid_walker(x) for x in p0[bad]], dtype=bool)
        bad = ~valid

    return p0


def init_walkers_from_flat_prior(
    likelihood_manager,
    nwalkers,
    seed: int = None,
    verbose: bool = False,
    boundary_fraction: float = 0.02,
):
    """Draw initial walkers from the prior.

    Bounded parameters are drawn uniformly within their prior range, shrunk by
    boundary_fraction of its width at each end so no walker starts on a bound; the
    phase and spin-direction coordinates are drawn from their standard-normal priors.
    """
    boundary_fraction = float(boundary_fraction)
    if not (0.0 <= boundary_fraction < 0.5):
        raise ValueError("boundary_fraction must be in [0, 0.5).")

    rng = np.random.default_rng(seed)
    log_prior = likelihood_manager.log_prior
    sampled_keys = list(log_prior.sampled_keys)
    idx_map = {key: i for i, key in enumerate(sampled_keys)}
    p0 = np.empty((nwalkers, len(sampled_keys)), dtype=float)

    def _shrink_bounds(lo, hi):
        """Move both bounds inwards by boundary_fraction of the width."""
        width = float(hi) - float(lo)
        pad = abs(boundary_fraction) * width
        lo2 = float(lo) + pad
        hi2 = float(hi) - pad
        if not (np.isfinite(lo2) and np.isfinite(hi2) and lo2 < hi2):
            raise ValueError(f"Invalid boundary margin for bounds [{lo}, {hi}] with fraction {boundary_fraction}")
        return lo2, hi2

    # Draw logistic parameters from the prior-support values in physical coordinates
    # and map into logistic coordinates using the model transform.
    for param in log_prior.logistic_parameters:
        idx = idx_map[param.logistic_name]

        if hasattr(param, "trig_name") and param.trig_name == "cos_inclination":
            inc_bounds = getattr(likelihood_manager, "custom_inclination_prior_bounds", None)
            if inc_bounds is not None:
                lo_i, hi_i = float(inc_bounds[0]), float(inc_bounds[1])
                if not (np.isfinite(lo_i) and np.isfinite(hi_i) and lo_i < hi_i):
                    raise ValueError(f"Invalid custom inclination bounds: {inc_bounds}")
                lo_u, hi_u = np.cos(hi_i), np.cos(lo_i)
            else:
                lo_u, hi_u = param.limit
        else:
            lo_u, hi_u = param.limit

        lo_u, hi_u = _shrink_bounds(lo_u, hi_u)

        u = rng.uniform(lo_u, hi_u, size=nwalkers)
        p0[:, idx] = _logit_from_uniform(u, lo_u, hi_u)

    # Cartesian angles are represented by x/y coordinates with standard-normal prior.
    for angle_param in log_prior.cartesian_angles:
        idx_x = idx_map[angle_param.x_name]
        idx_y = idx_map[angle_param.y_name]
        p0[:, idx_x] = rng.normal(size=nwalkers)
        p0[:, idx_y] = rng.normal(size=nwalkers)

    # Spin Cartesian components are also standard normal (or a subset for aligned models).
    for key in ['c1_x', 'c1_y', 'c1_z', 'c2_x', 'c2_y', 'c2_z']:
        if key in idx_map:
            p0[:, idx_map[key]] = rng.normal(size=nwalkers)

    # Optional reference-time sampling.
    if log_prior.vary_time:
        idx_t = idx_map['geocenter_time']
        p0[:, idx_t] = rng.normal(loc=log_prior.reference_time, scale=log_prior.sigma_time, size=nwalkers)

    if verbose:
        print("Initializing walkers by sampling directly from prior support.")

    return p0


def init_walkers_from_reference_values(likelihood_manager, nwalkers,
                               reference_parameters, verbose=True,
                               jitter_frac_of_spread=0.10,  # 10% of current spread
                               jitter_floor=1e-3,
                               jitter_cap=0.05,
                               seed=1234):
    """Initial walkers in a small ball around the reference (injected) parameters.

    TDInf places the walkers around the reference values; each sampling coordinate
    then gets Gaussian jitter of jitter_frac_of_spread times the walkers' robust
    spread, clipped to [jitter_floor, jitter_cap].
    """
    rng = np.random.default_rng(seed)

    print("Initializing walkers based on reference parameters.")
    p0 = likelihood_manager.log_prior.initialize_walkers(
        nwalkers, reference_parameters, reference_posterior=None, verbose=verbose
    )
    p0 = np.array(p0, dtype=float)

    # Robust scale per dimension from current p0 (MAD -> sigma)
    for i in range(p0.shape[1]):
        x = p0[:, i]
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        robust_sigma = 1.4826 * mad  # ~sigma if normal
        spread = max(robust_sigma, np.std(x), jitter_floor)

        scale = jitter_frac_of_spread * spread
        scale = min(max(scale, jitter_floor), jitter_cap)

        p0[:, i] = x + scale * rng.standard_normal(size=p0.shape[0])

    if verbose:
        print(f"Added jitter: {jitter_frac_of_spread*100:.1f}% of robust spread "
              f"(floor={jitter_floor}, cap={jitter_cap})")

    return p0


def enforce_inclination_prior_bounds(likelihood_manager, inclination_bounds, verbose=False):
    """Restrict the inclination prior to inclination_bounds [rad] without changing tdinf itself.

    Wraps `log_prior.get_lnprior` so that points with inclination outside the bounds
    have zero prior probability (log prior -inf).
    """
    if inclination_bounds is None:
        return

    lo, hi = float(inclination_bounds[0]), float(inclination_bounds[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"Invalid inclination bounds: {inclination_bounds}")

    log_prior = likelihood_manager.log_prior
    log_prior_cls = log_prior.__class__

    # Patch class once with a pickle-safe, module-level wrapper.
    if not getattr(log_prior_cls, "_inclination_runtime_patch_applied", False):
        log_prior_cls._orig_get_lnprior_func = log_prior_cls.get_lnprior
        log_prior_cls.get_lnprior = _inclination_bounded_get_lnprior
        log_prior_cls._inclination_runtime_patch_applied = True

    # Apply/refresh bounds for this specific likelihood instance.
    log_prior._inclination_bounds = (lo, hi)
    likelihood_manager.custom_inclination_prior_bounds = (lo, hi)

    if verbose:
        print(f"Enabled runtime inclination prior bounds: [{lo}, {hi}] rad")


def _inclination_bounded_get_lnprior(self, x, phys_dict=None):
    """
    Pickle-safe wrapper for log prior with custom inclination bounds.
    Bound at runtime to `likelihood_manager.log_prior`.
    """
    if phys_dict is None:
        try:
            phys_dict = self.samp_to_phys(x)
        except Exception:
            return -np.inf

    bounds = getattr(self, "_inclination_bounds", None)
    if bounds is not None:
        inc = phys_dict.get("inclination", np.nan)
        b_lo, b_hi = bounds
        if (not np.isfinite(inc)) or (inc < b_lo) or (inc > b_hi):
            return -np.inf

    return self.__class__._orig_get_lnprior_func(self, x, phys_dict=phys_dict)

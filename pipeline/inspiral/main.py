"""Run a TDInf + emcee analysis of the injected data (stage 2 of the pipeline).

Conditions the H1/L1 data, truncates them at the cutoff time (--Tcut-tm, in units
of T_M relative to the peak), builds the TDInf likelihood with the chosen waveform
model and samples it with emcee. Called by run_tdinf.sh, which sets the options of
the paper's runs; run `python main.py --help` for every option.

Settings that are not command-line options are the burn-in/reseed knobs below.

Walker initialisation (--walkerinit):
  flat       draw walkers uniformly from the prior support, keeping a margin of
             FLAT_INIT_BOUNDARY_FRACTION from each bound (used in the paper)
  reference  start in a small ball around the injected parameters in the metadata
             (optionally changed with the --override-* options)
  posterior  fit a Gaussian to the samples of an earlier run (--init-from-backend)
             and draw walkers from it; see the POSTERIOR_INIT_* knobs below
  resume     continue the run already stored in --output-h5 (no burn-in or reseed)

Debugging options (run before sampling; add --run-emcee to sample afterwards):
  --find-likelihood-slice   1D log-likelihood slice in inclination, saved as a PNG
  --one-d-optimizer         maximise the likelihood over inclination only
  --full-d-optimizer        local MAP and maximum-likelihood search in all parameters
  --create-hphc             save h_+, h_x at the reference parameters as CSV
  --output-waveform         save the projected reference waveform in each detector as CSV

Output: the emcee backend (--output-h5) and the post-processed samples in
physical parameters, written next to it with the extension .dat.
"""
import os
import sys
from contextlib import closing
from multiprocessing import Pool, cpu_count

import emcee
import lalsimulation as lalsim
import numpy as np
import pandas as pd
import tdinf.utils as utils

from analysis_utils import (
    create_run_sampler_arg_parser,
    enforce_inclination_prior_bounds,
    get_conditioned_time_and_data,
    init_walkers_from_backend_gaussian,
    init_walkers_from_flat_prior,
    init_walkers_from_reference_values,
    initialize_kwargs,
)
from debug_utils import (
    local_map_and_ml_search,
    max_inclination_1d,
    plot_likelihood_slice,
)

# ---------- burn-in and reseed knobs ----------
# A burn-in of RESEED_NBURN steps is run first and not saved. The walkers are then
# scored by their median log posterior over the last RESEED_WINDOW burn-in steps, and
# the worst RESEED_FRAC_REPLACE of them are moved onto copies of the best walkers
# (plus a small jitter), so no walkers are left stranded in low-probability regions.
RESEED_ENABLE = True        # False: no burn-in, all nsteps are production steps
RESEED_NBURN = 20000        # burn-in steps (at most half of --nsteps)
RESEED_WINDOW = 2000        # last burn-in steps used to score the walkers
RESEED_FRAC_REPLACE = 0.30  # fraction of walkers to replace
RESEED_JITTER = 1e-3        # jitter added to cloned walkers, in sampling coordinates
RESEED_SEED = 1234          # None for a different reseed on each run

# ---------- walker initialisation knobs ----------
FLAT_INIT_BOUNDARY_FRACTION = 0.02  # flat: margin kept from each prior bound, as a fraction of its width
POSTERIOR_INIT_DISCARD_FRAC = 0.5   # posterior: fraction of the earlier chain discarded as burn-in
POSTERIOR_INIT_THIN = 2             # posterior: thinning of the earlier chain
POSTERIOR_INIT_INFLATE = 1.25       # posterior: widen the fitted Gaussian by this factor

# ---------- convergence check ----------
# Production stops early once the chain is longer than CONVERGENCE_TAU_FACTOR
# autocorrelation times and tau has changed by less than CONVERGENCE_TAU_RTOL.
CONVERGENCE_CHECK_EVERY = 1000
CONVERGENCE_TAU_FACTOR = 100
CONVERGENCE_TAU_RTOL = 0.01

OVERRIDES = ("total_mass", "mass_ratio", "luminosity_distance", "inclination", "phase")
SPIN_KEYS = ("spin1_x", "spin1_y", "spin1_z", "spin2_x", "spin2_y", "spin2_z", "chi1_mag", "chi2_mag")


def sort_on_runtime(pos):
    """Order walkers by decreasing first coordinate, so emcee hands out the slowest likelihoods first."""
    p = np.atleast_2d(pos)
    idx = np.argsort(p[:, 0])[::-1]
    return p[idx], idx


def reseed_worst_walkers(p_last, logp_burn, frac_replace=0.30, window=500, jitter_scale=1e-3, seed=1234):
    """Replace stranded walkers before production to avoid split ensembles.

    p_last:    (nwalkers, ndim) last burn-in positions
    logp_burn: (nburn, nwalkers) burn-in log-probability history
    Each walker is scored by its median log probability over the last `window`
    steps. The worst `frac_replace` of the walkers are moved onto randomly chosen
    walkers from the best ones, plus Gaussian jitter of size `jitter_scale`.
    Returns the new positions and the indices of the replaced walkers.
    """
    rng = np.random.default_rng(seed)

    nburn, nwalkers = logp_burn.shape
    w = min(max(10, int(window)), nburn)
    score = np.median(logp_burn[-w:, :], axis=0)

    nrep = int(np.ceil(frac_replace * nwalkers))
    order = np.argsort(score)  # worst first
    worst = order[:nrep]
    donor_pool = order[-max(2 * nrep, 5):]

    p_new = np.array(p_last, copy=True, dtype=float)
    for wi in worst:
        donor = rng.choice(donor_pool)
        p_new[wi, :] = p_new[donor, :] + jitter_scale * rng.standard_normal(size=p_new.shape[1])

    return p_new, worst.tolist()


def load_reference_parameters(args, verbose=False):
    """Injection metadata in TDInf's naming, with the --override-* options applied.

    Missing spin parameters are set to zero, and the phase is also stored as the
    unit vector (phase_x, phase_y) that TDInf samples in.
    """
    reference_parameters = pd.read_csv(args.metadata).iloc[0].to_dict()
    reference_parameters["right_ascension"] = reference_parameters["ra"]
    reference_parameters["declination"] = reference_parameters["dec"]
    reference_parameters["polarization"] = reference_parameters["psi"]

    for key in OVERRIDES:
        value = getattr(args, "override_" + ("distance" if key == "luminosity_distance" else key))
        if value is not None:
            reference_parameters[key] = value
            if verbose:
                print(f"Overriding {key} to {value}")

    for key in SPIN_KEYS:
        reference_parameters.setdefault(key, 0.0)

    if "phase" in reference_parameters:
        reference_parameters["phase_x"] = np.cos(reference_parameters["phase"])
        reference_parameters["phase_y"] = np.sin(reference_parameters["phase"])
    return reference_parameters


def data_and_psd_paths(args):
    """Strain and PSD file of each detector: from --h1-data etc., or default names in --data-dir."""
    data_paths, psd_paths = {}, {}
    for ifo in args.ifos:
        data_paths[ifo] = getattr(args, f"{ifo.lower()}_data", None) or os.path.join(args.data_dir, f"{ifo}_injections.h5")
        psd_paths[ifo] = getattr(args, f"{ifo.lower()}_psd", None) or os.path.join(args.data_dir, f"{ifo}_PSD.dat")
    return data_paths, psd_paths


def set_frequencies(args, reference_parameters):
    """Fill in f_low, f_ref and f22_start from the metadata where they were not given."""
    if args.flow is None:
        args.flow = reference_parameters["f_min"]

    # LALSuite postprocessing requires f_ref > 0; NR injections may store f_ref = 0
    if args.fref is None or args.fref <= 0:
        args.fref = reference_parameters["f_ref"] if reference_parameters["f_ref"] > 0 else args.flow

    # f22_start is the frequency at which the raw waveform starts, before tapering. It is
    # below f_low, so the model waveform can be tapered in the same way as the injection.
    if args.f22_start is None:
        args.f22_start = reference_parameters.get("f22_start", 0.75 * args.flow)


def initial_walkers(args, likelihood_manager, backend, reference_parameters, nwalkers, ndim, verbose=False):
    """Starting positions for emcee, following --walkerinit (see the module docstring).

    Returns (p0, n_done, backend): the positions, the number of steps already in the
    backend, and the backend (recreated if an empty file could not be reset).
    """
    init_mode = args.walkerinit
    backend_path = args.output_h5

    if init_mode == "resume":
        if not os.path.exists(backend_path):
            raise FileNotFoundError(f"--walkerinit resume needs an existing backend at {backend_path}")
        try:
            p0 = backend.get_last_sample()
            n_done = backend.get_chain().shape[0]
            print(f"Resuming from {backend_path} after {n_done} steps")
            return p0, n_done, backend
        except AttributeError:
            # the file exists but holds no samples yet: start again from the prior
            print("WARNING: backend file holds no samples; starting a new run from the prior.")
            try:
                backend.reset(nwalkers, ndim)
            except OSError:
                os.remove(backend_path)
                backend = emcee.backends.HDFBackend(backend_path)
                backend.reset(nwalkers, ndim)
            p0 = init_walkers_from_flat_prior(likelihood_manager, nwalkers, verbose=verbose,
                                              boundary_fraction=FLAT_INIT_BOUNDARY_FRACTION)
            return p0, 0, backend

    backend.reset(nwalkers, ndim)
    if init_mode == "posterior":
        if args.init_from_backend is None:
            raise ValueError("--walkerinit posterior requires --init-from-backend.")
        p0 = init_walkers_from_backend_gaussian(
            likelihood_manager,
            backend_path=args.init_from_backend,
            nwalkers=nwalkers,
            discard_frac=POSTERIOR_INIT_DISCARD_FRAC,
            thin=POSTERIOR_INIT_THIN,
            inflate=POSTERIOR_INIT_INFLATE,
        )
    elif init_mode == "reference":
        p0 = init_walkers_from_reference_values(likelihood_manager, nwalkers, reference_parameters, verbose=verbose)
    elif init_mode == "flat":
        p0 = init_walkers_from_flat_prior(likelihood_manager, nwalkers, verbose=verbose,
                                          boundary_fraction=FLAT_INIT_BOUNDARY_FRACTION)
    else:
        raise ValueError(f"Unknown --walkerinit value '{init_mode}'.")
    return p0, 0, backend


def run_debug_options(args, likelihood_manager, reference_parameters, time_dict, out_dir, verbose=False):
    """Run the optional waveform exports, optimisers and likelihood slice requested on the command line."""
    if args.create_hphc:
        # h_+, h_x at the reference parameters, with zero phase
        params = dict(reference_parameters, phase=0)
        delta_t = time_dict[args.ifos[0]][1] - time_dict[args.ifos[0]][0]
        hp, hc = likelihood_manager.waveform_manager.get_hplus_hcross(
            params, delta_t, f22_start=args.f22_start, f_ref=args.fref)
        for pol, h in (("hp", hp), ("hc", hc)):
            path = os.path.join(out_dir, f"{args.prefix}_{pol}_waveform.csv")
            pd.DataFrame({"times": h.times.value, "strain": h.value}).to_csv(path, index=False)
            print(f"Saved {path}")

    if args.output_waveform:
        # reference waveform projected onto each detector and windowed as in the likelihood
        ref_wf_dict = likelihood_manager.waveform_manager.get_projected_waveform(
            reference_parameters, args.ifos, time_dict,
            f22_start=args.f22_start, f_ref=args.fref, window=True,
        )
        for ifo in args.ifos:
            path = os.path.join(out_dir, f"{args.prefix}_{ifo}_reference_waveform.csv")
            pd.DataFrame({"times": time_dict[ifo], "strain": ref_wf_dict[ifo]}).to_csv(path, index=False)
            print(f"Saved {path}")

    if args.one_d_optimizer:
        print("\n##### 1D inclination maximisation #####")
        inc_hat, logL_hat, res = max_inclination_1d(likelihood_manager, reference_parameters)
        print(f"inclination = {inc_hat}, logL = {logL_hat}, success = {res.success}")

    if args.full_d_optimizer:
        print("\n##### Bounded MAP and ML search #####")
        out = local_map_and_ml_search(likelihood_manager, reference_parameters, step=3.0)
        print("Start logPost:", out["logpost_x0"])
        print("MAP  success:", out["res_map"].success, "logPost:", out["logPost_map"], "logL:", out["logL_map"])
        print(" ML  success:", out["res_ml"].success, "logPost:", out["logPost_ml"], "logL:", out["logL_ml"])
        for k in OVERRIDES:
            if k in out["p_map"]:
                print("MAP", k, out["p_map"][k])
            if k in out["p_ml"]:
                print(" ML", k, out["p_ml"][k])

    if args.find_likelihood_slice:
        param = "inclination"
        print(f"\n##### Likelihood slice in {param} #####")
        liks, vals = plot_likelihood_slice(likelihood_manager, param, reference_parameters, args,
                                           out_dir=out_dir, verbose=verbose, plot=False)
        peak = vals[np.argmax(liks)]
        ref = reference_parameters[param]
        print(f"Peak at {param} = {peak:.4f} ({peak / np.pi:.4f} pi), log_lik = {np.max(liks):.2f}")
        print(f"Reference {param} = {ref:.4f} ({ref / np.pi:.4f} pi), difference = {peak - ref:.4f}")


def main():
    """Set up the data and likelihood, run any debugging options, then sample with emcee."""
    args = create_run_sampler_arg_parser().parse_args()
    verbose = args.verbose
    backend_path = args.output_h5
    out_dir = os.path.dirname(os.path.abspath(backend_path))

    reference_parameters = load_reference_parameters(args, verbose=verbose)
    data_paths, psd_paths = data_and_psd_paths(args)
    set_frequencies(args, reference_parameters)
    if verbose:
        print(f"Mode: {args.mode}, interferometers: {args.ifos}")
        print(f"Data paths: {data_paths}")
        print(f"PSD paths: {psd_paths}")
        print(f"Frequencies: f_ref = {args.fref} Hz, f_low = {args.flow} Hz, f22_start = {args.f22_start} Hz")

    # condition and truncate the data
    kwargs = initialize_kwargs(args, reference_parameters)
    wf_manager = utils.WaveformManager(args.ifos, vary_time=False, vary_skypos=False, **kwargs)
    time_dict, data_dict, psd_dict = get_conditioned_time_and_data(
        args, wf_manager=wf_manager, reference_parameters=reference_parameters,
        data_paths=data_paths, psd_paths=psd_paths, verbose=verbose,
    )

    # With --no-spins, the parameterisation is built from a spinless approximant and
    # the requested waveform model (e.g. NRSur7dq4) is restored afterwards.
    kwargs_params = dict(kwargs)
    if args.no_spins:
        kwargs_params["approx"] = "EOBNRv2"

    likelihood_manager = utils.LnLikelihoodManager(
        psd_dict=psd_dict, time_dict=time_dict, data_dict=data_dict,
        vary_time=False, vary_skypos=False,
        f_max=args.fmax, only_prior=args.only_prior,
        **kwargs_params,
    )
    enforce_inclination_prior_bounds(likelihood_manager, args.inclination_prior_bounds, verbose=verbose)

    if args.no_spins:
        likelihood_manager.waveform_manager.approx_name = kwargs["approx"]
        likelihood_manager.waveform_manager.approximant = lalsim.SimInspiralGetApproximantFromString(kwargs["approx"])

    run_debug_options(args, likelihood_manager, reference_parameters, time_dict, out_dir, verbose=verbose)

    if not args.run_emcee:
        print("--run-emcee not set; exiting without sampling.")
        sys.exit(0)

    # ---------- set up the sampler ----------
    nwalkers = args.nwalkers
    ndim = len(likelihood_manager.sampled_keys)
    print(f"Sampling {ndim} parameters: {likelihood_manager.sampled_keys}")
    if verbose:
        print(f"Fixed parameters: {likelihood_manager.fixed}")

    backend = emcee.backends.HDFBackend(backend_path)
    p0, n_done, backend = initial_walkers(args, likelihood_manager, backend, reference_parameters, nwalkers, ndim,
                                 verbose=verbose)
    nsteps = args.nsteps - n_done
    reseed = RESEED_ENABLE and args.walkerinit != "resume"

    # one thread per worker process
    if args.ncpu > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
    print(f"Running with {args.ncpu} of {cpu_count()} cores")

    with closing(Pool(processes=args.ncpu)) as pool:

        # ---------- burn-in (not saved) and reseed ----------
        nsteps_to_run = nsteps
        if reseed and nsteps > 0:
            nburn = min(RESEED_NBURN, max(0, nsteps // 2))
            if nburn > 0:
                print(f"Burn-in: {nburn} steps (not saved), then {nsteps - nburn} production steps")
                burn_sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, likelihood_manager.get_log_posterior,
                    pool=pool, runtime_sortingfn=sort_on_runtime, kwargs=kwargs,
                )
                burn_sampler.run_mcmc(p0, nburn, progress=True)
                p0, worst_idx = reseed_worst_walkers(
                    burn_sampler.get_last_sample().coords, burn_sampler.get_log_prob(),
                    frac_replace=RESEED_FRAC_REPLACE,
                    window=min(RESEED_WINDOW, nburn),
                    jitter_scale=RESEED_JITTER,
                    seed=RESEED_SEED,
                )
                print(f"Reseed: replaced {len(worst_idx)} walkers: {worst_idx}")
                nsteps_to_run = nsteps - nburn

        # ---------- production run (saved to the backend) ----------
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, likelihood_manager.get_log_posterior,
            backend=backend, pool=pool, runtime_sortingfn=sort_on_runtime, kwargs=kwargs,
        )
        if nsteps_to_run > 0:
            old_tau = np.inf
            for _ in sampler.sample(p0, iterations=nsteps_to_run, progress=True):
                if sampler.iteration % CONVERGENCE_CHECK_EVERY:
                    continue
                tau = sampler.get_autocorr_time(tol=0)
                converged = np.all(tau * CONVERGENCE_TAU_FACTOR < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < CONVERGENCE_TAU_RTOL)
                if converged:
                    break
                old_tau = tau

        pool.terminate()

    # ---------- post-process into physical parameters and save ----------
    print(f"Production chain shape (steps, walkers, parameters): {sampler.get_chain().shape}")
    df = utils.postprocess_samples(sampler, likelihood_manager, **kwargs)
    sample_path = backend_path.replace("h5", "dat")
    df.to_csv(sample_path, sep=" ", index=False)
    print(f"Saved {sample_path}")


if __name__ == "__main__":
    main()

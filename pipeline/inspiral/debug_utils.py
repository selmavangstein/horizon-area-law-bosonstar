"""Debugging tools for main.py: likelihood slices and local optimisers.

These check the likelihood set-up (e.g. that it peaks near the injected values)
before committing to a long emcee run. None of them were needed for the paper's
results. They are switched on with the --find-likelihood-slice, --one-d-optimizer
and --full-d-optimizer options of main.py.
"""
import os

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def plot_likelihood_slice(likelihood_manager, param_to_vary, reference_parameters, args, num_points=100,
                          out_dir="output", verbose=False, plot=True):
    """Log likelihood along one parameter across its prior range, others fixed at the reference values.

    Saves the slice as <out_dir>/likelihood_slice_<param>.png (and shows it if plot=True).
    Returns (log_likelihoods, parameter_values).
    """
    import matplotlib.pyplot as plt

    fixed_params = dict(reference_parameters)
    if "phase_x" in fixed_params and "phase_y" in fixed_params:
        fixed_params["phase"] = np.arctan2(fixed_params["phase_y"], fixed_params["phase_x"])

    bounds_dict = {
        "total_mass": args.total_mass_prior_bounds,
        "mass_ratio": args.mass_ratio_prior_bounds,
        "luminosity_distance": args.luminosity_distance_prior_bounds,
        "inclination": args.inclination_prior_bounds,
    }
    param_bounds = bounds_dict.get(param_to_vary)
    if param_bounds is None:
        raise ValueError(f"Parameter {param_to_vary} not recognized for likelihood slice.")

    param_values = np.linspace(param_bounds[0], param_bounds[1], num_points)
    log_likelihoods = []
    for val in param_values:
        params = fixed_params.copy()
        params[param_to_vary] = val
        log_likelihoods.append(likelihood_manager.get_log_likelihood(params))

    plt.figure(figsize=(8, 6))
    plt.plot(param_values, log_likelihoods, label="Log Likelihood Slice")
    plt.axvline(reference_parameters[param_to_vary], color="r", linestyle="--", label="Injected Value")
    plt.xlabel(param_to_vary)
    plt.ylabel("Log Likelihood")
    plt.title(f"Likelihood Slice varying {param_to_vary}")
    plt.legend()
    plt.grid()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"likelihood_slice_{param_to_vary}.png")
    plt.savefig(path)
    if plot:
        plt.show()
    plt.close()
    print(f"Saved {path}")

    return log_likelihoods, param_values


def max_inclination_1d(likelihood_manager, reference_parameters):
    """Maximise the likelihood over inclination in [0, pi], others fixed at the reference values.

    Returns (inclination, log_likelihood, scipy result).
    """
    def neg_logL(inc):
        """Negative log likelihood at inclination inc."""
        p = dict(reference_parameters)
        p["inclination"] = float(inc)
        return -likelihood_manager.get_log_likelihood(p)

    res = minimize_scalar(
        neg_logL,
        bounds=(0.0, np.pi),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return res.x, -res.fun, res


def _clip_open_interval(u, lo, hi, eps=1e-6, atol=0.0):
    """Move u off the bounds of [lo, hi] by eps of the width, so its logistic transform is finite."""
    u = float(u)
    lo = float(lo)
    hi = float(hi)

    width = hi - lo
    if width <= 0:
        raise ValueError(f"Invalid interval [{lo}, {hi}]")

    tol = max(
        float(atol),
        10.0 * np.finfo(float).eps * max(1.0, abs(lo), abs(hi), abs(u)),
    )
    if u <= lo + tol:
        u = lo + eps * width
    if u >= hi - tol:
        u = hi - eps * width
    return u


def make_x0_from_physical(likelihood_manager, reference_parameters, eps=1e-3, atol=0.0, verbose=False):
    """Sampling coordinates of the reference parameters (the starting point of the optimisers)."""
    sampled_keys = list(likelihood_manager.sampled_keys)
    lp_by_logistic = {lp.logistic_name: lp for lp in likelihood_manager.logistic_parameters}

    x0 = []
    for k in sampled_keys:
        if k in ("phase_x", "phase_y"):
            x0.append(float(reference_parameters[k]))
            continue

        if not k.startswith("x_"):
            raise KeyError(f"Unknown sampled key {k}")

        lp = lp_by_logistic[k]
        if getattr(lp, "trig_name", None) == "cos_inclination":
            inc = float(np.clip(reference_parameters["inclination"], 1e-6, np.pi - 1e-6))
            if verbose:
                print(f"Using trig-logistic for {k}: inclination={inc}")
            x0.append(float(lp.physical_to_logistic(inc)))
            continue

        u = float(reference_parameters[lp.physical_name])
        lo, hi = lp.limit
        u_before = u
        u = _clip_open_interval(u, lo, hi, eps=eps, atol=atol)
        if verbose and (u != u_before):
            print(f"Clipped {k} ({lp.physical_name}) from {u_before} to {u} within ({lo},{hi})")
        x0.append(float(lp.physical_to_logistic(u)))

    return np.asarray(x0, dtype=float)


def decode_x_to_physical(likelihood_manager, x):
    """Physical parameters (dict) of a point x in sampling coordinates; unsampled spins are zero."""
    sampled_keys = list(likelihood_manager.sampled_keys)
    lp_by_logistic = {lp.logistic_name: lp for lp in likelihood_manager.logistic_parameters}

    phys = dict(likelihood_manager.fixed) if hasattr(likelihood_manager, "fixed") else {}
    for k, xv in zip(sampled_keys, x):
        xv = float(xv)

        if k == "phase_x":
            phys["phase_x"] = xv
            continue
        if k == "phase_y":
            phys["phase_y"] = xv
            continue
        if not k.startswith("x_"):
            raise KeyError(f"Unknown sampled key {k}")

        lp = lp_by_logistic[k]
        if getattr(lp, "trig_name", None) == "cos_inclination":
            inc = float(lp.logistic_to_physical(xv))
            phys["inclination"] = inc
            phys["cos_inclination"] = float(np.cos(inc))
        else:
            phys[lp.physical_name] = float(lp.logistic_to_physical(xv))

    if "phase_x" in phys and "phase_y" in phys:
        phys["phase"] = float(np.arctan2(phys["phase_y"], phys["phase_x"]))

    for key in ["spin1_x", "spin1_y", "spin1_z", "spin2_x", "spin2_y", "spin2_z"]:
        phys.setdefault(key, 0.0)

    return phys


def make_local_bounds(sampled_keys, x0, step=3.0):
    """Optimiser bounds: +-step around x0 in each logistic coordinate, [-1, 1] for the phase vector."""
    bounds = []
    for k, x0k in zip(sampled_keys, x0):
        if k in ("phase_x", "phase_y"):
            bounds.append((-1.0, 1.0))
        else:
            bounds.append((float(x0k - step), float(x0k + step)))
    return bounds


def local_map_and_ml_search(likelihood_manager, reference_parameters, step=3.0):
    """Bounded L-BFGS-B searches for the maximum posterior (MAP) and maximum likelihood (ML) points.

    Both start at the reference parameters and stay within make_local_bounds(step).
    Returns a dict with the start point, both optimiser results, and the MAP and ML
    points in sampling (x_*) and physical (p_*) coordinates with their logL and log posterior.
    """
    sampled_keys = list(likelihood_manager.sampled_keys)

    x0 = make_x0_from_physical(likelihood_manager, reference_parameters, eps=1e-3)
    lp0 = likelihood_manager.get_log_posterior(x0)
    if not np.isfinite(lp0):
        raise ValueError(f"Invalid start point: logpost(x0)={lp0}")

    bounds = make_local_bounds(sampled_keys, x0, step=step)

    def obj_post(x):
        """Negative log posterior (large if invalid)."""
        lp = likelihood_manager.get_log_posterior(x)
        return 1e99 if not np.isfinite(lp) else -lp

    def obj_like(x):
        """Negative log likelihood (large if invalid)."""
        x_phys = decode_x_to_physical(likelihood_manager, x)
        for key in ["spin1_x", "spin1_y", "spin1_z", "spin2_x", "spin2_y", "spin2_z"]:
            x_phys.setdefault(key, 0.0)
        x_phys.setdefault("chi1_mag", 0.0)
        x_phys.setdefault("chi2_mag", 0.0)
        ll = likelihood_manager.get_log_likelihood(x_phys)
        return 1e99 if not np.isfinite(ll) else -ll

    res_map = minimize(obj_post, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1500})
    res_ml = minimize(obj_like, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1500})

    x_map = res_map.x
    x_ml = res_ml.x
    p_map = decode_x_to_physical(likelihood_manager, x_map)
    p_ml = decode_x_to_physical(likelihood_manager, x_ml)

    return {
        "x0": x0,
        "logpost_x0": lp0,
        "res_map": res_map,
        "x_map": x_map,
        "p_map": p_map,
        "logL_map": likelihood_manager.get_log_likelihood(p_map),
        "logPost_map": likelihood_manager.get_log_posterior(x_map),
        "res_ml": res_ml,
        "x_ml": x_ml,
        "p_ml": p_ml,
        "logL_ml": likelihood_manager.get_log_likelihood(p_ml),
        "logPost_ml": likelihood_manager.get_log_posterior(x_ml),
    }

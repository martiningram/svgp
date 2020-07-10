import jax.numpy as jnp
import numpy as np
from ml_tools.constrain import apply_transformation
from svgp.jax.analytic_expectations import ppm_likelihood_berman_turner_expectation
from svgp.jax.kl import normal_kl_1d
from ml_tools.flattening import flatten_and_summarise, reconstruct
from jax import value_and_grad, jit
from scipy.optimize import minimize


def calculate_likelihood(
    theta, fg_ids, fg_covs, bg_covs, fg_covs_thin, bg_covs_thin, total_area, n_s
):

    # Project
    rel_means_fg = theta["w_means"][:, fg_ids].T
    rel_vars_fg = theta["w_vars"][:, fg_ids].T

    pred_mean_fg = jnp.sum(fg_covs * rel_means_fg, axis=1)
    pred_vars_fg = jnp.sum(fg_covs ** 2 * rel_vars_fg, axis=1)

    # Add thinning
    pred_mean_fg_thin = jnp.squeeze(fg_covs_thin @ theta["w_means_thin"])
    pred_vars_fg_thin = jnp.squeeze(fg_covs_thin ** 2 @ theta["w_vars_thin"])

    pred_mean_fg = pred_mean_fg + pred_mean_fg_thin
    pred_vars_fg = pred_vars_fg + pred_vars_fg_thin

    # Add intercept too
    pred_mean_fg = pred_mean_fg + theta["intercept_means"][fg_ids]
    pred_vars_fg = pred_vars_fg + theta["intercept_vars"][fg_ids]

    pred_mean_bg = bg_covs @ theta["w_means"] + theta["intercept_means"].reshape(1, -1)
    pred_vars_bg = bg_covs ** 2 @ theta["w_vars"] + theta["intercept_vars"].reshape(
        1, -1
    )

    pred_mean_bg = pred_mean_bg + bg_covs_thin @ theta["w_means_thin"]
    pred_vars_bg = pred_vars_bg + bg_covs_thin ** 2 @ theta["w_vars_thin"]

    # Berman-Turner weights
    n_bg = bg_covs.shape[0]
    n_fg = fg_covs.shape[0]
    weight = total_area / n_bg

    z_fg = jnp.repeat(1.0 / weight, n_fg)
    z_bg = jnp.repeat(0.0, n_bg * n_s)

    fg_result = ppm_likelihood_berman_turner_expectation(
        pred_mean_fg, pred_vars_fg, z_fg, weight, sum_result=True
    )

    bg_result = ppm_likelihood_berman_turner_expectation(
        pred_mean_bg.reshape(-1),
        pred_vars_bg.reshape(-1),
        z_bg,
        weight,
        sum_result=True,
    )

    return fg_result + bg_result


def calculate_kl(theta):

    kl_weights = normal_kl_1d(
        theta["w_means"], theta["w_vars"], theta["w_prior_mean"], theta["w_prior_var"]
    )

    kl_intercept = normal_kl_1d(
        theta["intercept_means"],
        theta["intercept_vars"],
        theta["intercept_prior_mean"],
        theta["intercept_prior_var"],
    )

    kl_thin = normal_kl_1d(theta["w_means_thin"], theta["w_vars_thin"], 0.0, 1.0)

    return jnp.sum(kl_weights) + jnp.sum(kl_intercept) + jnp.sum(kl_thin)


def fit(
    fg_covs, bg_covs, species_ids, total_area, fg_covs_thin=None, bg_covs_thin=None
):

    n_c = fg_covs.shape[1]
    n_s = len(np.unique(species_ids))
    n_c_thin = 0 if fg_covs_thin is None else fg_covs_thin.shape[1]

    init_theta = {
        "w_means": jnp.zeros((n_c, n_s)),
        "log_w_vars": jnp.zeros((n_c, n_s)) - 5,
        "intercept_means": jnp.zeros(n_s),
        "log_intercept_vars": jnp.zeros(n_s),
        "w_prior_mean": jnp.zeros((n_c, 1)),
        "log_w_prior_var": jnp.zeros((n_c, 1)) - 5,
        "intercept_prior_mean": jnp.array(0.0),
        "log_intercept_prior_var": jnp.array(0.0),
        # Thinning is assumed constant across species
        "w_means_thin": jnp.zeros((n_c_thin, 1)),
        "log_w_vars_thin": jnp.zeros((n_c_thin, 1)),
    }

    flat_theta, summary = flatten_and_summarise(**init_theta)

    def to_minimize(flat_theta):

        theta = reconstruct(flat_theta, summary, jnp.reshape)
        theta = apply_transformation(theta, "log_", jnp.exp, "")

        lik = calculate_likelihood(
            theta,
            species_ids,
            fg_covs,
            bg_covs,
            fg_covs_thin,
            bg_covs_thin,
            total_area,
            n_s,
        )
        kl = calculate_kl(theta)

        return -(lik - kl)

    with_grad = jit(value_and_grad(to_minimize))

    def annotated_with_grad(flat_theta):

        flat_theta = jnp.array(flat_theta)

        obj, grad = with_grad(flat_theta)

        print(obj, jnp.linalg.norm(grad))

        return np.array(obj).astype(np.float64), np.array(grad).astype(np.float64)

    result = minimize(annotated_with_grad, flat_theta, method="L-BFGS-B", jac=True)
    final_theta = reconstruct(result.x, summary, jnp.reshape)
    final_theta = apply_transformation(final_theta, "log_", jnp.exp, "")

    return final_theta


def predict(theta, new_covs, new_thin_covs=None):

    pred_means = new_covs @ theta["w_means"]
    pred_vars = new_covs ** 2 @ theta["w_vars"]

    pred_means = pred_means + theta["intercept_means"].reshape(1, -1)
    pred_vars = pred_vars + theta["intercept_vars"].reshape(1, -1)

    if new_thin_covs is not None:

        pred_mean_thin = new_thin_covs @ theta["w_means_thin"]
        pred_var_thin = new_thin_covs ** 2 @ theta["w_vars_thin"]

        pred_means = pred_means + pred_mean_thin
        pred_vars = pred_vars + pred_var_thin

    return pred_means, pred_vars

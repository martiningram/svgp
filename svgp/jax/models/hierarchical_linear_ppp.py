import jax.numpy as jnp
import numpy as np
from ml_tools.constrain import apply_transformation
from svgp.jax.analytic_expectations import expected_ppm_likelihood_quadrature_approx
from svgp.jax.quadrature import expectation
from svgp.jax.likelihoods import square_cox_lik
from svgp.jax.kl import normal_kl_1d
from ml_tools.flattening import flatten_and_summarise, reconstruct
from jax import value_and_grad, jit
from scipy.optimize import minimize
from functools import partial
from jax.scipy.stats import gamma, norm


@partial(jit, static_argnums=(8, 9))
def calculate_likelihood(
    theta,
    fg_ids,
    fg_covs,
    bg_covs,
    fg_covs_thin,
    bg_covs_thin,
    quad_weights,
    counts,
    n_s,
    n_fg,
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

    quad_weights_bg = jnp.tile(quad_weights, (n_s, 1)).T
    ys_bg = jnp.zeros_like(quad_weights_bg)

    n_fg = pred_mean_fg.shape[0]

    ys_full = jnp.concatenate([counts, ys_bg.reshape(-1)])
    weights_full = jnp.concatenate([jnp.repeat(-1, n_fg), quad_weights_bg.reshape(-1)])
    pred_mean_full = jnp.concatenate([pred_mean_fg, pred_mean_bg.reshape(-1)])
    pred_var_full = jnp.concatenate([pred_vars_fg, pred_vars_bg.reshape(-1)])

    liks = expected_ppm_likelihood_quadrature_approx(
        ys_full, weights_full, pred_mean_full, pred_var_full
    )

    # liks = expectation(
    #     ys_full,
    #     pred_var_full,
    #     pred_mean_full,
    #     partial(square_cox_lik, weights=weights_full),
    # )

    return jnp.sum(liks)


def calculate_kl(theta):

    kl_weights = normal_kl_1d(
        theta["w_means"], theta["w_vars"], theta["w_prior_mean"], theta["w_prior_var"]
    )

    kl_intercept = normal_kl_1d(
        theta["intercept_means"], theta["intercept_vars"], -2.0, 1.0
    )

    kl_thin = normal_kl_1d(
        theta["w_means_thin"], theta["w_vars_thin"], 0.0, theta["w_prior_var_thin"]
    )

    return jnp.sum(kl_weights) + jnp.sum(kl_intercept) + jnp.sum(kl_thin)


def fit(
    fg_covs,
    bg_covs,
    species_ids,
    quad_weights,
    counts,
    fg_covs_thin=None,
    bg_covs_thin=None,
):

    n_c = fg_covs.shape[1]
    n_s = len(np.unique(species_ids))
    n_c_thin = 0 if fg_covs_thin is None else fg_covs_thin.shape[1]
    n_fg = fg_covs.shape[0]
    n_bg = bg_covs.shape[0]

    fg_covs_thin = fg_covs_thin if fg_covs_thin is not None else jnp.zeros((n_fg, 0))
    bg_covs_thin = bg_covs_thin if bg_covs_thin is not None else jnp.zeros((n_bg, 0))

    init_theta = {
        "w_means": jnp.zeros((n_c, n_s)),
        "log_w_vars": jnp.log(jnp.tile(1.0 / n_c, (n_c, n_s))) - 5,
        "intercept_means": jnp.zeros(n_s) - 5,
        "log_intercept_vars": jnp.zeros(n_s) - 5,
        "w_prior_mean": jnp.zeros((n_c, 1)),
        "log_w_prior_var": jnp.log(jnp.tile(1.0 / n_c, (n_c, 1))) - 5,
        # Thinning is assumed constant across species
        "w_means_thin": jnp.zeros((n_c_thin, 1)),
        "log_w_vars_thin": jnp.zeros((n_c_thin, 1)) - 10
        if n_c_thin == 0
        else jnp.log(jnp.tile(1.0 / (n_c_thin), (n_c_thin, 1))) - 10,
        "log_w_prior_var_thin": jnp.array(0.0)
        if n_c_thin == 0
        else jnp.log(1.0 / (n_c_thin)) - 10,
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
            quad_weights,
            counts,
            n_s,
            n_fg,
        )
        kl = calculate_kl(theta)

        prior = jnp.sum(gamma.logpdf(theta["w_prior_var"], 0.5, scale=1.0 / n_c))
        prior = prior + jnp.sum(
            norm.logpdf(theta["w_prior_mean"], 0.0, scale=jnp.sqrt(1.0 / n_c))
        )

        return -(lik - kl + prior)

    with_grad = jit(value_and_grad(to_minimize))

    def annotated_with_grad(flat_theta, summary):

        flat_theta = jnp.array(flat_theta)

        obj, grad = with_grad(flat_theta)

        print(obj, jnp.linalg.norm(grad))

        if jnp.isnan(obj) or jnp.isinf(obj) or jnp.any(jnp.isnan(grad)):
            import ipdb

            problem = reconstruct(flat_theta, summary, jnp.reshape)

            ipdb.set_trace()

        return np.array(obj).astype(np.float64), np.array(grad).astype(np.float64)

    result = minimize(
        partial(annotated_with_grad, summary=summary),
        flat_theta,
        method="L-BFGS-B",
        jac=True,
    )
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

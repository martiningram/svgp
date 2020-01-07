# This is a convenience module to quickly and easily fit MOGP models.
# It's a bit experimental!
from typing import NamedTuple, Tuple
from scipy.stats import norm
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from functools import partial
from scipy.optimize import minimize
from ml_tools.gp import find_starting_z
from svgp.tf.utils import get_initial_values_from_kernel
from svgp.tf.likelihoods import bernoulli_probit_lik
from ml_tools.flattening import flatten_and_summarise_tf, reconstruct_tf
from .sogp_classifier import kern_lookup
from .mogp import (create_ls, compute_mogp_kl_term, project_latents,
                   calculate_approximate_means_and_vars, expectation)
from .kl import normal_kl_1d
from .config import DTYPE


# TODO: Maybe enforce that this is immutable somehow?
class MOGPResult(NamedTuple):

    L_elts: np.ndarray
    mu: np.ndarray
    kernel: str
    lengthscales: np.ndarray
    intercept_means: np.float64
    intercept_vars: np.float64
    w_means: np.float64
    w_vars: np.float64
    Z: np.ndarray


def get_kernel_funs(base_kern_fun, lscales):

    kerns = [partial(base_kern_fun, lengthscales=cur_lscales, alpha=1.) for
             cur_lscales in lscales]

    return kerns


def compute_intercept_kl(intercept_means, intercept_vars, intercept_prior_var):

    kl = normal_kl_1d(
        intercept_means, intercept_vars, tf.constant(0., dtype=DTYPE),
        intercept_prior_var
    )

    return tf.reduce_sum(kl)


def compute_kl_term(ms, Ls, ks, Z, w_means, w_vars, w_prior_mean, w_prior_var,
                    intercept_means, intercept_vars, intercept_prior_var):

    intercept_kl = compute_intercept_kl(intercept_means, intercept_vars,
                                        intercept_prior_var)

    mogp_kl = compute_mogp_kl_term(ms, Ls, ks, Z, w_means, w_vars,
                                   w_prior_mean, w_prior_var)

    return intercept_kl + mogp_kl


def compute_predictions(X, Z, ms, Ls, ks, w_means, w_vars, intercept_means,
                        intercept_vars):

    m_proj, var_proj = project_latents(X, Z, ms, Ls, ks)
    m_out, var_out = calculate_approximate_means_and_vars(
        m_proj, var_proj, w_means, tf.sqrt(w_vars))

    m_out = m_out + intercept_means
    var_out = var_out + intercept_vars

    return m_out, var_out


def compute_likelihood_term(X, y, Z, ms, Ls, ks, w_means, w_vars,
                            intercept_means, intercept_vars):

    pred_mean, pred_var = compute_predictions(
        X, Z, ms, Ls, ks, w_means, w_vars, intercept_means, intercept_vars)

    log_liks = expectation(tf.reshape(y, (-1,)), tf.reshape(pred_var, (-1,)),
                           tf.reshape(pred_mean, (-1,)), bernoulli_probit_lik)

    return tf.reduce_sum(log_liks)


def fit(X: np.ndarray,
        y: np.ndarray,
        n_inducing: int = 100,
        n_latent: int = 10,
        kernel: str = 'matern_3/2',
        kernel_lengthscale_prior: Tuple[float, float] = (3, 1 / 3),
        bias_variance_prior: Tuple[float, float] = (3 / 2, 3 / 2),
        random_seed: int = 2) \
        -> MOGPResult:

    np.random.seed(random_seed)

    # Note that input _must_ be scaled. Some way to enforce that?
    kernel_fun = kern_lookup[kernel]

    n_cov = X.shape[1]
    n_out = y.shape[1]

    # Set initial values
    start_lengthscales = np.random.uniform(2., 4., size=(n_latent, n_cov))

    Z = find_starting_z(X, n_inducing)
    Z = np.tile(Z, (n_latent, 1, 1))

    start_kernel_funs = get_kernel_funs(kernel_fun, start_lengthscales)

    init_Ls = np.stack([
        get_initial_values_from_kernel(cur_z, cur_kernel_fun)
        for cur_z, cur_kernel_fun in zip(Z, start_kernel_funs)
    ])

    init_ms = np.zeros((n_latent, n_inducing))
    w_prior_var_init = np.array(0.1)

    start_intercept_means = np.zeros(n_out)
    start_intercept_var = np.ones(n_out)
    intercept_prior_var_init = np.array(0.4)

    init_theta = {
        'L_elts': init_Ls,
        'mu': init_ms,
        'w_prior_var': w_prior_var_init,
        'intercept_means': start_intercept_means,
        'intercept_vars': start_intercept_var,
        'intercept_prior_var': intercept_prior_var_init,
        'w_means': np.random.randn(n_latent, n_out) * 0.01,
        'w_vars': np.ones((n_latent, n_out)),
        'lscales': np.sqrt(start_lengthscales),
        'Z': Z,
    }

    flat_theta, summary = flatten_and_summarise_tf(**init_theta)

    X = tf.constant(X.astype(np.float32))
    y = tf.constant(y.astype(np.float32))
    w_prior_mean = tf.constant(0., dtype=DTYPE)

    lscale_prior = tfp.distributions.Gamma(*kernel_lengthscale_prior)
    bias_var_prior = tfp.distributions.Gamma(*bias_variance_prior)

    # TODO: Think about priors for W?

    def to_minimize_with_grad(x):

        with tf.GradientTape() as tape:

            x_tf = tf.constant(x)
            x_tf = tf.cast(x_tf, tf.float32)

            tape.watch(x_tf)

            theta = reconstruct_tf(x_tf, summary)

            # Square the important parameters
            (lscales, w_prior_var, intercept_vars, intercept_prior_var,
             w_vars) = (theta['lscales']**2, theta['w_prior_var']**2,
                        theta['intercept_vars']**2,
                        theta['intercept_prior_var']**2,
                        theta['w_vars']**2)

            print(lscales)
            print(intercept_prior_var)
            print(theta['intercept_means'])

            Ls = create_ls(theta['L_elts'], n_inducing, n_latent)

            kern_funs = get_kernel_funs(kernel_fun, lscales)

            kl = compute_kl_term(theta['mu'], Ls, kern_funs, theta['Z'],
                                 theta['w_means'], w_vars, w_prior_mean,
                                 w_prior_var, theta['intercept_means'],
                                 intercept_vars, intercept_prior_var)

            lik = compute_likelihood_term(
                X, y, theta['Z'], theta['mu'], Ls, kern_funs, theta['w_means'],
                w_vars, theta['intercept_means'], intercept_vars)

            objective = -(lik - kl)

            objective = objective - (
                tf.reduce_sum(lscale_prior.log_prob(lscales))
                + bias_var_prior.log_prob(intercept_prior_var)
            )

            grad = tape.gradient(objective, x_tf)

        print(objective, np.linalg.norm(grad.numpy()))

        return (objective.numpy().astype(np.float64),
                grad.numpy().astype(np.float64))

    result = minimize(to_minimize_with_grad, flat_theta, jac=True,
                      method='L-BFGS-B')

    final_theta = reconstruct_tf(result.x, summary)
    final_theta = {x: y.numpy() for x, y in final_theta.items()}

    # Build the results
    fit_result = MOGPResult(
        L_elts=final_theta['L_elts'],
        mu=final_theta['mu'],
        kernel=kernel,
        lengthscales=final_theta['lscales']**2,
        intercept_means=final_theta['intercept_means'],
        intercept_vars=final_theta['intercept_vars']**2,
        w_means=final_theta['w_means'],
        w_vars=final_theta['w_vars']**2,
        Z=final_theta['Z'])

    return fit_result


def predict(fit_result: MOGPResult, X_new: np.ndarray):
    # TODO: Is there something I can do about the casts here?
    # TODO: Should there be an option to predict the latents only?

    n_inducing = fit_result.mu.shape[1]
    n_latent = fit_result.mu.shape[0]

    L = create_ls(fit_result.L_elts.astype(np.float32), n_inducing,
                  n_latent)

    base_kern = kern_lookup[fit_result.kernel]

    k_funs = get_kernel_funs(base_kern,
                             fit_result.lengthscales.astype(np.float32))

    pred_mean, pred_var = compute_predictions(
        X_new.astype(np.float32), fit_result.Z.astype(np.float32),
        fit_result.mu.astype(np.float32), L, k_funs,
        fit_result.w_means.astype(np.float32),
        fit_result.w_vars.astype(np.float32),
        fit_result.intercept_means.astype(np.float32),
        fit_result.intercept_vars.astype(np.float32)
    )

    return pred_mean.numpy(), np.sqrt(pred_var)


def predict_probs(fit_result: MOGPResult, X_new: np.ndarray,
                  n_draws: int = 1000):

    n_data = X_new.shape[0]
    n_out = fit_result.w_vars.shape[1]
    all_probs = np.zeros((n_data, n_out))

    pred_mean, pred_var = predict(fit_result, X_new)
    pred_sd = np.sqrt(pred_var)

    for i, (cur_means, cur_sds) in enumerate(zip(pred_mean.T, pred_sd.T)):

        cur_draws = np.random.normal(cur_means, cur_sds,
                                     size=(n_draws, n_data))
        probs = norm.cdf(cur_draws)
        probs = np.mean(probs, axis=0)

        all_probs[:, i] = probs

    return all_probs


def save_results(fit_result: MOGPResult, target_file: str):

    dict_version = fit_result._asdict()
    np.savez(target_file, **dict_version)


def load_results(file_to_load: str):

    loaded = np.load(file_to_load)
    dict_version = {x: loaded[x] for x in loaded.keys()}
    dict_version['kernel'] = str(dict_version['kernel'])
    return MOGPResult(**dict_version)

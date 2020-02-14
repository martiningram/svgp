# Poisson point process SOGP
# This module contains some convenience code to fit SOGP models to
# presence-only data.
from svgp.tf.experimental.inducing_point_gp import \
    InducingPointGPSpecification, calculate_kl, initialise_using_kernel_fun, \
    project_to_x
from ml_tools.tf_kernels import matern_kernel_32, bias_kernel
from functools import partial
import numpy as np
import tensorflow as tf
from ml_tools.gp import find_starting_z
from svgp.tf.likelihoods import \
    ppm_likelihood_quadrature_approx, ppm_likelihood_berman_turner
from svgp.tf.quadrature import expectation
from ml_tools.flattening import (flatten_and_summarise_tf, reconstruct_tf,
                                 reconstruct_np)
from scipy.optimize import minimize
from typing import Dict, Optional
import tensorflow_probability as tfp
from os.path import join
from os import makedirs

STEP = 0


def get_kernel_fun_covs_only(lengthscales, alpha, intercept_sd):

    matern_fun = partial(matern_kernel_32, lengthscales=tf.exp(lengthscales),
                         alpha=tf.exp(alpha))

    intercept_fun = partial(bias_kernel, sd=tf.exp(intercept_sd))

    def compute(x1, x2, diag_only=False):

        matern_result = matern_fun(x1, x2, diag_only=diag_only)
        intercept_result = intercept_fun(x1, x2, diag_only=diag_only)

        return matern_result + intercept_result

    return compute


def create_spec(theta):

    kernel_fun = get_kernel_fun_covs_only(
        theta['lscales'], theta['alpha'], theta['intercept_sd'])

    cur_spec = InducingPointGPSpecification(
        L_elts=theta['L_elts'],
        mu=theta['mu'],
        kernel_fun=kernel_fun,
        Z=theta['Z']
    )

    return cur_spec


def calculate_objective(X: tf.Tensor, z: tf.Tensor, weights: tf.Tensor,
                        gp_spec: InducingPointGPSpecification,
                        use_berman_turner=True):

    proj_mean, proj_var = project_to_x(gp_spec, X)

    if use_berman_turner:
        curried_lik_fun = partial(ppm_likelihood_berman_turner,
                                  weights=weights)
    else:
        curried_lik_fun = partial(ppm_likelihood_quadrature_approx,
                                  weights=weights)

    expected_lik = expectation(z, proj_var, proj_mean, curried_lik_fun)
    kl = calculate_kl(gp_spec)

    return expected_lik - kl


def fit(X: np.ndarray, z: np.ndarray, weights: np.ndarray, n_inducing: int,
        fit_inducing_using_presences_only: bool = True, verbose: bool = True,
        log_theta_dir: Optional[str] = None):
    # TODO: Perhaps allow a separate kernel to be placed on bias
    # In general, it would be nice to have more flexibility in how to set the
    # kernel... But keep it simple for now.
    # X is assumed to be scaled.
    # Also, might be nice to easily be able to switch between minibatching and
    # full batch. But for now, just do full batch.

    global STEP
    STEP = 0

    n_cov = X.shape[1]

    start_alpha = np.log(1.)
    start_lscales = np.log(np.random.uniform(2, 5, size=n_cov))
    start_intercept_sd = np.log(1.)
    start_kernel_fun = get_kernel_fun_covs_only(
        tf.constant(start_lscales.astype(np.float32)),
        tf.constant(np.array(start_alpha).astype(np.float32)),
        tf.constant(np.array(start_intercept_sd).astype(np.float32)))

    if fit_inducing_using_presences_only:
        X_to_cluster = X[z > 0, :]
    else:
        X_to_cluster = X

    init_Z = find_starting_z(X_to_cluster, n_inducing).astype(np.float32)

    # Initialise the GP
    init_spec = initialise_using_kernel_fun(start_kernel_fun, init_Z)

    start_theta = {
        'Z': init_Z,
        'lscales': start_lscales,
        'intercept_sd': start_intercept_sd,
        'alpha': start_alpha,
        'mu': init_spec.mu,
        'L_elts': init_spec.L_elts
        }

    # Make sure they're all the right type
    start_theta = {x: np.array(y).astype(np.float64) for x, y in
                   start_theta.items()}

    # Prepare the tensors
    X = tf.cast(tf.constant(X), tf.float32)
    z = tf.cast(tf.constant(z), tf.float32)
    weights = tf.cast(tf.constant(weights), tf.float32)

    flat_theta, summary = flatten_and_summarise_tf(**start_theta)

    # TODO: Can / should I abstract away the creation of this function?
    def to_optimise(flat_theta):

        global STEP

        flat_theta = tf.cast(tf.constant(flat_theta), tf.float32)

        with tf.GradientTape() as tape:

            tape.watch(flat_theta)

            theta = reconstruct_tf(flat_theta, summary)

            cur_spec = create_spec(theta)

            cur_objective = -calculate_objective(X, z, weights, cur_spec)

            # Add a prior on the lengthscales
            lscale_prior = tf.reduce_sum(
                tfp.distributions.Gamma(
                    3, 1 / 3).log_prob(tf.exp(theta['lscales'])))

            cur_objective = cur_objective - lscale_prior

            cur_grad = tape.gradient(cur_objective, flat_theta)

            if log_theta_dir is not None:
                makedirs(log_theta_dir, exist_ok=True)
                grads = reconstruct_np(cur_grad.numpy(), summary)
                theta = reconstruct_np(flat_theta.numpy(), summary)
                np.savez(join(log_theta_dir, f'grads_{STEP}'),
                         **grads, objective=cur_objective.numpy(),
                         step=STEP)
                np.savez(join(log_theta_dir, f'theta_{STEP}'),
                         **theta, objective=cur_objective.numpy(),
                         step=STEP)

            STEP += 1

            if verbose:
                print(cur_objective, np.linalg.norm(cur_grad.numpy()))

        return (cur_objective.numpy().astype(np.float64),
                cur_grad.numpy().astype(np.float64))

    result = minimize(to_optimise, flat_theta.numpy().astype(np.float64),
                      method='L-BFGS-B', jac=True)

    final_flat_theta = result.x.astype(np.float32)

    return reconstruct_tf(final_flat_theta, summary)


def predict(fit_results: Dict[str, tf.Tensor], X: np.ndarray) -> np.ndarray:

    spec = create_spec(fit_results)

    pred_mean, pred_var = project_to_x(spec, X.astype(np.float32))

    return pred_mean.numpy(), pred_var.numpy()


def save_results(fit_results: Dict[str, tf.Tensor], save_path: str):

    np_version = {x: y.numpy() for x, y in fit_results.items()}

    np.savez(save_path, **np_version)

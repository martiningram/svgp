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
from typing import Optional
import tensorflow_probability as tfp
from os.path import join
from os import makedirs
from svgp.tf.experimental.kernel_spec import (
    KernelSpec, update_parameters, get_kernel_fun, calculate_prior_prob)
import dill
from ml_tools.minibatching import optimise_minibatching
from ml_tools.adam import initialise_state, adam_step

STEP = 0


def get_default_kernel_spec(n_covariates):

    def base_kernel_fun(x1, x2, alpha, lengthscales, intercept_sd,
                        diag_only=False):

        matern_fun = partial(matern_kernel_32, lengthscales=lengthscales,
                             alpha=alpha, diag_only=diag_only)

        intercept_fun = partial(bias_kernel, sd=intercept_sd,
                                diag_only=diag_only)

        return matern_fun(x1, x2) + intercept_fun(x1, x2)

    init_params = {
        'lengthscales': np.log(np.random.uniform(2., 5., size=n_covariates)),
        'alpha': np.array(0.),
        'intercept_sd': np.array(0.)
    }

    init_params = {x: tf.constant(y.astype(np.float32)) for x, y in
                   init_params.items()}

    constraints = {x: '+' for x in init_params}

    priors = {
        'lengthscales': lambda x: tf.reduce_sum(
            tfp.distributions.Gamma(3, 1/3).log_prob(x))
    }

    return KernelSpec(
        base_kernel_fun=base_kernel_fun,
        parameters=init_params,
        constraints=constraints,
        priors=priors
    )


def get_thinned_kernel_spec(n_covariates, thinning_indices):

    def main_kernel(x1, x2, cov_alpha, cov_lengthscales, intercept_sd,
                    diag_only=False):

        matern_fun = partial(matern_kernel_32, lengthscales=cov_lengthscales,
                             alpha=cov_alpha, diag_only=diag_only)

        intercept_fun = partial(bias_kernel, sd=intercept_sd,
                                diag_only=diag_only)

        return matern_fun(x1, x2) + intercept_fun(x1, x2)

    def thin_kernel(x1, x2, thin_alpha, thin_lengthscales, diag_only=False):

        return matern_kernel_32(x1, x2, alpha=thin_alpha,
                                lengthscales=thin_lengthscales,
                                diag_only=diag_only)

    def combined_kernel(x1, x2, cov_alpha, cov_lengthscales, intercept_sd,
                        thin_alpha, thin_lengthscales, diag_only=False):

        cov_indices = np.setdiff1d(np.arange(n_covariates), thinning_indices)

        return (main_kernel(tf.gather(x1, cov_indices, axis=1),
                            tf.gather(x2, cov_indices, axis=1), cov_alpha,
                            cov_lengthscales, intercept_sd,
                            diag_only=diag_only)
                + thin_kernel(tf.gather(x1, thinning_indices, axis=1),
                              tf.gather(x2, thinning_indices, axis=1),
                              thin_alpha, thin_lengthscales,
                              diag_only=diag_only))

    n_thin_cov = len(thinning_indices)
    n_main_cov = n_covariates - n_thin_cov

    init_params = {
        'cov_lengthscales': np.log(
            np.random.uniform(2., 5., size=n_main_cov)),
        'thin_lengthscales': np.log(
            np.random.uniform(2., 5., size=n_thin_cov)),
        'cov_alpha': np.array(0.),
        'thin_alpha': np.array(0.),
        'intercept_sd': np.array(0.)
    }

    init_params = {x: tf.constant(y.astype(np.float32)) for x, y in
                   init_params.items()}

    constraints = {x: '+' for x in init_params}

    priors = {
        'cov_lengthscales': lambda x: tf.reduce_sum(
            tfp.distributions.Gamma(3, 1/3).log_prob(x)),
        'thin_lengthscales': lambda x: tf.reduce_sum(
            tfp.distributions.Gamma(3, 1/3).log_prob(x))
    }

    return KernelSpec(
        base_kernel_fun=combined_kernel,
        parameters=init_params,
        constraints=constraints,
        priors=priors
    )


def update_specs(theta, kernel_spec):

    kernel_spec = update_parameters(kernel_spec, theta)
    kernel_fun = get_kernel_fun(kernel_spec)

    gp_spec = InducingPointGPSpecification(
        L_elts=theta['L_elts'],
        mu=theta['mu'],
        kernel_fun=kernel_fun,
        Z=theta['Z']
    )

    return kernel_spec, gp_spec


def calculate_objective(X: tf.Tensor, z: tf.Tensor, weights: tf.Tensor,
                        gp_spec: InducingPointGPSpecification,
                        use_berman_turner=True, likelihood_scale_factor=1.):

    proj_mean, proj_var = project_to_x(gp_spec, X)

    if use_berman_turner:
        curried_lik_fun = partial(ppm_likelihood_berman_turner,
                                  weights=weights)
    else:
        curried_lik_fun = partial(ppm_likelihood_quadrature_approx,
                                  weights=weights)

    expected_lik = expectation(z, proj_var, proj_mean, curried_lik_fun)
    kl = calculate_kl(gp_spec)

    return likelihood_scale_factor * expected_lik - kl


def initialise_theta(n_cov, thinning_indices, init_Z):

    if len(thinning_indices) == 0:
        # We have an "un-thinned" point process. We need only consider the
        # covariates.
        init_kernel_spec = get_default_kernel_spec(n_cov)
    else:
        # Include thinning
        init_kernel_spec = get_thinned_kernel_spec(n_cov, thinning_indices)

    start_kernel_fun = get_kernel_fun(init_kernel_spec)

    # Initialise the GP
    init_spec = initialise_using_kernel_fun(start_kernel_fun, init_Z)

    start_theta = {
        'Z': init_Z,
        'mu': init_spec.mu,
        'L_elts': init_spec.L_elts
    }

    # Add the kernel parameters to the optimisation
    start_theta.update(init_kernel_spec.parameters)

    # Make sure they're all the right type
    start_theta = {x: np.array(y).astype(np.float32) for x, y in
                   start_theta.items()}

    return start_theta, init_kernel_spec


def to_optimise(flat_theta, X, z, weights, use_berman_turner, summary,
                init_kernel_spec, log_theta_dir=None, verbose=True,
                likelihood_scale_factor=1.):

    global STEP

    flat_theta = tf.cast(tf.constant(flat_theta), tf.float32)

    with tf.GradientTape() as tape:

        tape.watch(flat_theta)

        theta = reconstruct_tf(flat_theta, summary)

        kernel_spec, gp_spec = update_specs(theta, init_kernel_spec)

        cur_objective = -calculate_objective(
            X, z, weights, gp_spec, use_berman_turner=use_berman_turner)

        kernel_prior_prob = calculate_prior_prob(kernel_spec)
        cur_objective = cur_objective - kernel_prior_prob

        cur_grad = tape.gradient(cur_objective, flat_theta)

        if log_theta_dir is not None:
            makedirs(log_theta_dir, exist_ok=True)
            grads = reconstruct_np(cur_grad.numpy(), summary)
            theta = reconstruct_np(flat_theta.numpy(), summary)
            np.savez(join(log_theta_dir, f'grads_{STEP}'), **grads,
                     objective=cur_objective.numpy(), step=STEP)
            np.savez(join(log_theta_dir, f'theta_{STEP}'), **theta,
                     objective=cur_objective.numpy(), step=STEP)

        STEP += 1

        if verbose:
            print(cur_objective, np.linalg.norm(cur_grad.numpy()))

    return (cur_objective.numpy().astype(np.float64),
            cur_grad.numpy().astype(np.float64))


def fit(X: np.ndarray, z: np.ndarray, weights: np.ndarray, n_inducing: int,
        thinning_indices: Optional[np.ndarray] = np.array([]),
        fit_inducing_using_presences_only: bool = False, verbose: bool = True,
        log_theta_dir: Optional[str] = None, use_berman_turner: bool = False):

    global STEP
    STEP = 0

    n_cov = X.shape[1]

    if fit_inducing_using_presences_only:
        X_to_cluster = X[z > 0, :]
    else:
        X_to_cluster = X

    init_Z = find_starting_z(X_to_cluster, n_inducing).astype(np.float32)

    start_theta, init_kernel_spec = initialise_theta(n_cov, thinning_indices,
                                                     init_Z)

    # Prepare the tensors
    X = tf.cast(tf.constant(X), tf.float32)
    z = tf.cast(tf.constant(z), tf.float32)
    weights = tf.cast(tf.constant(weights), tf.float32)

    flat_theta, summary = flatten_and_summarise_tf(**start_theta)

    opt_fun = partial(to_optimise, X=X, z=z, weights=weights,
                      use_berman_turner=use_berman_turner, summary=summary,
                      init_kernel_spec=init_kernel_spec,
                      log_theta_dir=log_theta_dir, verbose=verbose,
                      likelihood_scale_factor=1.)

    result = minimize(opt_fun, flat_theta.numpy().astype(np.float64),
                      method='L-BFGS-B', jac=True)

    final_flat_theta = result.x.astype(np.float32)
    final_theta = reconstruct_tf(final_flat_theta, summary)
    _, final_spec = update_specs(final_theta, init_kernel_spec)

    return final_spec


def fit_minibatching(X: np.ndarray,
                     z: np.ndarray,
                     weights: np.ndarray,
                     n_inducing: int,
                     thinning_indices: Optional[np.ndarray] = np.array([]),
                     fit_inducing_using_presences_only: bool = False,
                     verbose: bool = True,
                     log_theta_dir: Optional[str] = None,
                     use_berman_turner: bool = False,
                     batch_size: int = 1000,
                     learning_rate: float = 0.01,
                     n_steps: int = 1000,
                     sqrt_decay_learning_rate: bool = True):

    global STEP
    STEP = 0

    makedirs(log_theta_dir, exist_ok=True)

    n_cov = X.shape[1]

    if fit_inducing_using_presences_only:
        X_to_cluster = X[z > 0, :]
    else:
        X_to_cluster = X

    init_Z = find_starting_z(X_to_cluster, n_inducing).astype(np.float32)

    start_theta, init_kernel_spec = initialise_theta(n_cov, thinning_indices,
                                                     init_Z)

    flat_theta, summary = flatten_and_summarise_tf(**start_theta)

    data_dict = {
        'X': X,
        'z': z,
        'weights': weights
    }

    data_dict = {x: y.astype(np.float32) for x, y in data_dict.items()}

    if sqrt_decay_learning_rate:
        # Decay with sqrt of time
        step_size_fun = lambda t: learning_rate * (1 / np.sqrt(t))
    else:
        # Constant learning rate
        step_size_fun = lambda t: learning_rate

    opt_fun = partial(to_optimise, use_berman_turner=use_berman_turner,
                      summary=summary, init_kernel_spec=init_kernel_spec,
                      log_theta_dir=log_theta_dir, verbose=verbose,
                      likelihood_scale_factor=X.shape[0] / batch_size)

    adam_state = initialise_state(flat_theta.shape[0])

    adam_fun = partial(adam_step, step_size_fun=step_size_fun)

    result, loss_log = optimise_minibatching(
        data_dict, opt_fun, adam_fun, flat_theta, batch_size, n_steps,
        X.shape[0], join(log_theta_dir, 'loss.txt'), False, adam_state)

    final_flat_theta = result.numpy().astype(np.float32)
    final_theta = reconstruct_tf(final_flat_theta, summary)
    _, final_spec = update_specs(final_theta, init_kernel_spec)

    return final_spec


def predict(gp_spec: InducingPointGPSpecification,
            X: np.ndarray) -> np.ndarray:

    pred_mean, pred_var = project_to_x(
        gp_spec, tf.constant(X.astype(np.float32)))

    return pred_mean.numpy(), pred_var.numpy()


def save_results(fit_spec: InducingPointGPSpecification, save_path: str):

    with open(save_path, 'wb') as f:
        dill.dump(fit_spec, f)

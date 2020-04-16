import numpy as np
from svgp.tf.likelihoods import ppm_likelihood_quadrature_approx
from svgp.tf.analytic_expectations import \
    ppm_likelihood_berman_turner_expectation
from functools import partial
import tensorflow as tf
from svgp.tf.quadrature import expectation
from ml_tools.gp import find_starting_z
from ml_tools.tf_kernels import matern_kernel_32
from ml_tools.flattening import (
    flatten_and_summarise_tf, reconstruct_tf, reconstruct_np)
import tensorflow_probability as tfp
import os
from ml_tools.utils import create_path_with_variables
from ml_tools.minibatching import (
    optimise_minibatching, save_theta_and_grad_callback, loss_log_callback)
from svgp.tf.experimental.linear_mogp import (
    LinearMOGPSpec, calculate_kl, project_selected_to_x, project_to_x)
from svgp.tf.experimental.multi_inducing_point_gp import \
    MultiInducingPointGPSpecification, initialise_using_kernel_funs
from typing import Optional
from ml_tools.adam import adam_step, initialise_state
from typing import NamedTuple, Tuple, Any, Dict


class PPPMOGPSpec(NamedTuple):

    cov_mogp_spec: LinearMOGPSpec
    thin_mogp_spec: Optional[LinearMOGPSpec] = None


def get_kernel_funs(lengthscales, alphas):

    return [partial(matern_kernel_32, lengthscales=tf.exp(cur_lscales),
                    alpha=tf.exp(cur_alpha)) for cur_lscales, cur_alpha in
            zip(lengthscales, alphas)]


def calculate_objective(mogp_spec, X, sp_num, z, weights, lik_scale_factor,
                        use_berman_turner=True, thinning_mogp_spec=None,
                        X_thin=None):

    res_means, res_vars = project_selected_to_x(mogp_spec, X, sp_num)

    kl = calculate_kl(mogp_spec)

    if thinning_mogp_spec is not None:

        kl += calculate_kl(thinning_mogp_spec)

        thin_means, thin_vars = project_selected_to_x(
            thinning_mogp_spec, X_thin, sp_num)

        res_means += thin_means
        res_vars += thin_vars

    if use_berman_turner:
        lik = tf.reduce_sum(ppm_likelihood_berman_turner_expectation(
            res_means, res_vars, z, weights))
    else:
        likelihood = partial(ppm_likelihood_quadrature_approx, weights=weights)
        lik = expectation(z, res_vars, res_means, likelihood)

    return lik_scale_factor * lik - kl


def build_spec(theta):

    n_latent = theta['Zs'].shape[0]
    is_thinned = 'thin_lscales' in theta

    kernel_funs = get_kernel_funs(
        theta['lscales'], tf.tile([theta['log_cov_alpha']], (n_latent,)))

    # Build the environment GP
    gp_spec = MultiInducingPointGPSpecification(
        L_elts=theta['L_elts'],
        mus=theta['mus'],
        kernel_funs=kernel_funs,
        Zs=theta['Zs']
    )

    # Build the linear MOGP spec on the environment
    linear_gp_spec = LinearMOGPSpec(
        multi_gp=gp_spec,
        w_means=theta['w_means'],
        w_vars=tf.exp(theta['w_vars']),
        w_prior_mean=theta['w_prior_mean'],
        w_prior_var=tf.exp(theta['w_prior_var']),
        intercept_means=theta['intercept_means'],
        intercept_vars=tf.exp(theta['intercept_vars']),
        intercept_prior_mean=theta['intercept_prior_mean'],
        intercept_prior_var=tf.exp(theta['intercept_prior_var'])
    )

    if is_thinned:

        k_funs_thin = get_kernel_funs(
            theta['thin_lscales'], tf.tile([theta['log_thin_alpha']], (1,)))

        # Build and add the thinning GP
        thin_gp_spec = MultiInducingPointGPSpecification(
            L_elts=theta['thin_L_elts'],
            mus=theta['thin_mus'],
            kernel_funs=k_funs_thin,
            Zs=theta['thin_Zs']
        )

        thin_linear_spec = LinearMOGPSpec(
            multi_gp=thin_gp_spec,
            w_means=theta['thin_w_means'],
            w_vars=tf.exp(theta['thin_w_vars']),
            w_prior_mean=theta['thin_w_prior_mean'],
            w_prior_var=tf.exp(theta['thin_w_prior_var']),
        )

    else:

        thin_linear_spec = None

    return PPPMOGPSpec(cov_mogp_spec=linear_gp_spec,
                       thin_mogp_spec=thin_linear_spec)


def objective_and_grad(flat_theta, X, X_thin, sp_num, z, weights, summary,
                       n_latent, n_data, use_berman_turner, log_cov_alpha,
                       log_thin_alpha=0., thin_Zs=None):

    # TODO: Make priors configurable; add docstrings.
    # Note: if thin_Zs is passed, we are not optimising their locations.
    # This is not the cleanest way of doing it, but it's the best I can think
    # of for now.

    flat_theta = tf.constant(flat_theta.astype(np.float32))

    if X_thin is not None:

        X, X_thin, z, weights = map(
            lambda x: tf.cast(tf.constant(x), tf.float32),
            [X, X_thin, z, weights])

    else:

        X, z, weights = map(
            lambda x: tf.cast(tf.constant(x), tf.float32),
            [X, z, weights])

    with tf.GradientTape() as tape:

        tape.watch(flat_theta)

        theta = reconstruct_tf(flat_theta, summary)

        if thin_Zs is not None:
            theta['thin_Zs'] = thin_Zs

        # This is fixed during optimisation, so we're setting it here
        theta['log_cov_alpha'] = log_cov_alpha
        theta['log_thin_alpha'] = log_thin_alpha

        spec = build_spec(theta)

        # Fix prior mean and var to start with
        obj = -calculate_objective(
            spec.cov_mogp_spec, X, sp_num, z, weights,
            lik_scale_factor=n_data / X.shape[0],
            thinning_mogp_spec=spec.thin_mogp_spec, X_thin=X_thin,
            use_berman_turner=use_berman_turner)

        # Add prior on lengthscales
        obj = obj - tf.reduce_sum(
                tfp.distributions.Gamma(3, 1/3).log_prob(
                    tf.exp(theta['lscales'])))

        # TODO: Make these configurable
        # Add prior on prior w means and variances
        obj = obj - tf.reduce_sum(
            tfp.distributions.Normal(0., 1.).log_prob(
                theta['w_prior_mean']))
        obj = obj - tf.reduce_sum(
            tfp.distributions.Gamma(0.5, 0.5).log_prob(
                tf.exp(theta['w_prior_var'])))

        # Add prior on intercept mean and variance
        obj = obj - tf.reduce_sum(
            tfp.distributions.Normal(0., 1.).log_prob(
                theta['intercept_prior_mean']))
        obj = obj - tf.reduce_sum(
            tfp.distributions.Gamma(0.5, 0.5).log_prob(
                tf.exp(theta['intercept_prior_var'])))

        if X_thin is not None:
            obj = obj - tf.reduce_sum(
                tfp.distributions.Gamma(3, 1/3).log_prob(
                    tf.exp(theta['thin_lscales'])))
            obj = obj - tf.reduce_sum(
                tfp.distributions.Normal(0., 1.).log_prob(
                    theta['thin_w_prior_mean']))
            obj = obj - tf.reduce_sum(
                tfp.distributions.Gamma(0.5, 0.5).log_prob(
                    tf.exp(theta['thin_w_prior_var'])))

        grad = tape.gradient(obj, flat_theta)

    if np.any(np.isnan(grad.numpy())):
        # Save the current state for investigation
        np.savez('theta_bug', **{x: y.numpy() for x, y in theta.items()})
        np.savez('data_bug', **{
            'X': X.numpy(), 'X_thin': X_thin.numpy(), 'sp_num': sp_num,
            'z': z.numpy(), 'weights': weights.numpy(), 'n_latent': n_latent,
            'n_data': n_data, 'use_berman_turner': use_berman_turner,
            'thin_Zs': thin_Zs.numpy()})
        exit()

    return obj.numpy().astype(np.float64), grad.numpy().astype(np.float64)


def initialise_theta(Z, n_latent, n_cov, n_out, Z_thin=None, init_w_var=1.,
                     log_cov_alpha=0., log_thin_alpha=0.):

    start_lscales = np.log(np.random.uniform(
        1., 4., size=(n_latent, n_cov)).astype(np.float32))
    start_alphas = np.repeat(log_cov_alpha, n_latent).astype(np.float32)

    start_k_funs = get_kernel_funs(start_lscales, start_alphas)

    Zs = np.tile(Z, (n_latent, 1, 1)).astype(np.float32)

    start_gp = initialise_using_kernel_funs(start_k_funs, Zs)

    w_means = np.random.randn(n_out, n_latent) * 0.01
    w_vars = np.log(init_w_var * np.ones_like(w_means))

    start_theta = {
        'Zs': start_gp.Zs,
        'lscales': start_lscales,
        'w_means': w_means,
        'w_vars': w_vars,
        'L_elts': start_gp.L_elts,
        'mus': start_gp.mus,
        'intercept_means': np.zeros(n_out),
        'intercept_vars': np.log(np.ones(n_out)),
        'intercept_prior_var': np.log(np.array(1.)),
        'w_prior_var': np.log(
            np.repeat(init_w_var, n_latent).reshape(1, -1)),
        'w_prior_mean': np.repeat(0., n_latent).reshape(1, -1),
        'intercept_prior_mean': np.array(0.),
    }

    if Z_thin is not None:

        n_cov_thin = Z_thin.shape[1]

        # Only one shared function for now
        Z_thins = np.tile(Z_thin, (1, 1, 1))
        thin_lscales = np.log(np.random.uniform(2, 5, size=(1, n_cov_thin)))
        log_thin_alphas = np.array([log_thin_alpha])

        start_k_funs = get_kernel_funs(
            thin_lscales.astype(np.float32),
            log_thin_alphas.astype(np.float32))

        start_gp_thin = initialise_using_kernel_funs(
            start_k_funs, Z_thins.astype(np.float32))

        w_thin_means = np.random.randn(n_out, 1) * 0.01
        w_thin_vars = np.log(np.ones_like(w_thin_means))

        start_theta.update({
            'thin_Zs': Z_thins,
            'thin_lscales': thin_lscales,
            'thin_L_elts': start_gp_thin.L_elts,
            'thin_mus': start_gp_thin.mus,
            'thin_w_means': w_thin_means,
            'thin_w_vars': w_thin_vars,
            'thin_w_prior_mean': np.repeat(0., 1).reshape(1, -1),
            'thin_w_prior_var': np.log(np.repeat(1., 1.)).reshape(1, -1)
        })

    # Make sure they're the same type
    start_theta = {x: tf.cast(tf.constant(y), tf.float32) for x, y in
                   start_theta.items()}

    return start_theta


def fit(X: np.ndarray,
        z: np.ndarray,
        weights: np.ndarray,
        sp_num: np.ndarray,
        n_inducing: int,
        n_latent: int,
        log_folder: str,
        use_berman_turner: bool = True,
        X_thin: Optional[np.ndarray] = None,
        n_thin_inducing: Optional[int] = None,
        learning_rate: float = 0.01,
        steps: int = 100000,
        batch_size: int = 50000,
        save_opt_state: bool = False,
        save_every: Optional[int] = 1000,
        fix_thin_inducing: bool = False,
        cov_alpha: Optional[float] = None,
        thin_alpha: Optional[float] = 1.):

    n_cov = X.shape[1]
    n_data = X.shape[0]
    n_out = len(np.unique(sp_num))

    Z = find_starting_z(X[(z == 0) & (sp_num == np.unique(sp_num)[0])],
                        n_inducing)

    if X_thin is not None:
        # Make sure we were given how many thinning inducing to use
        assert n_thin_inducing is not None
        Z_thin = find_starting_z(X_thin[
            (z == 0) & (sp_num == np.unique(sp_num)[0])], n_thin_inducing)
    else:
        Z_thin = None

    log_cov_alpha = np.log(cov_alpha) if cov_alpha is not None else tf.cast(
        tf.constant(np.log(np.sqrt(2. / n_latent))), tf.float32)
    log_thin_alpha = np.log(thin_alpha)

    start_theta = initialise_theta(Z, n_latent, n_cov, n_out, Z_thin=Z_thin,
                                   log_cov_alpha=log_cov_alpha,
                                   log_thin_alpha=log_thin_alpha)

    if fix_thin_inducing:
        # Remove them from the theta dict of parameters to optimise
        start_theta = {x: y for x, y in start_theta.items() if x != 'thin_Zs'}

    flat_theta, summary = flatten_and_summarise_tf(**start_theta)

    log_folder = os.path.join(
        log_folder, create_path_with_variables(
            lr=learning_rate, batch_size=batch_size,
            steps=steps))

    os.makedirs(log_folder, exist_ok=True)

    opt_step_fun = partial(adam_step, step_size_fun=lambda t: learning_rate)
    opt_state = initialise_state(flat_theta.shape[0])

    flat_theta = flat_theta.numpy()

    to_optimise = partial(objective_and_grad, n_data=n_data, n_latent=n_latent,
                          summary=summary, use_berman_turner=use_berman_turner,
                          log_cov_alpha=log_cov_alpha)

    if fix_thin_inducing:

        to_optimise = partial(
            to_optimise, thin_Zs=tf.constant(
                np.expand_dims(Z_thin.astype(np.float32), axis=0)))

    full_data = {'X': X, 'sp_num': sp_num, 'z': z, 'weights': weights}

    log_file = os.path.join(log_folder, 'losses.txt')

    if X_thin is not None:
        full_data['X_thin'] = X_thin
    else:
        to_optimise = partial(to_optimise, X_thin=None)

    loss_log_file = open(log_file, 'w')

    additional_vars = {}

    if fix_thin_inducing:
        # Store thin Zs for callback to save
        additional_vars['thin_Zs'] = np.expand_dims(Z_thin, axis=0)

    additional_vars['log_cov_alpha'] = log_cov_alpha
    additional_vars['log_thin_alpha'] = log_thin_alpha

    def opt_callback(step: int, loss: float, theta: np.ndarray,
                     grad: np.ndarray, opt_state: Any):

        # Save theta and the gradients
        save_theta_and_grad_callback(step, loss, theta, grad, opt_state,
                                     log_folder, summary, save_every,
                                     additional_vars=additional_vars)

        # Log the loss
        loss_log_callback(step, loss, theta, grad, opt_state, loss_log_file)

    flat_theta, loss_log, _ = optimise_minibatching(
        full_data,
        to_optimise,
        opt_step_fun,
        opt_state,
        flat_theta,
        batch_size,
        steps,
        X.shape[0],
        callback=opt_callback
    )

    # Cast to float32
    flat_theta = flat_theta.astype(np.float32)

    final_theta = reconstruct_np(flat_theta, summary)

    if fix_thin_inducing:
        final_theta['thin_Zs'] = np.expand_dims(Z_thin, axis=0)

    final_theta['log_cov_alpha'] = log_cov_alpha

    return final_theta


def predict(spec: PPPMOGPSpec, X: np.ndarray,
            X_thin: Optional[np.ndarray] = None) -> \
        Tuple[np.ndarray, np.ndarray]:

    # TODO: Fix up the types
    env_means, env_vars = project_to_x(spec.cov_mogp_spec,
                                       X.astype(np.float32))

    if X_thin is not None:

        thin_means, thin_vars = project_to_x(spec.thin_mogp_spec, X_thin)

        env_means += thin_means
        env_vars += thin_vars

    return env_means.numpy(), env_vars.numpy()


def predict_selected(
        spec: PPPMOGPSpec, X: np.ndarray, sp_nums: np.ndarray,
        X_thin: Optional[np.ndarray] = None) -> \
        Tuple[np.ndarray, np.ndarray]:

    env_means, env_vars = project_selected_to_x(
        spec.cov_mogp_spec, X.astype(np.float32), sp_nums)

    if X_thin is not None:

        thin_means, thin_vars = project_selected_to_x(
            spec.thin_mogp_spec, X_thin, sp_nums)

        env_means += thin_means
        env_vars += thin_vars

    return env_means.numpy(), env_vars.numpy()


# TODO: Add predict_selected.
def save_results(theta: Dict[str, np.ndarray], target_file: str):

    np.savez(target_file, **theta)


def load_spec_from_array(saved_array: str) -> PPPMOGPSpec:

    loaded = np.load(saved_array)
    spec = build_spec(loaded)
    return spec

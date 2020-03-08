import numpy as np
from svgp.tf.likelihoods import (
    ppm_likelihood_quadrature_approx, ppm_likelihood_berman_turner)
from functools import partial
import tensorflow as tf
from svgp.tf.quadrature import expectation
from ml_tools.gp import find_starting_z
from ml_tools.tf_kernels import matern_kernel_32
from ml_tools.flattening import (flatten_and_summarise_tf, reconstruct_tf,
                                 reconstruct_np)
import tensorflow_probability as tfp
import os
from ml_tools.utils import create_path_with_variables
from ml_tools.minibatching import optimise_minibatching
from svgp.tf.experimental.linear_mogp import (
    LinearMOGPSpec, calculate_kl, project_selected_to_x)
from svgp.tf.experimental.multi_inducing_point_gp import \
    MultiInducingPointGPSpecification, initialise_using_kernel_funs
from typing import Optional
from ml_tools.adam import adam_step, initialise_state


def get_kernel_funs(lengthscales, alphas):

    return [partial(matern_kernel_32, lengthscales=tf.exp(cur_lscales),
                    alpha=tf.exp(cur_alpha)) for cur_lscales, cur_alpha in
            zip(lengthscales, alphas)]


def calculate_objective(mogp_spec, X, sp_num, z, weights, lik_scale_factor,
                        use_berman_turner=True, thinning_mogp_spec=None,
                        X_thin=None):

    res_means, res_vars = project_selected_to_x(mogp_spec, X, sp_num)

    if use_berman_turner:
        likelihood = partial(ppm_likelihood_berman_turner, weights=weights)
    else:
        likelihood = partial(ppm_likelihood_quadrature_approx, weights=weights)

    kl = calculate_kl(mogp_spec)

    if thinning_mogp_spec is not None:

        kl += calculate_kl(thinning_mogp_spec)

        thin_means, thin_vars = project_selected_to_x(
            thinning_mogp_spec, X_thin, sp_num)

        res_means += thin_means
        res_vars += thin_vars

    lik = expectation(z, res_vars, res_means, likelihood)

    return lik_scale_factor * lik - kl


def objective_and_grad(flat_theta, X, X_thin, sp_num, z, weights, summary,
                       n_latent, n_data, use_berman_turner):

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

        kernel_funs = get_kernel_funs(
            theta['lscales'], tf.tile([1.], (n_latent,)))

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

        if X_thin is not None:

            k_funs_thin = get_kernel_funs(
                theta['thin_lscales'], tf.tile([1.], (1,)))

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

        # Fix prior mean and var to start with
        obj = -calculate_objective(
            linear_gp_spec, X, sp_num, z, weights,
            lik_scale_factor=n_data / X.shape[0],
            thinning_mogp_spec=thin_linear_spec, X_thin=X_thin,
            use_berman_turner=use_berman_turner)

        # Add prior on lengthscales
        obj = obj - tf.reduce_sum(
                tfp.distributions.Gamma(3, 1/3).log_prob(
                    tf.exp(theta['lscales'])))

        if X_thin is not None:
            obj = obj - tf.reduce_sum(
                tfp.distributions.Gamma(3, 1/3).log_prob(
                    tf.exp(theta['thin_lscales'])))

        grad = tape.gradient(obj, flat_theta)

    return obj.numpy().astype(np.float64), grad.numpy().astype(np.float64)


def initialise_theta(Z, n_latent, n_cov, n_out, Z_thin=None, init_w_var=1.,
                     alpha=1.):

    start_lscales = np.log(np.random.uniform(
        1., 4., size=(n_latent, n_cov)).astype(np.float32))
    start_alphas = np.log(np.repeat(alpha, n_latent).astype(np.float32))

    start_k_funs = get_kernel_funs(start_lscales, start_alphas)

    Zs = np.tile(Z, (n_latent, 1, 1)).astype(np.float32)

    start_gp = initialise_using_kernel_funs(start_k_funs, Zs)

    w_means = np.random.randn(n_out, n_latent) * 0.01
    w_vars = np.log(init_w_var * np.ones_like(w_means) / n_latent)

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
            np.repeat(init_w_var / n_latent, n_latent).reshape(1, -1)),
        'w_prior_mean': np.repeat(0., n_latent).reshape(1, -1),
        'intercept_prior_mean': np.array(0.)
    }

    if Z_thin is not None:

        n_cov_thin = Z_thin.shape[1]

        # Only one shared function for now
        Z_thins = np.tile(Z_thin, (1, 1, 1))
        thin_lscales = np.log(np.random.uniform(2, 5, size=(1, n_cov_thin)))
        thin_alphas = np.log(np.array([1.]))

        start_k_funs = get_kernel_funs(
            thin_lscales.astype(np.float32), thin_alphas.astype(np.float32))

        start_gp_thin = initialise_using_kernel_funs(
            start_k_funs, Z_thins.astype(np.float32))

        w_thin_means = np.random.randn(n_out, 1) * 0.01
        w_thin_vars = np.log(np.ones_like(w_thin_means))

        start_theta.update({
            'thin_Zs': Z_thins,
            'thin_lscales': thin_lscales,
            'thin_alphas': thin_alphas,
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
        use_berman_turner: bool = True,
        X_thin: Optional[np.ndarray] = None,
        n_thin_inducing: Optional[int] = None,
        learning_rate: float = 0.01,
        steps_per_run: int = 1000,
        n_runs: int = 200,
        batch_size: int = 50000,
        log_folder: Optional[str] = None):

    n_cov = X.shape[1]
    n_data = X.shape[0]
    n_out = len(np.unique(sp_num))

    Z = find_starting_z(X[z > 0], n_inducing)

    if X_thin is not None:
        Z_thin = find_starting_z(X_thin[z > 0], n_thin_inducing)
    else:
        Z_thin = None

    start_theta = initialise_theta(Z, n_latent, n_cov, n_out, Z_thin=Z_thin)

    flat_theta, summary = flatten_and_summarise_tf(**start_theta)

    if log_folder is not None:

        log_folder = os.path.join(
            log_folder, create_path_with_variables(
                lr=learning_rate, batch_size=batch_size, n_runs=n_runs,
                steps_per_run=steps_per_run, upscaled_weights=True))

        os.makedirs(log_folder, exist_ok=True)

    opt_step_fun = partial(adam_step, step_size_fun=lambda t: learning_rate)
    opt_state = initialise_state(flat_theta.shape[0])

    flat_theta = flat_theta.numpy()

    to_optimise = partial(objective_and_grad, n_data=n_data, n_latent=n_latent,
                          summary=summary, use_berman_turner=use_berman_turner)

    full_data = {'X': X, 'sp_num': sp_num, 'z': z, 'weights': weights}

    log_file = (os.path.join(log_folder, 'losses.txt')
                if log_folder is not None
                else None)

    if X_thin is not None:
        full_data['X_thin'] = X_thin
    else:
        to_optimise = partial(to_optimise, X_thin=None)

    for i in range(n_runs):

        flat_theta, loss_log = optimise_minibatching(
            full_data,
            to_optimise,
            opt_step_fun,
            flat_theta,
            batch_size,
            steps_per_run,
            X.shape[0],
            log_file=log_file,
            append_to_log_file=i != 0,
            opt_state=opt_state)

        fit_results = reconstruct_np(flat_theta, summary)

        np.savez(os.path.join(
            log_folder, f'fit_results_{(i + 1) * 1000}'),
                 **fit_results)

    return reconstruct_tf(flat_theta, summary)

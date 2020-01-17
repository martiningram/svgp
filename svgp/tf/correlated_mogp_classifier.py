import numpy as np
import tensorflow as tf
from typing import NamedTuple
from .mogp_classifier import get_kernel_funs
from svgp.tf.utils import get_initial_values_from_kernel
from .sogp_classifier import kern_lookup
from ml_tools.gp import find_starting_z
import svgp.tf.mogp_correlated_weights as corr_mogp
import svgp.tf.mogp as mogp
from ml_tools.tensorflow import lo_tri_from_elements, rep_vector
from ml_tools.flattening import flatten_and_summarise_tf, reconstruct_tf
from .config import JITTER
from .likelihoods import bernoulli_probit_lik
from scipy.optimize import minimize

# TODO: There is lots of duplication with mogp_classifier here. Maybe I can do
# better.


class CorrelatedMOGPResult(NamedTuple):

    Ls: np.ndarray
    mu: np.ndarray
    kernel: str
    lengthscales: np.ndarray
    w_means: np.ndarray
    w_cov: np.ndarray
    Z: np.ndarray
    w_prior_means: np.ndarray
    w_prior_cov: np.ndarray


def create_pos_def_mat_from_elts_batch(elements, mat_size, n_mats, jitter=JITTER):

    ls = mogp.create_ls(elements, mat_size, n_mats)
    pos_def = ls @ tf.transpose(ls, (0, 2, 1))
    pos_def = pos_def + tf.expand_dims(tf.eye(mat_size) * jitter, axis=0)

    return pos_def


def create_pos_def_mat_from_elts(elements, mat_size, jitter=JITTER):

    ls = lo_tri_from_elements(elements, mat_size)
    pos_def = ls @ tf.transpose(ls)
    pos_def = pos_def + tf.eye(mat_size) * jitter

    return pos_def


def fit(X: np.ndarray,
        y: np.ndarray,
        n_inducing: int = 100,
        n_latent: int = 10,
        kernel: str = 'matern_3/2',
        random_seed: int = 2):

    # TODO: This is copied from the mogp_classifier.
    # Maybe instead make it a function of some sort?
    np.random.seed(random_seed)

    # Note that input _must_ be scaled. Some way to enforce that?
    kernel_fun = kern_lookup[kernel]

    n_cov = X.shape[1]
    n_out = y.shape[1]

    # Set initial values
    start_lengthscales = np.random.uniform(2., 4., size=(n_latent, n_cov))

    Z = find_starting_z(X, n_inducing)
    Z = np.tile(Z, (n_latent, 1, 1))

    start_kernel_funs = get_kernel_funs(kernel_fun,
                                        np.sqrt(start_lengthscales))

    init_Ls = np.stack([
        get_initial_values_from_kernel(cur_z, cur_kernel_fun)
        for cur_z, cur_kernel_fun in zip(Z, start_kernel_funs)
    ])

    init_ms = np.zeros((n_latent, n_inducing))

    start_prior_cov = np.eye(n_latent)
    start_prior_mean = np.zeros(n_latent)
    start_prior_cov_elts = corr_mogp.get_initial_w_elements(
        start_prior_mean, start_prior_cov, n_out)

    start_w_cov_elts = rep_vector(start_prior_cov_elts, n_out)

    init_w_means = np.random.randn(n_out, n_latent)

    start_theta = {
            'mu': init_ms,
            'L_elts': init_Ls,
            'w_means': init_w_means,
            'w_cov_elts': start_w_cov_elts,
            'lengthscales': start_lengthscales,
            'w_prior_cov_elts': start_prior_cov_elts,
            'w_prior_mean': start_prior_mean,
            'Z': Z
    }

    flat_start_theta, summary = flatten_and_summarise_tf(**start_theta)

    X_tf = tf.constant(X.astype(np.float32))
    y_tf = tf.constant(y.astype(np.float32))

    def extract_cov_matrices(theta):

        w_covs = create_pos_def_mat_from_elts_batch(
            theta['w_cov_elts'], n_latent, n_out, jitter=JITTER)

        Ls = mogp.create_ls(theta['L_elts'], n_inducing, n_latent)

        w_prior_cov = create_pos_def_mat_from_elts(
            theta['w_prior_cov_elts'], n_latent, jitter=JITTER)

        return w_covs, Ls, w_prior_cov

    def calculate_objective(theta):

        w_covs, Ls, w_prior_cov = extract_cov_matrices(theta)

        kernel_funs = get_kernel_funs(kernel_fun, theta['lengthscales']**2)

        cur_objective = corr_mogp.compute_default_objective(
            X_tf, y_tf, theta['Z'], theta['mu'], Ls, theta['w_means'],
            w_covs, kernel_funs, bernoulli_probit_lik,
            theta['w_prior_mean'], w_prior_cov
        )

        return cur_objective

    def to_minimize(flat_theta):

        flat_theta = tf.constant(flat_theta)
        flat_theta = tf.cast(flat_theta, tf.float32)

        with tf.GradientTape() as tape:

            tape.watch(flat_theta)

            theta = reconstruct_tf(flat_theta, summary)

            objective = -calculate_objective(theta)

            grad = tape.gradient(objective, flat_theta)

        print(objective, np.linalg.norm(grad.numpy()))

        return (objective.numpy().astype(np.float64),
                grad.numpy().astype(np.float64))

    result = minimize(to_minimize, flat_start_theta, jac=True,
                      method='L-BFGS-B', tol=1)

    final_theta = reconstruct_tf(result.x.astype(np.float32), summary)

    w_covs, Ls, w_prior_cov = extract_cov_matrices(final_theta)

    return CorrelatedMOGPResult(
        Ls=Ls,
        mu=final_theta['mu'].numpy(),
        kernel=kernel,
        lengthscales=final_theta['lengthscales'].numpy()**2,
        w_means=final_theta['w_means'].numpy(),
        w_cov=w_covs.numpy(),
        Z=final_theta['Z'].numpy(),
        w_prior_means=final_theta['w_prior_mean'].numpy(),
        w_prior_cov=w_prior_cov.numpy()
    )

import numpy as np
from ml_tools.normals import moments_of_linear_combination_rvs_batch
import tensorflow as tf
from .kl import mvn_kl
from .mogp import project_latents
from .quadrature import expectation
from .svgp import compute_kl_term as compute_single_kl_term


def calculate_moments(m_proj, var_proj, w_means, w_covs):
    # m_proj is [n_l x n]
    # var_proj is [n_l x n]
    # w_means is [n_out x n_l]
    # w_covs is [n_out x n_l x n_l]

    # First, we need to transpose m_proj and var_proj
    m_proj = tf.transpose(m_proj)
    var_proj = tf.transpose(var_proj)

    # First, convert var_proj to a diagonal covariance matrix
    cov_proj = tf.linalg.diag(var_proj)

    return moments_of_linear_combination_rvs_batch(
        m_proj, cov_proj, w_means, w_covs, einsum_fun=tf.einsum)


def get_initial_w_elements(prior_mean, prior_cov, n_out):
    # NOTE: This is a numpy function.

    # Do a cholesky on the prior
    prior_cov_chol = np.linalg.cholesky(prior_cov)

    # Extract the elements
    elts = np.tril_indices_from(prior_cov_chol)

    return prior_cov_chol[elts]


def compute_kl_term(ms, Ls, ks, Z, w_means, w_covs, w_prior_mean, w_prior_cov):

    # Compute the KL term for the latent GPs
    kls = tf.reduce_sum([compute_single_kl_term(cur_m, cur_l, cur_z, cur_k) for
                         cur_m, cur_l, cur_k, cur_z in zip(ms, Ls, ks, Z)])

    # Compute the KL term for the W values
    # TODO: I think this works -- check.
    batch_kl = tf.reduce_sum(mvn_kl(w_means, w_covs, w_prior_mean, w_prior_cov,
                                    is_batch=True))

    return kls + batch_kl


def compute_site_means_and_vars(X, Z, ms, Ls, ks, w_means, w_covs):

    m_proj, var_proj = project_latents(X, Z, ms, Ls, ks)
    m_out, var_out = calculate_moments(m_proj, var_proj, w_means, w_covs)

    return m_out, var_out


def compute_likelihood(y, m_out, var_out, log_lik_fun):

    return tf.reduce_sum(expectation(
        tf.reshape(y, (-1,)), tf.reshape(var_out, (-1,)),
        tf.reshape(m_out, (-1,)), log_lik_fun))


def compute_default_objective(X, y, Z, ms, Ls, w_means, w_covs, ks,
                              log_lik_fun, w_prior_mean, w_prior_cov):
    # TODO: It would be nice to make this more general somehow to be able to
    # add intercept terms etc.
    # Maybe it could take a function that alters the predicted means
    # and variances (due to an intercept), or something.

    m_out, var_out = compute_site_means_and_vars(X, Z, ms, Ls, ks, w_means,
                                                 w_covs)

    likelihood = compute_likelihood(y, m_out, var_out, log_lik_fun)

    kl = compute_kl_term(ms, Ls, ks, Z, w_means, w_covs, w_prior_mean,
                         w_prior_cov)

    total_objective = likelihood - kl

    return total_objective

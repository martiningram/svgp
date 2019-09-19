import numpy as np
import tensorflow as tf
from svgp.tf.quadrature import expectation, expectation_map
from svgp.tf.svgp import compute_qf_mean_cov, compute_kl_term
from svgp.tf.kl import normal_kl_1d
from .config import DTYPE


def create_ls(elements, n_inducing, n_latent):

    Ls = list()

    for i in range(n_latent):

        indices = np.array(np.tril_indices(n_inducing)).T
        L = tf.scatter_nd(indices, elements[i], (n_inducing, n_inducing))
        Ls.append(L)

    return tf.stack(Ls)


def project_latents(x, Z, ms, Ls, ks):

    # First trick will be to project all the latent stuff:
    # I'm doing a for loop here for now but ultimately we can do better things
    m_proj = list()
    cov_proj = list()

    for cur_m, cur_l, cur_k in zip(ms, Ls, ks):

        cur_mean, cur_cov = compute_qf_mean_cov(cur_l, cur_m, x, Z, cur_k)

        m_proj.append(cur_mean)
        cov_proj.append(cur_cov)

    m_proj = tf.stack(m_proj)
    cov_proj = tf.stack(cov_proj)

    return m_proj, cov_proj


def calculate_approximate_means_and_vars(m_proj, cov_proj, w_means, w_sds):

    M1 = tf.transpose(m_proj)
    M2 = w_means

    V1 = tf.transpose(tf.matrix_diag_part(cov_proj))
    V2 = w_sds**2

    means = M1 @ M2
    vars = M1**2 @ V2 + V1 @ M2**2 + V1 @ V2

    return means, vars


def compute_objective(x, y, Z, ms, Ls, w_means, w_vars, ks, log_lik_fun,
                      w_prior_mean, w_prior_var):

    m_proj, cov_proj = project_latents(x, Z, ms, Ls, ks)
    m_out, var_out = calculate_approximate_means_and_vars(
        m_proj, cov_proj, w_means, tf.sqrt(w_vars))

    # curried_exp = lambda inputs: expectation_custom(*inputs,
    #                                                 log_y_f=log_lik_fun)

    # log_liks = tf.map_fn(curried_exp, [tf.transpose(y), tf.transpose(var_out),
    #                                    tf.transpose(m_out)], dtype=DTYPE)

    log_liks = expectation_map(
        tf.reshape(y, (-1,)), tf.reshape(var_out, (-1,)),
        tf.reshape(m_out, (-1,)), log_lik_fun)

    total_log_lik = tf.reduce_sum(log_liks)

    kls = tf.reduce_sum([compute_kl_term(cur_m, cur_l, Z, cur_k) for cur_m,
                         cur_l, cur_k in zip(ms, Ls, ks)])

    kl_w = tf.reduce_sum(normal_kl_1d(w_means, w_vars, w_prior_mean,
                                      w_prior_var))

    total_objective = total_log_lik - kls - kl_w

    return total_objective

import numpy as np
import tensorflow as tf
from .quadrature import expectation
from .kl import mvn_kl
from .config import DTYPE, JITTER
from ml_tools.tensorflow import solve_via_cholesky


def project_to_f(kmm, knm, knn, m, L, diag_only=True):

    m = tf.reshape(m, (-1, 1))

    kmm_chol = tf.linalg.cholesky(kmm)

    mean = tf.matmul(knm, solve_via_cholesky(kmm_chol, m))

    S = tf.matmul(L, tf.transpose(L)) + \
        tf.eye(int(L.shape[0]), dtype=DTYPE) * JITTER

    V1 = solve_via_cholesky(kmm_chol, S - kmm)
    V2 = solve_via_cholesky(kmm_chol, tf.transpose(knm))

    if diag_only:

        cov = knn + tf.einsum('ik,kl,li->i', knm, V1, V2)

    else:

        cov = knn + tf.matmul(knm, tf.matmul(V1, V2))

    return tf.squeeze(mean), cov


def compute_qf_mean_cov(L, m, X, Z, kernel_fn, diag_only=True):
    """
    Computes q[f], the variational distribution on the function values for
    each data point X.
    """

    knm = kernel_fn(X, Z)
    kmm = kernel_fn(Z, Z)
    knn = kernel_fn(X, X, diag_only=diag_only)

    mean, cov = project_to_f(kmm, knm, knn, m, L, diag_only=diag_only)

    return mean, cov


def compute_expected_log_lik(mean, var, y_batch, log_lik_fn):

    individual_expectations = expectation(y_batch, var, mean, log_lik_fn)

    return tf.reduce_sum(individual_expectations)


def compute_kl_term(m, L, Z, kern_fn):

    # For the log lik, we need q(u) and p(u).
    p_u_mean = tf.zeros_like(m)
    p_u_cov = kern_fn(Z, Z)
    q_u_cov = tf.matmul(L, tf.transpose(L)) + \
        tf.eye(int(L.shape[0]), dtype=DTYPE) * JITTER
    q_u_mean = m

    kl = mvn_kl(q_u_mean, q_u_cov, p_u_mean, p_u_cov)

    return kl


def compute_objective(x, y, m, L, Z, log_lik_fn, kern_fn):

    mean, var = compute_qf_mean_cov(L, m, x, Z, kern_fn, diag_only=True)

    expected_log_lik = compute_expected_log_lik(mean, var, y, log_lik_fn)

    kl_term = compute_kl_term(m, L, Z, kern_fn)
    objective = expected_log_lik - kl_term

    return objective


def extract_params(theta, n_inducing, square_kern_params=True):

    # Get the parameters
    m = theta[:n_inducing]

    indices = np.array(np.tril_indices(n_inducing)).T
    num_indices = len(indices)
    L_elts = theta[n_inducing:n_inducing+num_indices]

    L = tf.scatter_nd(indices, L_elts, (n_inducing, n_inducing))

    kern_params = theta[n_inducing+num_indices:]

    if square_kern_params:
        kern_params = kern_params**2

    return m, L, kern_params

import jax.numpy as jnp
from jax.ops.scatter import index_update
import numpy as onp

from .quadrature import expectation
from .kl import mvn_kl
from jax import jit


@jit
def project_to_f(kmm, knm, knn, m, L):

    mean = knm @ jnp.linalg.solve(kmm, m)

    S = L @ L.T

    V1 = jnp.linalg.solve(kmm, S - kmm)
    V2 = jnp.linalg.solve(kmm, knm.T)

    cov = knn + knm @ V1 @ V2

    return mean, cov


def compute_qf_mean_cov(L, m, X, Z, kernel_fn):

    knm = kernel_fn(X, Z)
    kmm = kernel_fn(Z, Z)
    knn = kernel_fn(X, X)

    mean, cov = project_to_f(kmm, knm, knn, m, L)

    return mean, cov


def compute_expected_log_lik(mean, cov, y_batch, log_lik_fn):

    individual_expectations = expectation(
        y_batch, jnp.diag(cov), mean, log_lik_fn)

    return jnp.sum(individual_expectations)


def compute_kl_term(m, L, Z, kern_fn):

    # For the log lik, we need q(u) and p(u).
    p_u_mean = jnp.zeros_like(m)
    p_u_cov = kern_fn(Z, Z)
    q_u_cov = L @ L.T
    q_u_mean = m

    kl = mvn_kl(q_u_mean, q_u_cov, p_u_mean, p_u_cov)

    return kl


def compute_objective(x, y, m, L, Z, log_lik_fn, kern_fn):

    mean, cov = compute_qf_mean_cov(L, m, x, Z, kern_fn)

    expected_log_lik = compute_expected_log_lik(
        mean, cov, y, log_lik_fn)

    kl_term = compute_kl_term(m, L, Z, kern_fn)
    objective = expected_log_lik - kl_term

    return objective


def extract_params(theta, n_inducing, square_kern_params=True):

    # Get the parameters
    m = theta[:n_inducing]

    L = jnp.zeros((n_inducing, n_inducing))
    indices = jnp.tril_indices(L.shape[0])
    num_indices = indices[0].shape[0]

    L_elts = theta[n_inducing:n_inducing+num_indices]
    L = index_update(L, indices, L_elts)

    kern_params = theta[n_inducing+num_indices:]

    if square_kern_params:
        kern_params = kern_params**2

    return m, L, kern_params


def get_starting_m_and_l(n_inducing):

    m = onp.random.randn(n_inducing)
    L = jnp.zeros((n_inducing, n_inducing))

    indices = jnp.tril_indices(L.shape[0])
    num_indices = indices[0].shape[0]

    random_vals = onp.random.randn(num_indices)

    L = index_update(L, indices, random_vals)

    return m, L

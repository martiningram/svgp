import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.scipy.linalg import cho_factor, cho_solve
from ml_tools.jax import pos_def_mat_from_tri_elts, diag_elts_of_triple_matmul


def _evaluate_kernels(X, Z, kernel_fn, diag_only=False):

    knn = kernel_fn(X, X, diag_only=diag_only)
    kmm = kernel_fn(Z, Z)
    knm = kernel_fn(X, Z)

    return knn, kmm, knm


@partial(jit, static_argnums=(5))
def _project_to_f_given_kernels_old(knn, kmm, knm, m, S, diag_only):

    kmm_chol, lower = cho_factor(kmm)

    pred_mean = knm @ cho_solve((kmm_chol, lower), m)

    D = cho_solve((kmm_chol, lower), S) - jnp.eye(m.shape[0])
    B = (cho_solve((kmm_chol, lower), D.T)).T

    if diag_only:

        cov = knn + diag_elts_of_triple_matmul(knm, B, knm.T)

    else:

        cov = knn + knm @ B @ knm.T

    return jnp.squeeze(pred_mean), cov


def _project_to_f_given_kernels(knn, kmm, knm, m, S, diag_only=True):

    m = jnp.reshape(m, (-1, 1))

    mean = knm @ jnp.linalg.solve(kmm, m)

    A_t = jnp.linalg.solve(kmm, knm.T)

    brackets = S - kmm

    if diag_only:

        # Trick to compute the diagonal only of the matrix product
        cov = knn + diag_elts_of_triple_matmul(A_t.T, brackets, A_t)

    else:

        cov = knn + jnp.matmul(A_t.T, jnp.matmul(brackets, A_t))

    return jnp.squeeze(mean), cov


def project_to_f(X, Z, m, S, kernel_fn, diag_only=False):

    knn, kmm, knm = _evaluate_kernels(X, Z, kernel_fn, diag_only)

    pred_mean, pred_cov = _project_to_f_given_kernels(knn, kmm, knm, m, S, diag_only)

    return pred_mean, pred_cov


def project_to_f_given_L_elts(X, Z, m, L_elts, kernel_fn, diag_only=False):

    S = pos_def_mat_from_tri_elts(L_elts, Z.shape[0])

    return project_to_f(X, Z, m, S, kernel_fn, diag_only)

import numpy as np
import tensorflow as tf


def get_initial_values_from_kernel(inducing_pts, kernel_fun, lo_tri=True):

    kmm = kernel_fun(inducing_pts, inducing_pts)

    if lo_tri:

        L = np.linalg.cholesky(kmm)
        elts = np.tril_indices_from(L)
        return L[elts]

    else:

        return kmm.reshape(-1)


def compute_diagonal_matmul(M1, M2, M3):

    return tf.einsum('ik,kl,li->i', M1, M2, M3)

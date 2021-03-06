import numpy as np
import tensorflow as tf
from .config import N_QUAD, DTYPE


x_quad, w_quad = np.polynomial.hermite.hermgauss(N_QUAD)
x_quad_tf = tf.constant(x_quad, dtype=DTYPE)
w_quad_tf = tf.constant(w_quad, dtype=DTYPE)


def transform_x(x, sigma, mu):

    return tf.sqrt(tf.constant(2., dtype=DTYPE)) * sigma * x + mu


def expectation(ys, vars, means, log_y_f, sum_result=True):
    """
    Returns the individual expectations for each of the ys.

    Args:
        ys: Outcomes, of shape (n,).
        vars: Marginal variances, of shape(n,).
        means: Marginal means, of shape (n,).
        log_y_f: The log likelihood of y given f (a function).
        sum_result: If true, returns the sum of the individual expectations.
            Otherwise, one per data point.

    Returns:
        If sum_result=True, the summed result of computing each expected log
        likelihood; otherwise, [n,] of them, one for each data point.
    """

    x_to_eval = transform_x(
        tf.reshape(x_quad_tf, (-1, 1)), tf.sqrt(vars), means)

    multiplied = tf.reshape(
        w_quad_tf, (-1, 1)) * log_y_f(ys, x_to_eval)

    if sum_result:
        reduced = tf.reduce_sum(multiplied)
    else:
        reduced = multiplied

    return reduced / tf.sqrt(tf.constant(np.pi, dtype=DTYPE))

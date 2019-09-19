import numpy as np
import tensorflow as tf
from .config import N_QUAD, DTYPE


x_quad, w_quad = np.polynomial.hermite.hermgauss(N_QUAD)
x_quad_tf = tf.constant(x_quad, dtype=DTYPE)
w_quad_tf = tf.constant(w_quad, dtype=DTYPE)


def transform_x(x, sigma, mu):

    return tf.sqrt(tf.constant(2., dtype=DTYPE)) * sigma * x + mu


def expectation(ys, vars, means, log_y_f):
    """
    Returns the individual expectations for each of the ys.

    Args:
        ys: Outcomes, of shape (n,).
        vars: Marginal variances, of shape(n,).
        means: Marginal means, of shape (n,).
        log_y_f: The log likelihood of y given f (a function).

    Returns:
        The expected log likelihood for each of the yn.
    """

    x_to_eval = transform_x(
        tf.reshape(x_quad_tf, (-1, 1)), tf.sqrt(vars), means)

    multiplied = tf.reshape(
        w_quad_tf, (-1, 1)) * log_y_f(ys, x_to_eval)

    import ipdb; ipdb.set_trace()

    reduced = tf.reduce_sum(multiplied)

    return reduced / tf.sqrt(tf.constant(np.pi, dtype=DTYPE))

def expectation_map(ys, vars, means, log_y_f):
    """
    Returns the individual expectations for each of the ys.

    Args:
        ys: Outcomes, of shape (n,).
        vars: Marginal variances, of shape(n,).
        means: Marginal means, of shape (n,).
        log_y_f: The log likelihood of y given f (a function).

    Returns:
        The expected log likelihood for each of the yn.
    """

    def to_map(inputs):

        x, w = inputs

        x_to_eval = transform_x(x, tf.sqrt(vars), means)
        multiplied = w * log_y_f(ys, x_to_eval)
        return tf.reduce_sum(multiplied)

    multiplied = tf.map_fn(to_map, (x_quad_tf, w_quad_tf), dtype=DTYPE)
    reduced = tf.reduce_sum(multiplied)

    return reduced / tf.sqrt(tf.constant(np.pi, dtype=DTYPE))


@tf.custom_gradient
def expectation_custom(ys, vars, means, log_y_f):

    x_to_eval = transform_x(
        tf.reshape(x_quad_tf, (-1, 1)), tf.sqrt(vars), means)

    w_col = tf.reshape(w_quad_tf, (-1, 1))

    def grad(dy):

        with tf.GradientTape() as g:

            g.watch(x_to_eval)

            with tf.GradientTape() as gg:

                gg.watch(x_to_eval)
                lik_val = log_y_f(ys, x_to_eval)

            grad = gg.gradient(lik_val, x_to_eval)

        second_grad = g.gradient(grad, x_to_eval)

        grad_mean = tf.reduce_sum(w_col * grad)
        grad_var = 0.5 * tf.reduce_sum(w_col * second_grad)

        return None, grad_var, grad_mean

    lik_val = log_y_f(ys, x_to_eval)
    expectation_value = tf.reduce_sum(w_col * lik_val)

    return expectation_value, grad

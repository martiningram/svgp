import numpy as np
import tensorflow as tf


x_quad, w_quad = np.polynomial.hermite.hermgauss(100)
x_quad_tf = tf.constant(x_quad, dtype=tf.float64)
w_quad_tf = tf.constant(w_quad, dtype=tf.float64)


def transform_x(x, sigma, mu):

    return tf.sqrt(tf.constant(2., dtype=tf.float64)) * sigma * x + mu


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

    reduced = tf.reduce_sum(multiplied, axis=0)

    return reduced / tf.sqrt(tf.constant(np.pi, dtype=tf.float64))


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

        grad_mean = tf.reduce_sum(w_col * grad, axis=0)
        grad_var = 0.5 * tf.reduce_sum(w_col * grad, axis=0)

        return None, grad_var, grad_mean

    lik_val = log_y_f(ys, x_to_eval)
    expectation_value = tf.reduce_sum(w_col * lik_val, axis=0)

    return expectation_value, grad

import numpy as np
import tensorflow as tf


x_quad, w_quad = np.polynomial.hermite.hermgauss(100)
x_quad_tf = tf.constant(x_quad, dtype=tf.float64)
w_quad_tf = tf.constant(w_quad, dtype=tf.float64)


def transform_x(x, sigma, mu):

    return tf.sqrt(tf.constant(2., dtype=tf.float64)) * sigma * x + mu


def expectation(ys, vars, means, log_y_f):
    # Returns the individual expectations for each of the ys.

    x_to_eval = transform_x(
        tf.reshape(x_quad_tf, (-1, 1)), tf.sqrt(vars), means)

    multiplied = tf.reshape(
        w_quad_tf, (-1, 1)) * log_y_f(ys, x_to_eval)

    return tf.reduce_sum(multiplied, axis=0) / tf.sqrt(
        tf.constant(np.pi, dtype=tf.float64))

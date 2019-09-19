import tensorflow as tf
import tensorflow_probability as tfp
from .config import DTYPE


def bernoulli_probit_lik(y, f):

    mean = tf.constant(0., dtype=DTYPE)
    sd = tf.constant(1., dtype=DTYPE)

    norm = tfp.distributions.normal.Normal(mean, sd)

    return y * norm.log_cdf(f) + (1 - y) * norm.log_cdf(-f)

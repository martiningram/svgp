import tensorflow as tf
from .config import DTYPE


def mvn_kl(mu_0, sigma_0, mu_1, sigma_1):

    logdet_sigma_1 = tf.linalg.logdet(sigma_1)
    logdet_sigma_0 = tf.linalg.logdet(sigma_0)
    term_1 = 0.5 * (logdet_sigma_1 - logdet_sigma_0)

    # I wonder if there's a more efficient way?
    mu_outer = tf.einsum('i,j->ij', mu_0 - mu_1, mu_0 - mu_1)
    inside_term = mu_outer + sigma_0 - sigma_1
    solved = tf.linalg.solve(sigma_1, inside_term)
    term_2 = 0.5 * tf.linalg.trace(solved)

    return term_1 + term_2


def normal_kl_1d(mu1, var1, mu2, var2):

    sd1 = tf.sqrt(var1)
    sd2 = tf.sqrt(var2)

    log_term = tf.math.log(sd2) - tf.math.log(sd1)

    main_term = (var1 + (mu1 - mu2)**2) / (2 * var2)

    const_term = tf.constant(-0.5, dtype=DTYPE)

    return log_term + main_term + const_term

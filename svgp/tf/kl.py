import tensorflow as tf


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

import tensorflow as tf
import tensorflow_probability as tfp
from .config import DTYPE


def bernoulli_probit_lik(y, f):

    mean = tf.constant(0., dtype=DTYPE)
    sd = tf.constant(1., dtype=DTYPE)

    norm = tfp.distributions.normal.Normal(mean, sd)

    return y * norm.log_cdf(f) + (1 - y) * norm.log_cdf(-f)


def binomial_probit_lik(k, f, n):

    mean = tf.constant(0., dtype=DTYPE)
    sd = tf.constant(1., dtype=DTYPE)

    norm = tfp.distributions.normal.Normal(mean, sd)

    return k * norm.log_cdf(f) + (n - k) * norm.log_cdf(-f)


# def ordered_probit_lik(y, f, cut_points, sd, min_val=-20, fallback=False):
#     # TODO: Consider returning a fixed negative value when predictions are
#     # really off?
#
#     ymin = 0
#     ymax = len(cut_points)
#
#     def lower_function(f):
#         # For the lower edge, it's just a normal probit
#
#         dist = tfp.distributions.Normal(loc=f, scale=sd)
#
#         return dist.log_cdf(cut_points[0])
#
#     def upper_function(f):
#         # For the upper edge, it's the same but 1 - that
#
#         dist = tfp.distributions.Normal(loc=f, scale=sd)
#
#         return dist.log_cdf(-cut_points[-1])
#
#     def middle_function(f):
#
#         # For these middle functions it's a bit more complicated.
#         # First, find the appropriate upper and lower cut points.
#         upper_to_select = y
#         lower_to_select = y - 1
#
#         # HACK:
#         # Really this function should fail when y is not greater than 0.
#         # But because tensorflow's where evaluates each branch, I hack it
#         # here.
#          upper_to_select = tf.minimum(ymax - 1,
#                                       tf.maximum(1, upper_to_select))
#          lower_to_select = tf.minimum(ymax - 2,
#                                       tf.maximum(0, lower_to_select))
#
#         upper_cut_points = tf.gather(cut_points, upper_to_select)
#         lower_cut_points = tf.gather(cut_points, lower_to_select)
#
#         base_dist = tfp.distributions.Normal(loc=f, scale=sd)
#
#         if fallback:
#
#             upper_prob = base_dist.cdf(upper_cut_points)
#             lower_prob = base_dist.cdf(lower_cut_points)
#             log_diff = tf.math.log(upper_prob - lower_prob + 10**-6)
#
#         else:
#
#             upper_logcdf = base_dist.log_cdf(upper_cut_points)
#             lower_logcdf = base_dist.log_cdf(lower_cut_points)
#
#             stacked = tf.stack([upper_logcdf, lower_logcdf], axis=1)
#             weights = tf.stack([tf.ones_like(f), -tf.ones_like(f)], axis=1)
#
#             log_diff = tfp.math.reduce_weighted_logsumexp(stacked, weights,
#                                                           axis=1)
#
#         return log_diff
#
#     lower_result = lower_function(f)
#     upper_result = upper_function(f)
#     middle_result = middle_function(f)
#
#     # Do two where clauses here
#     result_1 = tf.where(tf.equal(y, ymin), lower_result, upper_result)
#     result_2 = tf.where(
#         tf.logical_and(tf.greater(y, ymin),
#                        tf.less(y, ymax)), middle_result, result_1)
#
#     return tf.maximum(result_2, min_val)


def ppm_likelihood_berman_turner(y, f, weights):
    """
    This uses the "Berman-Turner device" to write the PPM likelihood as a
    weighted Poisson likelihood.

    Args:
        y: 1 / w_i if entry i is an observed point, 0 otherwise.
        f: The [log] intensities at each point.
        weights: The quadrature weights.
    """

    return weights * (y * f - tf.exp(f))


def ppm_likelihood_quadrature_approx(y, f, weights):
    """
    This computes the PPM likelihood without including the observed points in
    the approximation of the integral.

    Args:
        y: 1 if observed point, 0 if quadrature point.
        f: The [log] intensities at each point.
        weights: The quadrature weights. Note that weights for the observed
            points must be included, but their value does not matter.
    """

    presence_contrib = y * f
    quad_contrib = (1 - y) * weights * tf.exp(f)

    return presence_contrib - quad_contrib

import numpy as np
import tensorflow_probability as tfp
from svgp.tf.likelihoods import ordered_probit_lik
from scipy.stats import norm


def test_ordered_probit_lik():

    # Make up some data
    cut_points = np.array([-0.2, 0.4, 0.9])
    lik_sd = 0.1

    predicted_f = np.random.randn(100) * 0.01
    labels = np.random.choice(len(cut_points) + 1, size=100)

    result = ordered_probit_lik(labels, predicted_f, cut_points, lik_sd)

    simple_computations = np.zeros_like(predicted_f)

    # Compare against a simple naive version:
    for i, (cur_label, cur_predicted) in enumerate(zip(labels, predicted_f)):

        if cur_label == 0:
            simple_computations[i] = norm.logcdf(cut_points[0],
                                                 loc=cur_predicted,
                                                 scale=lik_sd)
        elif cur_label == np.max(labels):
            simple_computations[i] = norm.logcdf(-cut_points[-1],
                                                 loc=cur_predicted,
                                                 scale=lik_sd)
        else:
            upper_cutpoint = cut_points[cur_label]
            lower_cutpoint = cut_points[cur_label - 1]

            upper_logcdf = norm.logcdf(upper_cutpoint, loc=cur_predicted,
                                       scale=lik_sd)
            lower_logcdf = norm.logcdf(lower_cutpoint, loc=cur_predicted,
                                       scale=lik_sd)

            weights = [1, -1]

            simple_computations[i] = tfp.math.reduce_weighted_logsumexp(
                [upper_logcdf, lower_logcdf], weights)

    assert np.allclose(simple_computations, result)

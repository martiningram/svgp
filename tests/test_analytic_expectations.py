from svgp.tf.analytic_expectations import \
    ppm_likelihood_berman_turner_expectation
from svgp.tf.quadrature import expectation
from svgp.tf.likelihoods import ppm_likelihood_berman_turner
from functools import partial
import numpy as np


def test_analytic_berman_against_quadrature():

    # Make up a data point
    w_i = 1.
    z_i = 1.
    mu_i = 0.5
    sigma_sq_i = 0.25

    analytic = ppm_likelihood_berman_turner_expectation(
        mu_i, sigma_sq_i, z_i, w_i)

    lik = partial(ppm_likelihood_berman_turner, weights=w_i)

    quad = expectation(z_i, sigma_sq_i, mu_i, lik)

    assert np.allclose(analytic, quad)

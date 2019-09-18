import numpy as np
from scipy.integrate import quad
from quadrature import log_y_f, expectation
from functools import partial
from jax import jit
from jax.scipy.stats import norm
import jax.numpy as jnp


def test_quadrature_against_scipy():
    # The idea here is to compare the Gauss-Hermite quadrature against a
    # library routine.

    ys = np.random.randint(0, 2, size=10)
    vars = np.random.randn(10)**2
    means = np.random.randn(10)

    expectations = expectation(ys, vars, means)

    @jit
    def to_quadrature(f, cur_y, cur_mean, cur_var):

        log_prob = log_y_f(cur_y, f)
        q = norm.pdf(f, cur_mean, jnp.sqrt(cur_var))

        return log_prob * q

    quad_res = np.zeros_like(means)

    for i, (cur_y, cur_mean, cur_var) in enumerate(
            zip(ys, means, vars)):

        quad_fun = partial(to_quadrature, cur_y=cur_y, cur_mean=cur_mean,
                           cur_var=cur_var)

        quad_res[i] = quad(quad_fun, -np.inf, np.inf)[0]

    assert np.allclose(quad_res, expectations)

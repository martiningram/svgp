import numpy as onp
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


def make_2d_quadrature_grid(x_quad, w_quad):

    grid = jnp.meshgrid(x_quad, x_quad)
    weights = jnp.outer(w_quad, w_quad).reshape(-1)
    stacked = jnp.stack([grid[0].reshape(-1), grid[1].reshape(-1)], axis=1)

    return stacked.T, weights


# TODO: Add config to specify number of quad points
x_quad, w_quad = onp.polynomial.hermite.hermgauss(15)
x_quad, w_quad = jnp.array(x_quad), jnp.array(w_quad)

# TODO: Add config to specify number of quad points
x_quad_2d, w_quad_2d = onp.polynomial.hermite.hermgauss(10)
x_quad_2d, w_quad_2d = jnp.array(x_quad_2d), jnp.array(w_quad_2d)
x_quad_2d, w_quad_2d = make_2d_quadrature_grid(x_quad_2d, w_quad_2d)


@jit
def transform_x(x, sigma, mu):

    return jnp.sqrt(2) * sigma * x + mu


@partial(jit, static_argnums=3)
def expectation(ys, vars, means, log_y_f):
    # Returns the individual expectations for each of the ys.

    x_to_eval = transform_x(jnp.reshape(x_quad, (-1, 1)), jnp.sqrt(vars), means)

    multiplied = jnp.reshape(w_quad, (-1, 1)) * log_y_f(ys, x_to_eval)

    return jnp.sum(multiplied, axis=0) / jnp.sqrt(jnp.pi)


@partial(jit, static_argnums=2)
def expectation_2d(means, covs, fun, *args):
    """Calculates the expectation under a bivariate normal distribution with
    mean given by means and covariances given by covs.

    Args:
    means: Means of the bivariate normals; shape should be [N x 2].
    covs: Covariances of the bivariate normals; shape should be [N x 2 x 2].
    fun: The function whose expectation to compute under the bivariate
    normal. Should take in a matrix of shape [N x 2] and return a vector of
    shape [N,].
    args: Other args to fun. Each must be an array with initial shape [N,] to
    iterate over.
    
    Returns:
    A vector of N expectations, one for each of the inputs.
    """

    single_exp = lambda means, covs, *args: single_expectation_2d(
        means, covs, fun, *args
    )

    results = vmap(single_exp)(means, covs, *args)

    return results


@partial(jit, static_argnums=2)
def single_expectation_2d(mean, cov, fun, *args):

    cov_chol = jnp.linalg.cholesky(cov)

    scaled = (cov_chol * jnp.sqrt(2)) @ x_quad_2d + mean.reshape(-1, 1)

    return jnp.sum(fun(scaled.T, *args) * w_quad_2d) / jnp.pi

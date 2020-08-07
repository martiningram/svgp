import jax.numpy as jnp


def ppm_likelihood_berman_turner_expectation(f_mu, f_var, z, w, sum_result=False):
    """
    This computes the expectation of the Berman-Turner device approximation to
    the inhomogeneous Poisson point process log likelihood, assuming that the
    prediction f is distributed as a Gaussian with mean f_mu and variance
    f_var.

    Args:
        f_mu: The predicted mean.
        f_var: The predicted variance.
        z: 1 / w_i if entry i is an observed point, 0 otherwise.
        w: The quadrature weights

    Returns:
        The expected log likelihood for each input point.
    """

    result = w * (z * f_mu - jnp.exp(f_mu + f_var / 2))

    if sum_result:
        return jnp.sum(result)
    else:
        return result


def expected_ppm_likelihood_quadrature_approx(y, weights, f_mean, f_var):

    presence_contrib = f_mean
    quad_contrib = weights * jnp.exp(f_mean + f_var / 2)

    # results = y * presence_contrib - (y < 1) * quad_contrib

    results = jnp.where(y > 0.0, y * presence_contrib, -quad_contrib)

    return results

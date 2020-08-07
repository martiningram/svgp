from jax.scipy.stats import norm
from jax import jit
import jax.numpy as jnp


def bernoulli_probit_lik(y, f):

    return y * norm.logcdf(f) + (1 - y) * norm.logcdf(-f)


@jit
def gaussian_lik(y, f, f_sd):

    return norm.logpdf(y, f, f_sd)


@jit
def square_cox_lik(y, f, weights):

    # Lambda
    lamb = f ** 2

    values = jnp.where(y > 0.0, y * jnp.log(lamb), -weights * lamb)

    return values

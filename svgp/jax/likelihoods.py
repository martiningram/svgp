from jax.scipy.stats import norm
from jax import jit


def bernoulli_probit_lik(y, f):

    return y * norm.logcdf(f) + (1 - y) * norm.logcdf(-f)


@jit
def gaussian_lik(y, f, f_sd):

    return norm.logpdf(y, f, f_sd)

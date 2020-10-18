from jax.scipy.stats import norm
from jax import jit
import jax.numpy as jnp
from jax.nn import log_sigmoid
from jax.scipy.special import logsumexp


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


@jit
def thinned_bernoulli_lik(m, f, logit_g):

    log_prob_obs_given_pres = log_sigmoid(logit_g)
    log_prob_not_obs_given_pres = log_sigmoid(-logit_g)

    log_prob_suitable = norm.logcdf(f)
    log_prob_not_suitable = norm.logcdf(-f)

    log_prob_missing = logsumexp(
        jnp.stack(
            [
                log_prob_suitable + log_prob_not_obs_given_pres.reshape(1, -1),
                log_prob_not_suitable,
            ],
            axis=2,
        ),
        axis=2,
    )

    log_prob_not_missing = log_prob_suitable + log_prob_obs_given_pres.reshape(1, -1)

    return m * log_prob_missing + (1 - m) * log_prob_not_missing

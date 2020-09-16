import jax.numpy as jnp
from jax import jit


@jit
def mvn_kl(mu_0, sigma_0, mu_1, sigma_1):

    logdet_sigma_1 = jnp.prod(jnp.array(jnp.linalg.slogdet(sigma_1)))
    logdet_sigma_0 = jnp.prod(jnp.array(jnp.linalg.slogdet(sigma_0)))
    term_1 = 0.5 * (logdet_sigma_1 - logdet_sigma_0)

    # I wonder if there's a more efficient way?
    mu_outer = jnp.outer(mu_0 - mu_1, mu_0 - mu_1)
    inside_term = mu_outer + sigma_0 - sigma_1
    solved = jnp.linalg.solve(sigma_1, inside_term)
    term_2 = 0.5 * jnp.trace(solved)

    return term_1 + term_2


@jit
def mvn_kl_alt(mu_1, cov_1, mu_2, cov_2):

    _, logdet_1 = jnp.linalg.slogdet(cov_1)
    _, logdet_2 = jnp.linalg.slogdet(cov_2)

    logdet_term = logdet_2 - logdet_1

    # There's a d term here but since it's constant w.r.t all parameters
    # I'll ignore it.

    # Could do something via Cholesky here, too
    solved = jnp.linalg.solve(cov_2, cov_1)
    tr_term = jnp.trace(solved)

    mean_term = (mu_2 - mu_1) @ (jnp.linalg.solve(cov_2, (mu_2 - mu_1)))

    return 0.5 * (logdet_term + tr_term + mean_term)


@jit
def normal_kl_1d(mu1, var1, mu2, var2):

    sd1 = jnp.sqrt(var1)
    sd2 = jnp.sqrt(var2)

    log_term = jnp.log(sd2) - jnp.log(sd1)
    main_term = (var1 + (mu1 - mu2) ** 2) / (2 * var2)

    # TODO: TF had a constant term here, but I'm dropping it. It shouldn't
    # matter for optimisation unless I'm missing something

    return log_term + main_term

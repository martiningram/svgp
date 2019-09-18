import numpy as onp
import jax.numpy as jnp
from jax import jit


x_quad, w_quad = onp.polynomial.hermite.hermgauss(100)


@jit
def transform_x(x, sigma, mu):

    return jnp.sqrt(2) * sigma * x + mu


def expectation(ys, vars, means, log_y_f):
    # Returns the individual expectations for each of the ys.

    x_to_eval = transform_x(jnp.reshape(x_quad, (-1, 1)), jnp.sqrt(vars),
                            means)

    multiplied = jnp.reshape(w_quad, (-1, 1)) * log_y_f(ys, x_to_eval)

    return jnp.sum(multiplied, axis=0) / jnp.sqrt(jnp.pi)

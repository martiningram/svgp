import jax.numpy as jnp
from ml_tools.constrain import apply_transformation
from functools import partial
from ml_tools.jax_kernels import matern_kernel_32, bias_kernel
import svgp.jax.helpers.svgp_spec as sv
from ml_tools.flattening import flatten_and_summarise, reconstruct
from svgp.jax.quadrature import expectation_1d
from jax.scipy.stats import gamma
from ml_tools.jax import convert_decorator
from jax import jit, value_and_grad
from svgp.jax.likelihoods import bernoulli_probit_lik
from scipy.optimize import minimize
from ml_tools.gp import find_starting_z


gamma_default_lscale_prior_fn = lambda params: jnp.sum(
    gamma.logpdf(params["lengthscales"], 3.0, scale=3.0)
)

constrain_positive = partial(
    apply_transformation, search_key="log_", transformation=jnp.exp, replace_with=""
)


def ard_kernel_currier(params, base_kernel=matern_kernel_32):
    """A kernel getter to be used with get_kernel_fun. Given a parameter_dict
    containing the entries "lengthscales" and "alpha", returns the base_kernel
    ready to evaluate."""

    curried_kernel_fun = lambda x1, x2, diag_only=False: base_kernel(
        x1, x2, params["lengthscales"], params["alpha"], diag_only
    )

    return curried_kernel_fun


def bias_kernel_currier(params):

    curried_kernel_fun = lambda x1, x2, diag_only=False: bias_kernel(
        x1, x2, params["bias_sd"], diag_only
    )

    return curried_kernel_fun


def add_kernel_funs(*kerns):

    return lambda x1, x2, diag_only=False: sum(
        [cur_kern(x1, x2, diag_only=diag_only) for cur_kern in kerns]
    )


def ard_plus_bias_kernel_currier(params, base_kernel=matern_kernel_32):

    curried_base = ard_kernel_currier(params, base_kernel)
    curried_bias = bias_kernel_currier(params)

    return add_kernel_funs(curried_base, curried_bias)


def get_kernel_fun(
    kernel_currier, parameter_dict, transformation_fun=constrain_positive
):
    """A helper function to standardise kernel construction.
    
    Args:
    kernel_currier: This is a function taking in parameters and returning a
    kernel function which takes two arguments -- x1 and x2 -- as well as an
    optional argument diag_only.
    parameter_dict: This is a dictionary containing the parameters required by
    base_kernel_fun, possibly untransformed.
    transformation_fun: This function will be applied to the parameter_dict,
    allowing each parameter to be transformed before being used by
    base_kernel_fun.
    
    Returns:
    The kernel function, ready to be evaluated.
    """

    # Transform the parameters
    params = transformation_fun(parameter_dict)
    curried_kernel_fun = kernel_currier(params)

    return curried_kernel_fun


def fit(
    X,
    init_kernel_params,
    kernel_currier,
    likelihood_fun,
    prior_fun,
    transformation_fun=constrain_positive,
    n_inducing=100,
    verbose=False,
    Z=None,
):

    if Z is None:
        Z = find_starting_z(X, n_inducing)

    init_kern_fn = get_kernel_fun(
        kernel_currier, init_kernel_params, transformation_fun
    )

    init_spec = sv.initialise_using_kernel_fun(init_kern_fn, Z)

    theta = {
        "mu": init_spec.m,
        "L_elts": init_spec.L_elts,
        "Z": jnp.array(Z),
        **init_kernel_params,
    }

    flat_theta, summary = flatten_and_summarise(**theta)

    def to_minimize(flat_theta):

        theta = reconstruct(flat_theta, summary, jnp.reshape)

        kern_fn = get_kernel_fun(kernel_currier, theta, transformation_fun)

        spec = sv.SVGPSpec(
            m=theta["mu"], L_elts=theta["L_elts"], Z=theta["Z"], kern_fn=kern_fn
        )

        pred_mean, pred_var = sv.project_to_x(spec, X)

        kl = sv.calculate_kl(spec)

        lik = jnp.sum(expectation_1d(likelihood_fun, pred_mean, pred_var))

        prior = prior_fun(transformation_fun(theta))

        return -(lik - kl + prior)

    with_grad = partial(convert_decorator, verbose=verbose)(
        jit(value_and_grad(to_minimize))
    )

    result = minimize(with_grad, flat_theta, method="L-BFGS-B", jac=True)

    final_theta = reconstruct(result.x, summary, jnp.reshape)

    kern = get_kernel_fun(kernel_currier, final_theta, transformation_fun)

    final_spec = sv.SVGPSpec(
        m=final_theta["mu"],
        L_elts=final_theta["L_elts"],
        Z=final_theta["Z"],
        kern_fn=kern,
    )

    return final_spec, transformation_fun(final_theta)


def fit_bernoulli_sogp(
    X,
    y,
    init_params,
    kernel_currier,
    prior_fun,
    transformation_fun=constrain_positive,
    n_inducing=100,
    verbose=False,
    Z=None,
):

    likelihood_fun = lambda f: bernoulli_probit_lik(y, f)

    return fit(
        X,
        init_params,
        kernel_currier,
        likelihood_fun,
        prior_fun,
        transformation_fun,
        n_inducing,
        verbose,
        Z,
    )

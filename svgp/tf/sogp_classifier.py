# This is a convenience module to quickly and easily fit SOGP models.
# It's a bit experimental!
from typing import NamedTuple, Tuple
import tensorflow as tf
import numpy as np
import ml_tools.tf_kernels as tfk
import tensorflow_probability as tfp
from scipy.optimize import minimize
from ml_tools.gp import find_starting_z
from ml_tools.tensorflow import lo_tri_from_elements
from svgp.tf.utils import get_initial_values_from_kernel
from svgp.tf.svgp import compute_qf_mean_cov
from svgp.tf.svgp import compute_objective
from svgp.tf.likelihoods import bernoulli_probit_lik
from ml_tools.flattening import flatten_and_summarise_tf, reconstruct_tf
from ml_tools.tensorflow import lo_tri_from_elements
from scipy.stats import norm


kern_lookup = {
    'matern_3/2': tfk.matern_kernel_32,
    'matern_1/2': tfk.matern_kernel_12,
    'rbf': tfk.ard_rbf_kernel
}


# TODO: Maybe enforce that this is immutable somehow?
class SOGPResult(NamedTuple):

    L_elts: np.ndarray
    mu: np.ndarray
    kernel: str
    lengthscales: np.ndarray
    alpha: np.float64
    bias_sd: np.float64
    Z: np.ndarray


def get_kernel_fun(base_kern_fun, alpha, lscales, bias_sd):

    return (
        lambda x1, x2, diag_only=False: base_kern_fun(
            x1, x2, lengthscales=lscales, alpha=alpha, diag_only=diag_only)
        + tfk.bias_kernel(x1, x2, sd=bias_sd, diag_only=diag_only))


def fit(X: np.ndarray,
        y: np.ndarray,
        n_inducing: int = 100,
        kernel: str = 'matern_3/2',
        kernel_variance_prior: Tuple[float, float] = (3 / 2, 3 / 2),
        kernel_lengthscale_prior: Tuple[float, float] = (3, 1 / 3),
        bias_variance_prior: Tuple[float, float] = (3 / 2, 3 / 2),
        random_seed: int = 2) \
        -> SOGPResult:

    np.random.seed(random_seed)

    assert kernel in ['matern_3/2', 'matern_1/2', 'rbf'], \
        'Only these three kernels are currently supported!'

    # Note that input _must_ be scaled. Some way to enforce that?

    kernel_fun = kern_lookup[kernel]

    n_cov = X.shape[1]

    # Set initial values
    start_alpha = np.array(1.)
    start_lengthscales = np.random.uniform(2., 4., size=n_cov)
    start_bias_sd = np.array(1.)

    Z = find_starting_z(X, n_inducing)

    start_kernel_fun = get_kernel_fun(
        kernel_fun, start_alpha, start_lengthscales, start_bias_sd)

    init_L = get_initial_values_from_kernel(Z, start_kernel_fun)
    init_mu = np.zeros(n_inducing)

    init_theta = {
        'L_elts': init_L,
        'mu': init_mu,
        'alpha': start_alpha,
        'lscales': np.sqrt(start_lengthscales),
        'Z': Z,
        'bias_sd': start_bias_sd
    }

    flat_theta, summary = flatten_and_summarise_tf(**init_theta)

    X = tf.constant(X.astype(np.float32))
    y = tf.constant(y.astype(np.float32))

    lscale_prior = tfp.distributions.Gamma(*kernel_lengthscale_prior)
    kernel_var_prior = tfp.distributions.Gamma(*kernel_variance_prior)
    bias_var_prior = tfp.distributions.Gamma(*bias_variance_prior)

    def to_minimize_with_grad(x):

        with tf.GradientTape() as tape:

            x_tf = tf.constant(x)
            x_tf = tf.cast(x_tf, tf.float32)

            tape.watch(x_tf)

            theta = reconstruct_tf(x_tf, summary)

            alpha, lscales, bias_sd = (
                theta['alpha']**2, theta['lscales']**2, theta['bias_sd']**2)

            L_cov = lo_tri_from_elements(theta['L_elts'], n_inducing)

            kern_fun = get_kernel_fun(kernel_fun, alpha, lscales, bias_sd)

            objective = -compute_objective(
                X, y, theta['mu'], L_cov, theta['Z'], bernoulli_probit_lik,
                kern_fun)

            objective = objective - (
                tf.reduce_sum(lscale_prior.log_prob(lscales))
                + kernel_var_prior.log_prob(alpha**2)
                + bias_var_prior.log_prob(bias_sd**2)
            )

            grad = tape.gradient(objective, x_tf)

        print(objective, np.linalg.norm(grad.numpy()))

        return (objective.numpy().astype(np.float64),
                grad.numpy().astype(np.float64))

    result = minimize(to_minimize_with_grad, flat_theta, jac=True,
                      method='L-BFGS-B')

    final_theta = reconstruct_tf(result.x, summary)
    final_theta = {x: y.numpy() for x, y in final_theta.items()}

    # Build the results
    fit_result = SOGPResult(
        L_elts=final_theta['L_elts'],
        mu=final_theta['mu'],
        kernel=kernel,
        lengthscales=final_theta['lscales']**2,
        alpha=final_theta['alpha']**2,
        bias_sd=final_theta['bias_sd']**2,
        Z=final_theta['Z'])

    return fit_result


def predict(fit_result: SOGPResult, X_new: np.ndarray):
    # TODO: Is there something I can do about the casts here?

    n_inducing = fit_result.mu.shape[0]

    L = lo_tri_from_elements(fit_result.L_elts.astype(np.float32), n_inducing)

    base_kern = kern_lookup[fit_result.kernel]

    k_fun = get_kernel_fun(base_kern,
                           fit_result.alpha.astype(np.float32),
                           fit_result.lengthscales.astype(np.float32),
                           fit_result.bias_sd.astype(np.float32))

    pred_mean, pred_var = compute_qf_mean_cov(
        L, fit_result.mu.astype(np.float32), X_new.astype(np.float32),
        fit_result.Z.astype(np.float32), k_fun)

    return pred_mean.numpy(), np.sqrt(pred_var)


def predict_probs(fit_result: SOGPResult, X_new: np.ndarray,
                  n_draws: int = 10000):

    n_data = X_new.shape[0]

    pred_mean, pred_var = predict(fit_result, X_new)
    pred_sd = np.sqrt(pred_var)
    draws = np.random.normal(pred_mean, pred_sd, size=(n_draws, n_data))
    probs = norm.cdf(draws)

    return np.mean(probs, axis=0)


def save_results(fit_result: SOGPResult, target_file: str):

    dict_version = fit_result._asdict()
    np.savez(target_file, **dict_version)


def load_results(file_to_load: str):

    loaded = np.load(file_to_load)
    dict_version = {x: loaded[x] for x in loaded.keys()}
    dict_version['kernel'] = str(dict_version['kernel'])
    return SOGPResult(**dict_version)

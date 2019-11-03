import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from sdm_ml.dataset import BBSDataset
from sklearn.preprocessing import StandardScaler
from ml_tools.lin_alg import num_triangular_elts
from sdm_ml.gp.utils import find_starting_z
from svgp.tf.mogp import create_ls, compute_objective
from functools import partial
from ml_tools.tf_kernels import (ard_rbf_kernel, bias_kernel, matern_kernel_32,
                                 matern_kernel_12)
from svgp.tf.likelihoods import bernoulli_probit_lik
from scipy.optimize import minimize
from svgp.tf.config import DTYPE, JITTER
from ml_tools.tensorflow import rep_matrix
from svgp.tf.utils import get_initial_values_from_kernel
from ml_tools.flattening import flatten_and_summarise, reconstruct_tf


def extract_parameters(theta, summaries, n_inducing, n_latent, same_z=False):
    # Picks out the parameters and constrains some to be positive

    theta_dict = reconstruct_tf(theta, summaries)

    ms = theta_dict['m']
    Ls = create_ls(theta_dict['L_elts'], n_inducing, n_latent)

    w_means = theta_dict['W_means']
    w_vars = theta_dict['W_sds']**2

    if same_z:
        Z = rep_matrix(theta_dict['Z'], n_latent)
    else:
        Z = theta_dict['Z']

    w_prior_means = theta_dict['W_prior_mean']
    w_prior_vars = theta_dict['W_prior_sd']**2

    print(w_prior_means)
    print(w_prior_vars)

    kern_params = theta_dict['kernel_params']**2

    intercept = tf.squeeze(theta_dict['intercept'])

    return (ms, Ls, w_means, w_vars, Z, kern_params, w_prior_means,
            w_prior_vars, intercept)


def create_ks_fixed_variance(flat_kern_params, kern_fun):
    # Fixed variance

    # 0.1 variance for all kernel components except the last
    alphas = [np.sqrt(0.1) for _ in range(n_latent - 1)]
    alphas.append(np.sqrt(0.1))

    ks = [partial(kern_fun, alpha=tf.constant(cur_alpha, dtype=DTYPE),
                  lengthscales=cur_params, jitter=JITTER) for cur_params,
          cur_alpha in zip(tf.reshape(flat_kern_params, (n_latent, -1)),
                           alphas)]

    print(np.round(tf.reshape(flat_kern_params, (n_latent, -1)).numpy(), 2))

    # ks.append(partial(bias_kernel, jitter=JITTER,
    #                   sd=tf.constant(0.1, dtype=DTYPE)))

    return ks


def create_ks(flat_kern_params):

    print_parameter_summary(flat_kern_params)

    rbf_params = flat_kern_params[:-1]
    bias_sd = flat_kern_params[-1]

    ks = [partial(ard_rbf_kernel, alpha=cur_params[0],
                  lengthscales=cur_params[1:], jitter=JITTER) for cur_params in
          tf.reshape(rbf_params, (n_latent - 1, -1))]

    ks.append(partial(bias_kernel, jitter=JITTER, sd=bias_sd))

    return ks


def print_parameter_summary(flat_kern_params):

    print(f'Bias variance is: {flat_kern_params[-1]}')

    rbf_params = tf.reshape(flat_kern_params[:-1], (n_latent - 1, -1))
    rbf_vars = rbf_params[:, 0]
    rbf_lscales = rbf_params[:, 1:]

    print(f'RBF variances are: {np.round(rbf_vars.numpy(), 2)}')

    print(np.round(rbf_lscales.numpy(), 2))


def initialise_covariance_entries(kernel_creation_fun, flat_kernel_params,
                                  start_z):

    init_kernels = kernel_creation_fun(tf.constant(kernel_params, dtype=DTYPE))
    start_cov_elts = list()

    for cur_kernel_fun in init_kernels:
        # Get the initial values
        cur_vals = get_initial_values_from_kernel(
            tf.constant(start_z, dtype=DTYPE), cur_kernel_fun, lo_tri=True)
        start_cov_elts.append(cur_vals)

    start_cov_elts = tf.stack(start_cov_elts, axis=0).numpy()

    return start_cov_elts


dataset = BBSDataset.init_using_env_variable()

cov_df = dataset.training_set.covariates
out_df = dataset.training_set.outcomes

test_run = True
same_z = False
kern_to_use = matern_kernel_32

np.random.seed(2)

if test_run:

    # Choose a subset of birds to start with before I work out memory fix
    bird_subset = np.random.choice(out_df.columns, size=16, replace=False)
    if 'Willet' not in bird_subset:
        bird_subset[0] = 'Willet'

else:

    bird_subset = out_df.columns

out_df = out_df[bird_subset]

assert 'Willet' in bird_subset

if test_run:

    site_subset = np.random.choice(len(cov_df.index), size=100, replace=False)
    cov_df = cov_df.iloc[site_subset]
    out_df = out_df.iloc[site_subset]

scaler = StandardScaler()

x = scaler.fit_transform(cov_df.values)
y = out_df.values

if test_run:

    n_inducing = 20
    n_latent = 4

else:

    n_inducing = 100
    n_latent = 12

n_cov = int(x.shape[1])

start_z = find_starting_z(x, n_inducing)

x = tf.constant(x, dtype=DTYPE)
y = tf.constant(y, dtype=DTYPE)

n_out = int(y.shape[1])

# kernel_vars = np.random.uniform(0.1, 0.4, size=n_latent - 1)
kernel_lscales = np.random.uniform(2., 4., size=(n_latent, n_cov))
# kernel_params = np.concatenate([kernel_vars.reshape(-1, 1), kernel_lscales],
#                                axis=1).reshape(-1)
kernel_params = kernel_lscales.reshape(-1)
# kernel_params = np.append(kernel_params, np.array([0.1]))

create_k_fun = partial(create_ks_fixed_variance, kern_fun=kern_to_use)

start_cov_elts = initialise_covariance_entries(
    create_k_fun, kernel_params, start_z)

if same_z:
    z_init = start_z
else:
    z_init = rep_matrix(start_z, n_latent).numpy()


start_theta_dict = {
    'm': np.zeros((n_latent, n_inducing)),
    'L_elts': start_cov_elts,
    'W_means': np.random.randn(n_latent, n_out) * 0.1,
    'W_sds': np.ones((n_latent, n_out)),
    'Z': z_init,
    'W_prior_mean': np.zeros((n_latent, 1)),
    'W_prior_sd': np.ones((n_latent, 1)),
    'kernel_params': kernel_params,
    'intercept': np.zeros(1)
}

start_theta, summaries = flatten_and_summarise(**start_theta_dict)

start_theta_tensor = tf.Variable(start_theta, dtype=DTYPE)


def to_minimize(theta):

    (ms, Ls, w_means, w_vars, Z, kern_params, w_prior_means, w_prior_vars,
     intercept) = extract_parameters(theta, summaries, n_inducing, n_latent,
                                     same_z=same_z)

    print(np.round(w_means.numpy(), 2))
    print(intercept)

    ks = create_k_fun(kern_params)

    return -compute_objective(
        x, y, Z, ms, Ls, w_means, w_vars, ks, bernoulli_probit_lik,
        w_prior_means, w_prior_vars, intercept) + tf.reduce_sum(
            w_prior_vars**2 / 2.) + tf.reduce_sum(w_prior_means**2 / 2.)


def to_minimize_with_grad(theta):

    theta_tensor = tf.Variable(theta, dtype=DTYPE)

    with tf.GradientTape() as tape:

        tape.watch(theta_tensor)

        result = to_minimize(theta_tensor)

    result_grad = tape.gradient(result, theta_tensor)

    print(result, flush=True)

    return result.numpy().astype(np.float64), result_grad.numpy().astype(
        np.float64)


result = minimize(to_minimize_with_grad, start_theta, jac=True,
                  method='L-BFGS-B')

final_params = result.x

(ms, Ls, w_means, w_vars, Z, kern_params, w_prior_means, w_prior_vars,
 intercept) = extract_parameters(final_params, summaries, n_inducing, n_latent)

np.savez('final_params_split_separate_fit_prior_var', ms=ms, Ls=Ls,
         w_means=w_means, w_vars=w_vars, kern_params=kern_params,
         n_inducing=n_inducing, n_latent=n_latent, birds=bird_subset, Z=Z,
         w_prior_vars=w_prior_vars, intercept=intercept,
         w_prior_means=w_prior_means)

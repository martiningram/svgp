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
import tensorflow_probability as tfp


def extract_parameters(theta, n_inducing, n_latent, n_out, n_cov,
                       same_z=False):

    n_m = n_inducing * n_latent

    ms_flat = theta[:n_m]
    ms = tf.reshape(ms_flat, (n_latent, n_inducing))

    n_l = num_triangular_elts(n_inducing) * n_latent
    Ls_flat = theta[n_m:n_m+n_l]
    Ls_flat_per_latent = tf.reshape(Ls_flat, (n_latent, -1))
    Ls = create_ls(Ls_flat_per_latent, n_inducing, n_latent)

    n_w = n_latent * n_out

    w_means_flat = theta[n_m+n_l:n_m+n_l+n_w]
    w_vars_flat = theta[n_m+n_l+n_w:n_m+n_l+2*n_w]**2

    w_means = tf.reshape(w_means_flat, (n_latent, n_out))
    w_vars = tf.reshape(w_vars_flat, (n_latent, n_out))

    if same_z:

        n_z = n_inducing * n_cov
        z_flat = theta[n_m+n_l+2*n_w:n_m+n_l+2*n_w+n_z]
        Z = tf.reshape(z_flat, (n_inducing, n_cov))
        Z = rep_matrix(Z, n_latent)

    else:

        n_z = n_inducing * n_cov * n_latent
        z_flat = theta[n_m+n_l+2*n_w:n_m+n_l+2*n_w+n_z]
        Z = tf.reshape(z_flat, (n_latent, n_inducing, n_cov))

        print(np.round(
            tf.reduce_mean(Z, axis=(1, 2)).numpy(), 2
        ))

    kern_params = theta[n_m+n_l+2*n_w+n_z:]**2

    return ms, Ls, w_means, w_vars, Z, kern_params


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

    start_cov_elts = tf.concat(start_cov_elts, axis=0).numpy()

    return start_cov_elts


dataset = BBSDataset.init_using_env_variable()

cov_df = dataset.training_set.covariates
out_df = dataset.training_set.outcomes

test_run = False
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

    # pass
    site_subset = np.random.choice(len(cov_df.index), size=400, replace=False)
    cov_df = cov_df.iloc[site_subset]
    out_df = out_df.iloc[site_subset]

scaler = StandardScaler()

x = scaler.fit_transform(cov_df.values)
y = out_df.values

if test_run:

    n_inducing = 20
    n_latent = 8

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

start_theta = np.concatenate([
    np.zeros(n_inducing * n_latent),  # m
    start_cov_elts,  # L
    np.random.randn(n_out * n_latent) * 0.1,  # W means
    np.ones(n_out * n_latent),  # W sds
    z_init.reshape(-1),
    kernel_params
])

start_theta_tensor = tf.Variable(start_theta, dtype=DTYPE)

w_prior_mean = tf.constant(0., dtype=DTYPE)
w_prior_var = tf.constant(1., dtype=DTYPE)


def to_minimize(theta):

    ms, Ls, w_means, w_vars, Z, kern_params = extract_parameters(
        theta, n_inducing, n_latent, n_out, n_cov, same_z=same_z)

    print(np.round(w_means.numpy(), 2))

    ks = create_k_fun(kern_params)

    return -compute_objective(
        x, y, Z, ms, Ls, w_means, w_vars, ks, bernoulli_probit_lik,
        w_prior_mean, w_prior_var) - tf.reduce_sum(
            tfp.distributions.Gamma(3, 3).log_prob(kern_params))


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

ms, Ls, w_means, w_vars, Z, kern_params = extract_parameters(
    final_params, n_inducing, n_latent, n_out, n_cov)

np.savez('final_params_split_lscale_prior', ms=ms,
         Ls=Ls, w_means=w_means, w_vars=w_vars, kern_params=kern_params,
         n_inducing=n_inducing, n_latent=n_latent, birds=bird_subset, Z=Z)

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from sdm_ml.dataset import BBSDataset
from sklearn.preprocessing import StandardScaler
from ml_tools.lin_alg import num_triangular_elts
from sdm_ml.gp.utils import find_starting_z
from svgp.tf.mogp import create_ls, compute_objective
from functools import partial
from ml_tools.tf_kernels import ard_rbf_kernel
from svgp.tf.likelihoods import bernoulli_probit_lik
from scipy.optimize import minimize
from svgp.tf.config import DTYPE


def extract_parameters(theta, n_inducing, n_latent, n_out, n_cov):

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

    n_z = n_inducing * n_cov
    z_flat = theta[n_m+n_l+2*n_w:n_m+n_l+2*n_w+n_z]
    Z = tf.reshape(z_flat, (n_inducing, n_cov))

    kern_params = theta[n_m+n_l+2*n_w+n_z:]**2

    return ms, Ls, w_means, w_vars, Z, kern_params


def create_ks(flat_kern_params):

    ks = [partial(ard_rbf_kernel,
          alpha=cur_params[0],
          lengthscales=cur_params[1:])
          for cur_params in tf.reshape(flat_kern_params, (n_latent, -1))]

    return ks


dataset = BBSDataset.init_using_env_variable()

cov_df = dataset.training_set.covariates
out_df = dataset.training_set.outcomes

# Choose a subset of birds to start with before I work out memory fix
bird_subset = np.random.choice(out_df.columns, size=128, replace=False)
bird_subset[0] = 'Willet'
# bird_subset = out_df.columns
out_df = out_df[bird_subset]

assert 'Willet' in bird_subset

# site_subset = np.random.choice(len(cov_df.index), size=50, replace=False)
# cov_df = cov_df.iloc[site_subset]
# out_df = out_df.iloc[site_subset]

scaler = StandardScaler()

x = scaler.fit_transform(cov_df.values)
y = out_df.values

n_inducing = 50
n_latent = 8
n_cov = int(x.shape[1])

start_z = find_starting_z(x, n_inducing)

x = tf.constant(x, dtype=DTYPE)
y = tf.constant(y, dtype=DTYPE)

n_out = int(y.shape[1])

start_theta = np.concatenate([
    np.random.randn(n_inducing * n_latent) * 0.01,  # m
    np.random.randn(num_triangular_elts(n_inducing) * n_latent),  # L
    np.random.randn(2 * n_out * n_latent),  # W means and sds
    start_z.reshape(-1),
    np.random.uniform(1, 3, size=(n_cov + 1)*n_latent),  # kernel params
])

start_theta_tensor = tf.Variable(start_theta, dtype=DTYPE)

w_prior_mean = tf.constant(0., dtype=DTYPE)
w_prior_var = tf.constant(1., dtype=DTYPE)


def to_minimize(theta):

    ms, Ls, w_means, w_vars, Z, kern_params = extract_parameters(
        theta, n_inducing, n_latent, n_out, n_cov)

    ks = create_ks(kern_params)

    return -compute_objective(x, y, Z, ms, Ls, w_means, w_vars, ks,
                              bernoulli_probit_lik, w_prior_mean, w_prior_var)


def to_minimize_with_grad(theta):

    theta_tensor = tf.Variable(theta, dtype=DTYPE)

    with tf.GradientTape() as tape:

        tape.watch(theta_tensor)

        result = to_minimize(theta_tensor)

    result_grad = tape.gradient(result, theta_tensor)

    print(result)

    return result, result_grad


result = minimize(to_minimize_with_grad, start_theta, jac=True, tol=1e-1)

final_params = result.x

np.savez('final_params', final_params)

ms, Ls, w_means, w_vars, Z, kern_params = extract_parameters(
    final_params, n_inducing, n_latent, n_out, n_cov)

np.savez('final_params_split', ms=ms, Ls=Ls, w_means=w_means, w_vars=w_vars,
         kern_params=kern_params, n_inducing=n_inducing, n_latent=n_latent,
         birds=bird_subset)

from sdm_ml.dataset import BBSDataset
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from sklearn.preprocessing import StandardScaler
from svgp.tf.utils import get_initial_values_from_kernel
from sklearn.cluster import KMeans
from ml_tools.tf_kernels import matern_kernel_32
from functools import partial
from ml_tools.tensorflow import rep_vector
from ml_tools.flattening import flatten_and_summarise_tf, reconstruct_tf
from svgp.tf.svgp import compute_expected_log_lik
from svgp.tf.likelihoods import bernoulli_probit_lik
from svgp.tf.kl import mvn_kl
from svgp.tf.mogp import (project_latents, create_ls, compute_mogp_kl_term,
                          calculate_approximate_means_and_vars)
from ml_tools.normals import covar_to_corr
from scipy.optimize import minimize
from svgp.tf.config import JITTER, DTYPE


# More ideas to try:
# Independent approximating distribution for site latents, to allow larger
# number
# In that case, I can also do the moment-matching trick to get variances on
# b_mat
# I could try to optimise some of the priors

def get_data(n_inducing, n_latent, seed=2):

    dataset = BBSDataset.init_using_env_variable()

    tf.random.set_seed(seed)
    np.random.seed(seed)

    X = dataset.training_set.covariates
    y = dataset.training_set.outcomes

    # subset = np.random.choice(X.shape[0], size=100, replace=False)
    # X = X.iloc[subset]
    # y = y.iloc[subset]

    # species_subset = np.random.choice(y.columns, size=32, replace=False)
    species_subset = y.columns
    y = y[species_subset]

    X = X.values
    y = y.values.astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    k_means = KMeans(n_inducing)
    k_means.fit(X)
    Z = k_means.cluster_centers_
    Z = np.tile(Z, (n_latent, 1, 1))

    return X, y, Z, scaler, species_subset


n_inducing = 100
n_latent = 12
n_latent_site = 8
test_run = False
extra_args = {'options': {'maxiter': 10}} if test_run else {}

X, y, Z, scaler, species = get_data(n_inducing, n_latent, seed=2)

n_cov = X.shape[1]
n_species = y.shape[1]
n_sites = X.shape[0]
latent_diag_only = True

w_prior_mean = tf.constant(0.)
w_prior_var = tf.constant(1.)

site_prior_mean = tf.zeros(n_latent_site)
site_prior_cov = tf.eye(n_latent_site)


def get_mogp_initial_values(n_cov, n_latent, n_inducing):

    ms = tf.zeros((n_latent, n_inducing))

    start_lscales = tf.random.uniform((n_latent, n_cov), minval=np.sqrt(2),
                                      maxval=np.sqrt(4))
    # start_alphas = tf.random.uniform((n_latent,), minval=0.1, maxval=1.)
    start_alphas = tf.ones((n_latent,)) * tf.sqrt(0.1)

    start_kerns = [partial(matern_kernel_32, alpha=alpha, lengthscales=lscale,
                           jitter=JITTER)
                   for alpha, lscale in zip(start_alphas, start_lscales**2)]

    w_means = tf.random.normal((n_latent, n_species), stddev=0.01)
    w_vars = tf.ones((n_latent, n_species))

    init_ls = [tf.constant(get_initial_values_from_kernel(cur_z, cur_kern)) for
               cur_z, cur_kern in zip(Z, start_kerns)]

    return (ms, start_lscales, start_alphas, start_kerns, w_means, w_vars,
            init_ls)


def get_latent_initial_values(n_latent_site, diag_only=True):

    site_start_cov = np.eye(n_latent_site)

    if diag_only:
        site_start_elts = tf.ones(n_latent_site)
    else:
        site_start_elts = tf.Variable(
            site_start_cov[np.tril_indices_from(site_start_cov)],
            dtype=tf.float32)

    site_means = tf.zeros((n_sites, n_latent_site))
    site_l_elts = rep_vector(site_start_elts, n_sites)

    b_mat = tf.random.normal((n_latent_site, n_species), stddev=0.01)

    return site_means, site_l_elts, b_mat


def compute_objective(X, y, Z, env_ms, env_Ls, ks, w_means, w_vars,
                      w_prior_mean, w_prior_var, site_prior_mean,
                      site_prior_cov, site_means, site_ls, b_mat):

    # MOGP
    means, vars = project_latents(X, Z, env_ms, env_Ls, ks)

    mogp_kl = compute_mogp_kl_term(env_ms, env_Ls, ks, Z, w_means, w_vars,
                                   w_prior_mean, w_prior_var)

    data_means, data_vars = calculate_approximate_means_and_vars(
        means, vars, w_means, tf.sqrt(w_vars))

    # Latent
    site_covs = site_ls @ tf.transpose(site_ls, (0, 2, 1))
    site_covs = site_covs + tf.eye(site_prior_cov.shape[0]) * JITTER
    res_means = site_means @ b_mat

    # Trick to get diagonal only
    res_vars = tf.einsum('ik,jkl,li->ji', tf.transpose(b_mat),
                         site_covs, b_mat)

    # Combine for likelihood
    combined_means = data_means + res_means
    combined_vars = data_vars + res_vars

    lik = compute_expected_log_lik(
        tf.reshape(combined_means, (-1,)), tf.reshape(combined_vars, (-1,)),
        tf.reshape(tf.constant(y, dtype=tf.float32), (-1,)),
        bernoulli_probit_lik)

    # Latent KL
    res_kl = tf.reduce_sum(mvn_kl(site_means, site_covs, site_prior_mean,
                                  site_prior_cov, is_batch=True))

    total_kl = res_kl + mogp_kl

    if tf.math.is_nan(lik) or tf.math.is_nan(total_kl):
        import ipdb; ipdb.set_trace()

    return lik - total_kl


# Get the MOGP init values:
ms, lscales, alphas, kerns, w_means, w_vars, init_ls = get_mogp_initial_values(
    n_cov, n_latent, n_inducing)

# Get the latent init values
site_means, site_l_elts, b_mat = get_latent_initial_values(
    n_latent_site, diag_only=latent_diag_only)

start_theta, summary = flatten_and_summarise_tf(**{
    'env_ms': ms,
    'env_l_elts': tf.stack(init_ls),
    'lscales': lscales,
    'w_means': w_means,
    'w_vars': w_vars,
    'site_means': site_means,
    'site_l_elts': site_l_elts,
    'b_mat': b_mat,
    'Z': tf.Variable(Z, dtype=DTYPE)
})


def to_minimize(x):

    theta = reconstruct_tf(x, summary)

    # TODO: Check initial values are still consistent here
    kerns = [partial(matern_kernel_32, alpha=alpha, lengthscales=lscale,
                     jitter=JITTER) for
             alpha, lscale in zip(alphas, theta['lscales']**2)]

    if not latent_diag_only:
        site_ls = create_ls(theta['site_l_elts'], n_latent_site, n_sites)
    else:
        site_ls = tf.linalg.diag(theta['site_l_elts'])

    env_ls = create_ls(theta['env_l_elts'], n_inducing, n_latent)

    objective = compute_objective(
        X, y, theta['Z'], theta['env_ms'], env_ls, kerns, theta['w_means'],
        theta['w_vars']**2, w_prior_mean, w_prior_var, site_prior_mean,
        site_prior_cov, theta['site_means'], site_ls, theta['b_mat'])

    cur_corr_mat = tf.transpose(theta['b_mat']) @ theta['b_mat']

    lengthscale_prior = tf.reduce_sum(tfp.distributions.Gamma(3, 3).log_prob(
        theta['lscales']**2))

    b_mat_prior = tf.reduce_sum(tfp.distributions.Normal(0, 1.).log_prob(
        theta['b_mat']))

    objective += lengthscale_prior + b_mat_prior

    print(np.round(covar_to_corr(cur_corr_mat.numpy()), 2))

    if tf.math.is_nan(objective):
        import ipdb; ipdb.set_trace()

    return -objective


def to_minimize_with_grad(x):

    x = tf.Variable(x, dtype=tf.float32)

    with tf.GradientTape() as tape:

        tape.watch(x)

        cur_objective = to_minimize(x)

        grad = tape.gradient(cur_objective, x)

    print(cur_objective)

    return (cur_objective.numpy().astype(np.float64),
            grad.numpy().astype(np.float64))



result = minimize(to_minimize_with_grad, start_theta, jac=True,
                  method='L-BFGS-B', **extra_args)

final_theta = reconstruct_tf(result.x, summary)

np.savez('final_theta_diag_site_l',
         alphas=alphas.numpy(),
         species_subset=species, scaler_mean=scaler.mean_,
         scaler_scale=scaler.scale_, n_inducing=n_inducing,
         n_latent=n_latent, n_latent_site=n_latent_site,
         **{x: y.numpy() for x, y in final_theta.items()})

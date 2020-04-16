import numpy as np
import tensorflow as tf
from svgp.tf.models.ppp_mogp import initialise_theta, build_spec
from svgp.tf.experimental.linear_mogp import calculate_kl


def test_init_consistent():

    n_cov = 8
    n_inducing = 10
    n_latent = 4
    n_out = 16
    n_thin_inducing = 12
    n_thin_cov = 2

    init_w_var = 2.
    log_cov_alpha = 3.
    log_thin_alpha = 4.

    Z = np.random.randn(n_inducing, n_cov)
    Z_thin = np.random.randn(n_thin_inducing, n_thin_cov)

    start_theta = initialise_theta(
        Z, n_latent, n_cov, n_out, Z_thin, init_w_var, log_cov_alpha,
        log_thin_alpha)

    # FIXME: Adding these in here is a bit weird. Can't initialise_theta
    # set these?
    start_theta['log_cov_alpha'] = log_cov_alpha
    start_theta['log_thin_alpha'] = log_thin_alpha

    init_spec = build_spec(start_theta)

    cov_alphas = tf.stack([x.keywords['alpha'] for x in
                           init_spec.cov_mogp_spec.multi_gp.kernel_funs])
    thin_alphas = tf.stack([x.keywords['alpha'] for x in
                           init_spec.thin_mogp_spec.multi_gp.kernel_funs])

    w_vars = init_spec.cov_mogp_spec.w_vars

    assert all(cov_alphas == tf.exp(log_cov_alpha))
    assert all(thin_alphas == tf.exp(log_thin_alpha))
    assert tf.reduce_all(w_vars == init_w_var)

    # Check the KL divergence
    kl_cov_mogp = calculate_kl(init_spec.cov_mogp_spec)
    assert kl_cov_mogp < 1

    kl_thin_mogp = calculate_kl(init_spec.thin_mogp_spec)
    assert kl_thin_mogp < 1

import tensorflow as tf
from svgp.tf.kl import normal_kl_1d
import svgp.tf.experimental.multi_inducing_point_gp as m_gp
from ..mogp import calculate_approximate_means_and_vars
from ml_tools.normals import \
    moments_of_linear_combination_rvs_selected_independent
from typing import NamedTuple, Tuple, Optional


class LinearMOGPSpec(NamedTuple):

    multi_gp: m_gp.MultiInducingPointGPSpecification
    w_means: tf.Tensor
    w_vars: tf.Tensor
    w_prior_mean: tf.Tensor
    w_prior_var: tf.Tensor
    intercept_means: Optional[tf.Tensor] = None
    intercept_vars: Optional[tf.Tensor] = None
    intercept_prior_mean: Optional[tf.Tensor] = None
    intercept_prior_var: Optional[tf.Tensor] = None


def calculate_kl(linear_mogp: LinearMOGPSpec) -> float:

    mogp_kl = m_gp.calculate_kl(linear_mogp.multi_gp)

    # Compute the w KL
    kl_weights = tf.reduce_sum(
        normal_kl_1d(linear_mogp.w_means, linear_mogp.w_vars,
                     linear_mogp.w_prior_mean, linear_mogp.w_prior_var))

    if linear_mogp.intercept_prior_mean is not None:

        # Compute the intercept KL
        kl_intercept = tf.reduce_sum(
            normal_kl_1d(linear_mogp.intercept_means,
                         linear_mogp.intercept_vars,
                         linear_mogp.intercept_prior_mean,
                         linear_mogp.intercept_prior_var))

    else:

        kl_intercept = 0.

    return mogp_kl + kl_weights + kl_intercept


def project_to_x(linear_mogp: LinearMOGPSpec,
                 X: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

    # Project the multi_gp
    latent_mean, latent_var = m_gp.project_to_x(
        linear_mogp.multi_gp, X, diag_only=True)

    # Here, we compute all outputs. A different function will compute a subset.
    # TODO: Maybe refactor this function to use variances?
    out_means, out_vars = calculate_approximate_means_and_vars(
        latent_mean, latent_var, tf.transpose(linear_mogp.w_means),
        tf.transpose(tf.sqrt(linear_mogp.w_vars)))

    if linear_mogp.w_means is not None:

        # Add on the intercept
        # TODO: Do I need to reshape these?
        out_means += linear_mogp.intercept_means
        out_vars += linear_mogp.intercept_vars

    return out_means, out_vars


def project_selected_to_x(
        linear_mogp: LinearMOGPSpec, X: tf.Tensor, output_nums: tf.Tensor) \
        -> Tuple[tf.Tensor, tf.Tensor]:

    latent_mean, latent_var = m_gp.project_to_x(linear_mogp.multi_gp, X,
                                                diag_only=True)

    rel_means = tf.gather(linear_mogp.w_means, output_nums)
    rel_vars = tf.gather(linear_mogp.w_vars, output_nums)

    res_means, res_vars = \
        moments_of_linear_combination_rvs_selected_independent(
            tf.transpose(latent_mean), tf.transpose(latent_var), rel_means,
            rel_vars, sum_fun=tf.reduce_sum)

    if linear_mogp.w_means is not None:

        rel_intercept_means = tf.gather(linear_mogp.intercept_means,
                                        output_nums)
        rel_intercept_vars = tf.gather(linear_mogp.intercept_vars, output_nums)

        res_means += rel_intercept_means
        res_vars += rel_intercept_vars

    return res_means, res_vars

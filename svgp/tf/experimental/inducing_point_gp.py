import tensorflow as tf
from typing import NamedTuple, Callable, Optional, Tuple
from svgp.tf.svgp import compute_kl_term, compute_qf_mean_cov
from ml_tools.tensorflow import lo_tri_from_elements
from svgp.tf.utils import get_initial_values_from_kernel

# TODO: These type hints are kind of bogus because we can just as well use
# tensorflow objects.
# TODO: I could think about storing the kernel hyperparameters separately.


class InducingPointGPSpecification(NamedTuple):

    L_elts: tf.Tensor
    mu: tf.Tensor
    kernel_fun: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
    Z: tf.Tensor


def calculate_kl(gp: InducingPointGPSpecification) -> float:

    L = lo_tri_from_elements(gp.L_elts, gp.mu.shape[0])

    return compute_kl_term(gp.mu, L, gp.Z, gp.kernel_fun)


def initialise_using_kernel_fun(
        kernel_fun: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], Z: tf.Tensor,
        initial_mu: Optional[tf.Tensor] = None) \
        -> InducingPointGPSpecification:

    initial_L = get_initial_values_from_kernel(Z, kernel_fun)

    if initial_mu is None:
        initial_mu = tf.zeros(Z.shape[0])

    return InducingPointGPSpecification(mu=initial_mu, L_elts=initial_L,
                                        kernel_fun=kernel_fun, Z=Z)


def project_to_x(gp: InducingPointGPSpecification, X: tf.Tensor,
                 diag_only=True) -> Tuple[tf.Tensor, tf.Tensor]:

    L = lo_tri_from_elements(gp.L_elts, gp.mu.shape[0])

    return compute_qf_mean_cov(L, gp.mu, X, gp.Z, gp.kernel_fun,
                               diag_only=diag_only)

import tensorflow as tf
from typing import NamedTuple, Callable, Tuple, List
from ..mogp import create_ls, project_latents
from ..svgp import compute_kl_term
from ..utils import get_initial_values_from_kernel


class MultiInducingPointGPSpecification(NamedTuple):

    L_elts: tf.Tensor
    mus: tf.Tensor
    kernel_funs: List[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]
    Zs: tf.Tensor


def calculate_kl(multi_gp: MultiInducingPointGPSpecification) -> float:

    n_inducing = multi_gp.mus.shape[1]
    n_gps = multi_gp.mus.shape[0]

    Ls = create_ls(multi_gp.L_elts, n_inducing, n_gps)

    kl = tf.reduce_sum([compute_kl_term(cur_m, cur_l, cur_z, cur_k) for cur_m,
                        cur_l, cur_z, cur_k in zip(multi_gp.mus, Ls,
                                                   multi_gp.Zs,
                                                   multi_gp.kernel_funs)])

    return kl


def initialise_using_kernel_funs(
        kernel_funs: List[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
        Zs: tf.Tensor):

    initial_Ls = [get_initial_values_from_kernel(cur_Z, cur_kern)
                  for cur_Z, cur_kern in zip(Zs, kernel_funs)]

    initial_Ls = tf.stack(initial_Ls)

    initial_mu = tf.zeros((len(kernel_funs), Zs[0].shape[0]))

    return MultiInducingPointGPSpecification(
        mus=initial_mu, L_elts=initial_Ls, kernel_funs=kernel_funs,
        Zs=Zs)


def project_to_x(multi_gp: MultiInducingPointGPSpecification,
                 X: tf.Tensor, diag_only=True) -> Tuple[tf.Tensor, tf.Tensor]:

    n_inducing = multi_gp.mus.shape[1]
    n_gps = multi_gp.mus.shape[0]

    Ls = create_ls(multi_gp.L_elts, n_inducing, n_gps)

    return project_latents(X, multi_gp.Zs, multi_gp.mus, Ls,
                           multi_gp.kernel_funs)

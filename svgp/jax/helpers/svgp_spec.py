from typing import NamedTuple, Callable
import jax.numpy as jnp
from svgp.jax.kl import mvn_kl
from ml_tools.jax import pos_def_mat_from_tri_elts, vector_from_pos_def_mat
from svgp.jax.svgp import project_to_f_given_L_elts


class SVGPSpec(NamedTuple):

    m: jnp.ndarray
    L_elts: jnp.ndarray
    Z: jnp.ndarray
    kern_fn: Callable[[jnp.ndarray, jnp.ndarray, bool], jnp.ndarray]


def calculate_kl(spec: SVGPSpec):

    S = pos_def_mat_from_tri_elts(spec.L_elts, spec.m.shape[0], 1e-6)
    prior_mu = jnp.zeros_like(spec.m)
    prior_cov = spec.kern_fn(spec.Z, spec.Z, diag_only=False)

    return mvn_kl(spec.m, S, prior_mu, prior_cov)


def project_to_x(spec: SVGPSpec, X: jnp.ndarray, diag_only=True):

    pred_means, pred_vars = project_to_f_given_L_elts(
        X, spec.Z, spec.m, spec.L_elts, spec.kern_fn, diag_only=diag_only
    )

    return pred_means, pred_vars


def initialise_using_kernel_fun(kern_fn, Z):

    kmm = kern_fn(Z, Z, diag_only=False)
    L_elts = vector_from_pos_def_mat(kmm, 1e-6)
    m = jnp.zeros(Z.shape[0])

    return SVGPSpec(m, L_elts, Z, kern_fn)

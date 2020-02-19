import tensorflow as tf
from typing import NamedTuple, Callable, Dict, Any, Optional
from functools import partial


class KernelSpec(NamedTuple):

    # TODO: How do I express kwargs in the callable?
    base_kernel_fun: Callable[[tf.Tensor, tf.Tensor, Any], tf.Tensor]
    parameters: Dict[str, tf.Tensor]
    constraints: Dict[str, str] = dict()
    priors: Dict[str, Callable[[tf.Tensor], tf.Tensor]] = dict()


def constrain_positive(parameter: tf.Tensor) -> tf.Tensor:

    return tf.exp(parameter)


def constrain(parameter: tf.Tensor, constraint: Optional[str]) -> tf.Tensor:

    assert constraint in ['+', None], \
        'Only positive constraint supported at present.'

    if constraint is None:
        return parameter
    else:
        return constrain_positive(parameter)


def constrain_all(parameters: Dict[str, tf.Tensor],
                  constraints: Dict[str, str]) -> Dict[str, tf.Tensor]:

    constrained_params = {x: constrain(y, constraints.get(x, None))
                          for x, y in parameters.items()}

    return constrained_params


def get_kernel_fun(kernel_spec: KernelSpec) -> \
        Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:

    # Apply constraints to kernel parameters
    constrained_params = constrain_all(kernel_spec.parameters,
                                       kernel_spec.constraints)

    return partial(kernel_spec.base_kernel_fun, **constrained_params)


def update_parameters(kernel_spec: KernelSpec,
                      new_params: Dict[str, tf.Tensor]) -> KernelSpec:

    new_params = {x: y for x, y in new_params.items() if x in
                  kernel_spec.parameters}

    return KernelSpec(kernel_spec.base_kernel_fun, new_params,
                      kernel_spec.constraints, kernel_spec.priors)


def calculate_prior_prob(kernel_spec: KernelSpec):

    total_log_prob = 0.

    for cur_parameter, cur_prior_fun in kernel_spec.priors.items():

        constrained = constrain(kernel_spec.parameters[cur_parameter],
                                kernel_spec.constraints[cur_parameter])

        prior_prob = cur_prior_fun(constrained)

        total_log_prob += prior_prob

    return total_log_prob

import numpy as np
from functools import partial


def get_default_kernel(kernel_fun, n_cov, lscale_min=1., lscale_max=4.,
                       alpha=1., dtype=np.float32):

    lscales = np.random.uniform(
        lscale_min, lscale_max, size=n_cov).astype(dtype)

    return partial(kernel_fun, lengthscales=lscales, alpha=alpha)

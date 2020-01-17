# Let's just use a really small subset
import numpy as np
from sdm_ml.utils import get_bbs_dataset
from svgp.tf.correlated_mogp_classifier import fit

X, y, species, sites, scaler = get_bbs_dataset(None, None)

result = fit(X, y, n_inducing=100, n_latent=10)

np.savez('correlated_mogp_bbs_lscale_prior', **result._asdict())

# Let's just use a really small subset
from sdm_ml.utils import get_bbs_dataset
from svgp.tf.correlated_mogp_classifier import fit


n_out = 16

X, y, species, sites, scaler = get_bbs_dataset(100, n_out)

result = fit(X, y, n_inducing=10, n_latent=2)

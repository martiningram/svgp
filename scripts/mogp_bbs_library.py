import numpy as np
from sdm_ml.dataset import BBSDataset
from sklearn.preprocessing import StandardScaler
from svgp.tf.mogp_classifier import fit


dataset = BBSDataset.init_using_env_variable()

cov_df = dataset.training_set.covariates
out_df = dataset.training_set.outcomes

# test_choice = np.random.permutation(cov_df.shape[0])[:100]
# test_birds = np.random.permutation(out_df.shape[1])[:16]
#
# cov_df = cov_df.iloc[test_choice]
# out_df = out_df.iloc[test_choice].iloc[:, test_birds]

scaler = StandardScaler()

x = scaler.fit_transform(cov_df.values)
y = out_df.values

n_inducing = 100
n_latent = 10

result = fit(x, y, n_inducing, n_latent)
dict_version = result._asdict()

np.savez('mogp_fit_library_wvar_prior', **dict_version)

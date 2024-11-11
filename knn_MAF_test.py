from datetime import date
import os
import numpy as np
import distributions.sim_models as models
import estimators as est
import util.io
import util.misc
import argparse

today = date.today().strftime("%b_%d")

# Get input type
parser = argparse.ArgumentParser()
parser.add_argument("dist_type", help="Distribution: {uniform,gaussian,limited_gaussian}")
args = parser.parse_args()
if args.dist_type == "uniform":
    sim_model = models.Uniform(0, 1, 3)
    path = "saved_data/uniform"
    filename = "uniform_data({})".format(today)
elif args.dist_type == "gaussian":
    sim_model = models.Gaussian(0, 1, 3)
    path = "saved_data/gaussian"
    filename = "gaussian_data({})".format(today)
elif args.dist_type == "limited_gaussian":
    sim_model = models.Limited_Gaussian(0, 1, 3, 3)
    path = "saved_data/limited_gaussian"
    filename = "limited_gaussian_data({})".format(today)
else:
    raise ValueError("Invalid distribution type")

filename = util.io.update_filename(path=path, old_name=filename, rename=False)

# set up number of sims to run
n_samp_range = (10 ** np.linspace(2, 4.5, num=9)).astype(np.int32)
n_iterations = 100
k_range = [1, 2, 5, 10]

H_MAF = np.empty((n_iterations, len(n_samp_range))) * np.nan
H_Laplace = np.empty((n_iterations, len(n_samp_range), len(k_range))) * np.nan

H_true = sim_model.entropy()

for i in range(n_iterations):
    for n, n_samples in enumerate(n_samp_range):
        util.misc.print_border(f"Gaussian, n_samples={n_samples}, iter={i}")
        samples = sim_model.sim(n_samples)

        H_Laplace[i, n, :] = est.knn.knn_laplace(samples, k=k_range)
        # H_MAF[i, n] = est.maf.MAF_entropy(samples)

        filename = util.io.update_filename(path, filename, i)
        util.io.save(
            (n_samp_range, k_range, H_Laplace, H_MAF, H_true), os.path.join(path, filename)
        )

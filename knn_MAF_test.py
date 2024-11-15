from datetime import date
import os
import numpy as np
import distributions.sim_models as models
import estimators as est
import util.io
import util.misc
import argparse

today = date.today().strftime("%b_%d")

N_dims = 10
path = f"saved_data/{N_dims}d"

# Get input type
parser = argparse.ArgumentParser()
parser.add_argument("dist_type", help="Distribution: {uniform,gaussian,limited_gaussian}")
args = parser.parse_args()
if args.dist_type == "uniform":
    sim_model = models.Uniform(0, 1, N_dims)
elif args.dist_type == "gaussian":
    sim_model = models.Gaussian(0, 1, N_dims)
elif args.dist_type == "limited_gaussian":
    sim_model = models.Limited_Gaussian(0, 1, 3, N_dims)
elif args.dist_type == "laplace":
    sim_model = models.Laplace(0, 1, N_dims)
else:
    raise ValueError("Invalid distribution type")

path = os.path.join(path, args.dist_type)
filename = f"{args.dist_type}_data({today})"

filename = util.io.update_filename(path=path, old_name=filename, rename=False)

# set up number of sims to run
n_samp_range = (10 ** np.linspace(2, 4.5, num=9)).astype(np.int32)
n_iterations = 100
k_range = [1, 2, 5, 10]


H_Laplace = np.empty((n_iterations, len(n_samp_range), len(k_range))) * np.nan
H_MAF = np.empty((n_iterations, len(n_samp_range))) * np.nan
H_KNN_MAF = np.empty((n_iterations, len(n_samp_range))) * np.nan
H_uniformized = np.empty((n_iterations, len(n_samp_range))) * np.nan


H_true = sim_model.entropy()


for i in range(n_iterations):
    for n, n_samples in enumerate(n_samp_range):
        util.misc.print_border(f"Gaussian, n_samples={n_samples}, iter={i}")
        samples = sim_model.sim(n_samples)
        model = None

        H_Laplace[i, n, :] = est.knn.knn_laplace(samples, k=k_range)
        H_MAF[i, n], model = est.maf.MAF_entropy(samples, model)
        H_KNN_MAF[i, n], model = est.maf.MAF_KNN_entropy(samples, model)
        H_uniformized[i, n], model = est.maf.uniformized_entropy(samples, model)

        filename = util.io.update_filename(path, filename, i)
        util.io.save(
            (n_samp_range, k_range, H_Laplace, H_MAF, H_KNN_MAF, H_uniformized, H_true),
            os.path.join(path, filename),
        )

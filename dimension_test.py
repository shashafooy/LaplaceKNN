from datetime import date
import os
import numpy as np
import distributions.sim_models as models
import estimators as est
import util.io
import util.misc
import argparse

today = date.today().strftime("%b_%d")


path = f"saved_data/drange"

# Get input type
parser = argparse.ArgumentParser()
parser.add_argument("dist_type", help="Distribution: {uniform,gaussian,limited_gaussian,laplace}")
args = parser.parse_args()
if args.dist_type == "uniform":
    sim_model = models.Uniform(0, 1, 2)
elif args.dist_type == "gaussian":
    sim_model = models.Gaussian(0, 1, 2)
elif args.dist_type == "limited_gaussian":
    sim_model = models.Limited_Gaussian(0, 1, 3, 2)
elif args.dist_type == "laplace":
    sim_model = models.Laplace(0, 1, 2)
else:
    raise ValueError("Invalid distribution type")

path = os.path.join(path, args.dist_type)
filename = f"{args.dist_type}_data({today})"

filename = util.io.update_filename(path=path, old_name=filename, rename=False)

# set up number of sims to run
N_dims = np.asarray(range(1, 21))
n_samp_range = 10000 * N_dims
n_iterations = 100


H_Laplace = np.empty((n_iterations, len(N_dims))) * np.nan
H_MAF = np.empty((n_iterations, len(N_dims))) * np.nan
# H_KNN_MAF = np.empty((n_iterations, len(N_dims))) * np.nan
H_uniformized = np.empty((n_iterations, len(N_dims))) * np.nan


sim_model.set_dim(1)
H_true = sim_model.entropy() * N_dims


for i in range(n_iterations):
    for j, (dims, n_samples) in enumerate(zip(N_dims, n_samp_range)):
        util.misc.print_border(f"n_samples={n_samples}, dim={dims}, iter={i}")
        sim_model.set_dim(dims)
        samples = sim_model.sim(n_samples)
        model = None

        H_Laplace[i, j] = est.knn.knn_laplace(samples)
        H_MAF[i, j], model = est.maf.MAF_entropy(samples, model)
        # H_KNN_MAF[i, n], model = est.maf.MAF_KNN_entropy(samples, model)
        H_uniformized[i, j], model = est.maf.uniformized_entropy(samples, model)

        filename = util.io.update_filename(path, filename, i)
        util.io.save(
            (n_samp_range, N_dims, H_Laplace, H_MAF, H_uniformized, H_true),
            os.path.join(path, filename),
        )

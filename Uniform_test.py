from datetime import date
import os
import numpy as np
import distributions.sim_models as models
import estimators as est
import util.io
import util.misc

# set up number of sims to run
n_samp_range = (10 ** np.linspace(2, 4.5, num=9)).astype(np.int32)
n_iterations = 100

H_MAF = np.empty((n_iterations, len(n_samp_range))) * np.nan
H_Laplace = np.empty((n_iterations, len(n_samp_range))) * np.nan

path = "saved_data/3d_gaussian"
path = "temp_data/laplace_test/high_epoch"
today = date.today().strftime("%b_%d")
filename = "laplace_data({})".format(today)
filename = util.io.update_filename(path=path, old_name=filename, rename=False)

# sim_model = models.Gaussian(0, 1, N=3)
sim_model = models.Uniform(0, 1, 3)
H_true = sim_model.entropy()

for i in range(n_iterations):
    for n, n_samples in enumerate(n_samp_range):
        util.misc.print_border(f"Gaussian, n_samples={n_samples}, iter={i}")
        samples = sim_model.sim(n_samples)

        H_Laplace[i, n] = est.knn.knn_laplace(samples)
        H_MAF[i, n] = est.maf.MAF_entropy(samples)

        filename = util.io.update_filename(path, filename, i)
        util.io.save((n_samp_range, H_Laplace, H_MAF, H_true), os.path.join(path, filename))

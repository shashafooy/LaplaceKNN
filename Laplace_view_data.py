import re
import util.io
import os
import matplotlib.pyplot as plt
import numpy as np

# from simulators.CPDSSS_models import Laplace
from distributions import sim_models as mod
from util import viewData

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 14})


"""
Load and combine all datasets
"""
N_dims = 3

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True

filepath = f"saved_data/{N_dims}d/limited_gaussian"

name = rf"$\mathcal{{N}}(0,{{\bf I}}_{N_dims})$"
name = rf"$\mathcal{{N}}(0,{{\bf I}}_{N_dims}), ||{{\bf x}}|| \leq 3$"
# name = rf"Unif$([0,1]^{{N-dims}})$"

for filename in os.listdir(filepath):
    # if name is a folder or contains -1_iter, skip
    if not os.path.isfile(os.path.join(filepath, filename)) or re.search(r"\(-1_iter\)", filename):
        continue
    filename = os.path.splitext(filename)[0]  # remove extention
    _N_samples, _k_list, _H_Laplace, _H_MAF, _H_true = util.io.load(
        os.path.join(filepath, filename)
    )

    # Initialize arrays
    if "H_Laplace" not in locals():
        N_samples = _N_samples
        N_size = len(N_samples)
        k_list = _k_list
        H_Laplace = np.empty((0, N_size, len(k_list)))
        H_MAF = np.empty((0, N_size))
        H_true = _H_true

    H_Laplace, (_, k_list) = viewData.align_and_concatenate(
        H_Laplace, _H_Laplace, (N_samples, k_list), (_N_samples, _k_list)
    )
    H_MAF, N_samples = viewData.align_and_concatenate(H_MAF, _H_MAF, N_samples, _N_samples)

viewData.clean_data(H_Laplace)
viewData.clean_data(H_MAF)

# Remove any data that is outside of 3 standard deviations. These data points can be considered outliers.
if REMOVE_OUTLIERS:
    viewData.remove_outlier(H_Laplace)
    viewData.remove_outlier(H_MAF)


H_Laplace_mean = np.nanmean(H_Laplace, axis=0)
H_MAF_mean = np.nanmean(H_MAF, axis=0)

H_Laplace_std = np.nanstd(H_Laplace, axis=0)
H_MAF_std = np.nanstd(H_MAF, axis=0)


MSE_H_Laplace = np.nanmean((H_Laplace - H_true) ** 2, axis=0)
MSE_H_MAF = np.nanmean((H_MAF - H_true) ** 2, axis=0)

# RMSE_unif_KL = np.sqrt(MSE_unif_KL)
# RMSE_unif_KSG = np.sqrt(MSE_unif_KSG)
# RMSE_KL = np.sqrt(MSE_KL)
# RMSE_KSG = np.sqrt(MSE_KSG)

# err_unif_KL = np.abs(H_true - H_unif_KL_mean)
# err_unif_KSG = np.abs(H_true - H_unif_KSG_mean)
# err_KL = np.abs(H_true - H_KL_mean)
# err_KSG = np.abs(H_true - H_KSG_mean)

# PLOTS
N_samples = np.log10(N_samples)

# entropy
plt.figure(0)
plt.axhline(y=H_true, linestyle="dashed")
for i, k in enumerate(k_list):
    plt.plot(N_samples, H_Laplace_mean[:, i], label=rf"k={k}")
plt.plot(N_samples, H_MAF_mean, label=f"MAF")
# plt.xscale("log")

plt.title(name)
plt.legend()
plt.xlabel("log10 samples")
plt.ylabel("Entropy")
plt.tight_layout()


## Show error bars
plt.figure(1)
plt.axhline(y=H_true, linestyle="dashed")
for i, k in enumerate(k_list):
    plt.errorbar(N_samples, H_Laplace_mean[:, i], yerr=H_Laplace_std[:, i], label=rf"k={k}")
plt.errorbar(N_samples, H_MAF_mean, yerr=H_MAF_std, label=f"MAF")
# plt.xscale("log")

plt.title(name)
plt.legend()
plt.xlabel("log10 samples")
plt.ylabel("Entropy")
plt.tight_layout()


## MSE
plt.figure(2)
for i, k in enumerate(k_list):
    plt.plot(N_samples, MSE_H_Laplace[:, i], label=rf"k={k}")
plt.plot(N_samples, MSE_H_MAF, label=f"MAF")
# plt.xscale("log")

plt.title(f"{name} MSE")
plt.legend()
plt.xlabel("log10 samples")
plt.ylabel("MSE")
plt.tight_layout()

plt.show()

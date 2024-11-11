import re
import util.io
import os
import matplotlib.pyplot as plt
import numpy as np

# from simulators.CPDSSS_models import Laplace
from util import viewData

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 14})


"""
Load and combine all datasets
"""
max_T = 0
min_T = 0

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True

filepath = "saved_data/uniform"


N = 2
L = 2
# filepath=filepaths[1]
# idx=0
# for idx,filepath in enumerate(filepaths):
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

# entropy
plt.figure(0)
plt.axhline(y=H_true, linestyle="dashed")
for i, k in enumerate(k_list):
    plt.errorbar(N_samples, H_Laplace_mean[i], yerr=H_Laplace_std, label=rf"k={k}")
plt.errorbar(N_samples, H_MAF, yerr=H_MAF_std, label=f"MAF")
plt.title(os.path.basename(filepath))

plt.legend()
plt.xlabel("log10 samples")
plt.ylabel("Entropy")
plt.xticks(np.linspace(2, 4.5, 6))
plt.tight_layout()

# Absolute error
plt.figure(1)
plt.plot(N_samples, err_unif_KL, "--^")
plt.plot(N_samples, err_unif_KSG, "--v")
plt.plot(N_samples, err_KL, "--x")
plt.plot(N_samples, err_KSG, "--o")
plt.yscale("log")
plt.title("Entropy Error")
plt.legend(["Uniform KL", "Uniform KSG", "KL", "KSG"])
plt.xlabel("d dimensions")
plt.ylabel("log error")
plt.xticks(N_samples[::2])
plt.tight_layout()

# MSE
plt.figure(2)
plt.plot(N_samples, MSE_unif_KL, "--^")
plt.plot(N_samples, MSE_unif_KSG, "--v")
plt.plot(N_samples, MSE_KL, "--x")
plt.plot(N_samples, MSE_KSG, "--o")
plt.yscale("log")
plt.title("Entropy MSE Error")
plt.legend(["Uniform KL", "Uniform KSG", "KL", "KSG"])
plt.xlabel("d dimensions")
plt.ylabel("log MSE")
plt.xticks(N_samples[::2])
plt.tight_layout()

# RMSE
plt.figure(3)
plt.plot(N_samples, RMSE_unif_KL, "--^")
plt.plot(N_samples, RMSE_unif_KSG, "--v")
plt.plot(N_samples, RMSE_KL, "--x")
plt.plot(N_samples, RMSE_KSG, "--o")
plt.yscale("log")
plt.title("Entropy RMSE Error")
plt.legend(["Uniform KL", "Uniform KSG", "KL", "KSG"])
plt.xlabel("d dimensions")
plt.ylabel("log RMSE")
plt.xticks(N_samples[::2])
plt.tight_layout()

# STD
plt.figure(4)
plt.plot(N_samples, H_unif_KL_std, "--^")
plt.plot(N_samples, H_unif_KSG_std, "--v")
plt.plot(N_samples, H_KL_std, "--x")
plt.plot(N_samples, H_KSG_std, "--o")
plt.yscale("log")
plt.title("Entropy std")
plt.legend(["Uniform KL", "Uniform KSG", "KL", "KSG"])
plt.xlabel("d dimensions")
plt.ylabel("log std")
plt.xticks(N_samples[::2])
plt.tight_layout()


plt.show()

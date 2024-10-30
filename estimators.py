import numpy as np
import scipy.special as spl
import subprocess

#If a GPU exists, cuml version of KNN should be used
try:
    subprocess.check_output('nvidia-smi')
    from cuml.neighbors import NearestNeighbors
except Exception:
    from sklearn.neighbors import NearestNeighbors





dtype = np.float32

def knn_shannon(data, k=1, n=None):
    rng = np.random
    y=np.asarray(y,dtype)
    N,dim = y.shape
    if n is not None:
        N = min(n,N)

    # KNN

    return

def knn_laplace(data, k=1, n=None):
    rng = np.random
    data = np.asarray(data,dtype)
    N,dim = data.shape
    if n is not None:
        N = min(n,N)



    return
    

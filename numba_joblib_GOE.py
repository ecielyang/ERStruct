import numpy as np
from numpy import linalg as LA
import time
from joblib import Parallel, delayed  # For parallel computing in for-loops
from numba import njit # To accelerate Numpy computing

@njit
def func(n):
    nprime = n-1
    main = np.sqrt(2.) * np.random.normal(0., 1., (nprime))
    off = np.random.normal(0., 1., (nprime, nprime))
    tril = np.tril(off, -1)
    W_n = tril + tril.T
    np.fill_diagonal(W_n, main)

    eigenvalues = LA.eigvals(W_n)
    return np.sort(eigenvalues)[::-1][0:2]


def GOE_L12_sim_njit(n, rep):
    GOE_0 = Parallel(n_jobs=8,)(delayed(func)(n) for i in range(rep))
    GOE = np.sort(np.array(GOE_0).T, axis=0)[::-1]
    return GOE

start = time.time()
GOE_L12_sim_njit(1000, 200)
end = time.time()
print(end - start)
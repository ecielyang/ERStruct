import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Import package
from joblib import Parallel, delayed
import numpy as np
import torch


def simulate(n):
    nprime = n - 1
    main = np.sqrt(2) * torch.normal(mean=0, std=1, size=(1, nprime))
    off = torch.normal(mean=0, std=1, size=(nprime, nprime))
    tril = torch.tril(off, -1)
    W_n = tril + tril.T

    W_n[range(len(W_n)), range(len(W_n))] = main
    eigenvalues = torch.linalg.eigvalsh(W_n)
    w = np.sort(eigenvalues)[::-1]
    return w[0:2]


def GOE_L12_sim(n, rep, cpu_num):
    GOE_0 = Parallel(n_jobs=cpu_num)(
        delayed(simulate)(n)
        for i in range(rep))

    GOE = np.sort(np.array(GOE_0).T, axis=0)[::-1]
    return GOE




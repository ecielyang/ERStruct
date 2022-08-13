import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Import package
import numpy as np
import torch
from ERStruct.Eigens import Eigens_cpu,Eigens_gpu
from ERStruct.GOE import GOE_L12_sim

class erstruct:

    # n: Total number of individuals in the study
    # path: The path of data file(s)
    # filename: The name of the data file(s)
    # rep: Number of simulation times for the null distribution
    # alpha: Significance level, can be either a scaler or a vector
    # Kc: A coarse estimate of the top PCs number (set to `floor(n/10)` by default)
    # core_num: Optional, number of CPU cores to be used for parallel computing. The default is 1
    # device_idx: "cpu" pr "gpu". The default is "cpu".
    # varm: Allocated memory (in bytes) of GPUs for computing. When device_idx="cpu", varm should be specified clearly.

    def __init__(self, n, path, filename, rep, alpha, cpu_num=1, device_idx="cpu", varm=1, Kc=-1):
        self.n = n
        self.path = path
        self.filename = filename
        self.rep = rep
        self.alpha = alpha
        self.cpu_num = cpu_num
        self.device_idx = device_idx
        self.varm = varm
        self.Kc = Kc



    def run(self):
        # Return : K_hat-Estimated number of top PCs
        if self.device_idx == 'cpu':
            eigens, p = Eigens_cpu(self.n, self.path, self.filename)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device == "cpu":
                print("No GPU detected, run on cpu...")
            else:
                eigens, p = Eigens_gpu(self.n, self.varm, self.path, self.filename, device)

        if self.Kc == -1:
            Kc = int(np.floor(self.n / 10))

        print("Simulating...")
        GOE_L12_dist = GOE_L12_sim(self.n, self.rep, self.cpu_num)

        print('Testing...')
        K_tmp = 0
        K_hat = 0
        n_prime = self.n
        end = eigens.shape[0]
        stats0 = eigens[1:end - 1] / eigens[0:end - 2]
        stats = np.insert(stats0, 0, 0)
        xi_GOE_s = np.zeros((Kc, 1), dtype='complex_')

        while K_tmp < Kc:
            K_tmp = K_tmp + 1
            n_prime = n_prime - 1

            a_p_hat = np.sum(eigens[K_tmp - 1:end - 1]) / n_prime
            b_p_hat = p / n_prime * (np.sum(eigens[K_tmp - 1:end - 1] ** 2) / n_prime - a_p_hat ** 2)

            xi_GOE_rep = (GOE_L12_dist[1, :] * np.sqrt(b_p_hat / p) + a_p_hat) / (
                    GOE_L12_dist[0, :] * np.sqrt(b_p_hat / p) + a_p_hat)
            xi_GOE_rep = np.sort(xi_GOE_rep)
            xi_GOE_s[K_tmp - 1] = xi_GOE_rep[int(np.ceil(max(xi_GOE_rep.shape) * self.alpha)) - 1]

            if K_hat == 0 and stats[K_tmp] > xi_GOE_s[K_tmp - 1]:  # jump above threshold
                K_hat = K_tmp
            elif K_hat != 0 and stats[K_tmp] <= xi_GOE_s[K_hat - 1]:  # fake
                K_hat = 0

        # if output K_hat == 0, then show error message
        if K_hat == 0:
            print('Cannot find valid K_hat <= Kc')
        print("K_hat is ", K_hat)
        return K_hat



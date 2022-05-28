import numpy as np
import logging
from numpy import linalg as LA


def Eigens(n, path, filename):
    logging.getLogger().setLevel(logging.INFO)
    p = 0
    X = np.zeros((n, n))

    for i in range(len(filename)):
        logging.info('Processing file ' + str(i + 1))

        # Read npy file
        dataset = np.load(path + str(filename[i]) + ".npy")

        # centerlize and scale
        mu = np.nanmean(dataset, 0)
        mu = np.reshape(mu, (1, dataset.shape[1]))

        q = (np.nansum(dataset, 0) + 1) / (2 * n + 2)
        q = np.reshape(q, (1, dataset.shape[1]))

        M = (dataset - mu) / np.sqrt(2 * q * (1 - q))

        # replace initial missing data and the NaN caused by deviding 0 mean
        M = np.nan_to_num(M, nan=0)
        p_i = max(M.shape)
        X = X + M @ M.T
        p = p + p_i

    X = 1 / p * X
    eigenvalues = LA.eigvals(X)
    eigens = np.flip(np.sort(eigenvalues))

    return eigens, p


def GOE_L12_sim(n, rep):
    logging.getLogger().setLevel(logging.INFO)
    # Similate the distributions of the top 2 eigenvaules of GOE metrics
    logging.info('Simulating null distribution for testing...')

    nprime = n - 1  # adjust because of centerlization
    GOE_L12_dist = np.zeros((2, rep), dtype='complex_')  # contains L1 and L2

    for i in range(rep):
        main = np.sqrt(2) * np.random.normal(size=(1, nprime))
        off = np.random.normal(size=(nprime, nprime))
        tril = np.tril(off, -1)
        W_n = tril + tril.T
        np.fill_diagonal(W_n, main)

        eigenvalues = LA.eigvals(W_n)
        w = np.flip(np.sort(eigenvalues))
        GOE_L12_dist[:, i] = w[0:2]

    GOE_L12_dist = np.flip(np.sort(GOE_L12_dist, axis=0), axis=0)
    return GOE_L12_dist


class ER:
    def __init__(self):
        pass

    def ERStruct(self, n, path, filename, rep, alpha, Kc=None):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.getLogger().setLevel(logging.INFO)
        # n: Total number of individuals in the study
        # path: The path of data file(s)
        # filename: The name of the data file(s)
        # rep: Number of simulation times for the null distribution
        # alpha: Significance level, can be either a scaler or a vector
        # Kc: A coarse estimate of the top PCs number (set to `floor(n/10)` by default)
        # core_num: Optional, number of CPU cores to be used for parallel computing.

        # Return : K_hat-Estimated number of top PCs
        if Kc == None:
            Kc = int(np.floor(n / 10))

        eigens, p = Eigens(n, path, filename)
        GOE_L12_dist = GOE_L12_sim(n, rep)

        logging.info('Testing...')

        K_tmp = 0
        K_hat = 0
        n_prime = n
        end = eigens.shape[0]
        stats0 = eigens[1:end - 1] / eigens[0:end - 2]
        stats = np.insert(stats0, 0, 0)
        xi_GOE_s = np.zeros((Kc, 1), dtype='complex_')

        while K_tmp < Kc:
            K_tmp = K_tmp + 1
            n_prime = n_prime - 1

            a_p_hat = sum(eigens[K_tmp - 1:end - 1]) / n_prime
            b_p_hat = p / n_prime * (sum(eigens[K_tmp - 1:end - 1] ** 2) / n_prime - a_p_hat ** 2)

            xi_GOE_rep = (GOE_L12_dist[1, :] * np.sqrt(b_p_hat / p) + a_p_hat) / (
                    GOE_L12_dist[0, :] * np.sqrt(b_p_hat / p) + a_p_hat)
            xi_GOE_rep = np.sort(xi_GOE_rep)
            xi_GOE_s[K_tmp - 1] = xi_GOE_rep[int(np.ceil(max(xi_GOE_rep.shape) * alpha)) - 1]

            if K_hat == 0 and stats[K_tmp] > xi_GOE_s[K_tmp - 1]:  # jump above threshold
                K_hat = K_tmp
            elif K_hat != 0 and stats[K_tmp] <= xi_GOE_s[K_hat - 1]:  # fake
                K_hat = 0

        # if output K_hat == 0, then show error message
        if K_hat == 0:
            logging.error('Cannot find valid K_hat <= Kc')
        return K_hat


if __name__ == '__main__':
    obj = ER()
    obj.ERStruct(5, '/Users/eciel/Desktop/y3s2_sum/RA/Xu_YUyang/python_code/data_gene/', ["test11", "test12"], 50, 1e-4)

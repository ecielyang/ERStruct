import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch


def Eigens_cpu(n, path, filename):
    p = 0
    X = torch.zeros((n, n))

    for i in range(len(filename)):
        print('Processing file ', i + 1)

        # Read npy file
        dataset = np.load(path + str(filename[i]) + ".npy")
        dataset = torch.from_numpy(dataset)

        mu = torch.nanmean(dataset, 0)
        mu = torch.reshape(mu, (1, dataset.shape[1]))

        q = (torch.nansum(dataset, 0) + 1) / (2 * n + 2)
        q = torch.reshape(q, (1, dataset.shape[1]))

        M = (dataset - mu) / np.sqrt(2 * q * (1 - q))

        # replace initial missing data and the NaN caused by deviding 0 mean
        M = torch.nan_to_num(M, nan=0)
        p_i = max(M.shape)
        X = X + M @ M.T
        p = p + p_i

    X = 1 / p * X
    eigenvalues = torch.linalg.eigvalsh(X)
    eigens, _ = torch.sort(eigenvalues, descending=True)
    eigens = eigens.detach().numpy()

    return eigens, p


def Eigens_gpu(n, VRAM_available, path0, filename, device):
    device_num = 4.3
    max_memo = VRAM_available / device_num
    p = 0
    X = torch.zeros((n, n)).to(device)

    for i in range(len(filename)):
        print('Processing file ', i + 1)
        file_path = path0 + str(filename[i]) + ".npy"
        dataset = np.load(file_path)
        dataset_size = dataset.size * dataset.itemsize

        if dataset_size < max_memo:
            dataset = torch.from_numpy(dataset).to(device)

            mu = torch.nanmean(dataset, 0)
            mu = torch.reshape(mu, (1, dataset.shape[1]))

            q = (torch.nansum(dataset, 0) + 1) / (2 * n + 2)
            q = torch.reshape(q, (1, dataset.shape[1]))

            M = (dataset - mu) / torch.sqrt(2 * q * (1 - q))

            # replace initial missing data and the NaN caused by deviding 0 mean
            M = torch.nan_to_num(M, nan=0)

            p_i = max(M.shape)
            X = X + M @ M.T
            p = p + p_i
            del dataset, mu, q, M
            torch.cuda.empty_cache()

        else:
            # Split data
            split_num = np.ceil(dataset_size / max_memo)
            files = np.array_split(dataset, split_num, axis=1)
            del dataset

            for j in range(len(files)):

                dataset = torch.from_numpy(files[j]).to(device)

                mu = torch.nanmean(dataset, 0)
                mu = torch.reshape(mu, (1, dataset.shape[1]))

                q = (torch.nansum(dataset, 0) + 1) / (2 * n + 2)
                q = torch.reshape(q, (1, dataset.shape[1]))

                M = (dataset - mu) / torch.sqrt(2 * q * (1 - q))
                # replace initial missing data and the NaN caused by deviding 0 mean
                M = torch.nan_to_num(M, nan=0)

                p_i = max(M.shape)
                X = X + M @ M.T
                p = p + p_i

                del mu, q, M, dataset
                torch.cuda.empty_cache()
            del files

    X = 1 / p * X
    eigenvalues = torch.linalg.eigvalsh(X)
    eigens, _ = torch.sort(eigenvalues, descending=True)
    eigens = eigens.cpu().detach().numpy()

    return eigens, p
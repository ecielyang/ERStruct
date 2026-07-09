Let me look at the paper first to describe the method accurately.Here's a polished, more professional version of the README.

# ERStruct — Official Python Implementation

**ERStruct** is a Python package for inferring the number of top informative principal components (PCs) that capture population structure from whole-genome sequencing genotype data. It implements an eigenvalue-ratio approach designed to handle the ultra-high dimensionality and linkage disequilibrium of modern sequencing datasets, with support for parallel CPU computing and GPU acceleration for large-scale matrix operations.

> 📄 **Reference:** Yang, J., Xu, Y., & Liu, Z. *ERStruct: A Python Package for Inferring the Number of Top Principal Components from Whole Genome Sequencing Data.* bioRxiv, 2022. https://doi.org/10.1101/2022.08.15.503962

## Installation

Install the latest release from PyPI:

```bash
pip install ERStruct
```

### Dependencies

ERStruct requires `numpy`, `torch`, and `joblib`. These are installed automatically with the command above.

## Data Requirements

Input data must satisfy the following conditions:

1. **Format.** Data files must be NumPy arrays in `.npy` format. VCF files can be converted with [`vcfnp`](https://pypi.org/project/vcfnp/), and BGEN files with [`bgen-reader`](https://pypi.org/project/bgen-reader/).
2. **Entries.** The data matrix may contain only the values `0`, `1`, `2`, and/or `NaN` (missing). By default, ERStruct imputes missing values as `0`; users may apply an alternative imputation strategy beforehand.
3. **Shape.** Rows represent individuals and columns represent markers. When multiple data files are provided, all matrices must contain the same number of rows.

## Usage

### API

```python
erstruct(n, path, rep, alpha, cpu_num=1, device_idx="cpu", varm=2e8, Kc=-1)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Total number of individuals in the study. |
| `path` | `str` | Path to the data file(s). |
| `rep` | `int` | Number of simulations used to approximate the null distribution (default `5000`). A value between `2/alpha` and `5/alpha` is recommended. |
| `alpha` | `float` | Significance level, either a scalar or a vector (default `1e-3`). |
| `Kc` | `int` | Coarse estimate of the number of top PCs (default `-1`, which sets `Kc = floor(n/10)` at runtime). |
| `cpu_num` | `int` | Number of CPU cores used for parallel computing (default `1`). |
| `device_idx` | `str` | Computing device, `"cpu"` or `"gpu"` (default `"cpu"`). |
| `varm` | `int` | GPU memory (in bytes) allocated for computation. When `device_idx="gpu"`, increasing `varm` can improve speed (default `2e8`). |

### Examples

Import the algorithm:

```python
from ERStruct import erstruct
```

Download the sample dataset (chromosomes 21 and 22 for 500 individuals from the 1000 Genomes Project sequencing data):

```python
from ERStruct import download_sample
download_sample()
```

Run ERStruct on CPU:

```python
test = erstruct(500, ['chr21.npy', 'chr22.npy'], 1000, 5e-3, cpu_num=1, device_idx="cpu")
K = test.run()
```

Run ERStruct on GPU:

```python
test = erstruct(500, ['chr21.npy', 'chr22.npy'], 1000, 5e-3, device_idx="gpu", varm=2e8)
K = test.run()
```

## Further Reading

- **Package:** [ERStruct: A Python Package for Inferring the Number of Top Principal Components from Whole Genome Sequencing Data](https://www.biorxiv.org/content/10.1101/2022.08.15.503962v2)
- **Method:** [ERStruct: An Eigenvalue Ratio Approach to Inferring Population Structure from Sequencing Data](https://www.researchgate.net/publication/350647012_ERStruct_An_Eigenvalue_Ratio_Approach_to_Inferring_Population_Structure_from_Sequencing_Data)

## Contact

For questions, please contact **eciel@connect.hku.hk**.

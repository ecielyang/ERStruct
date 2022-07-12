# ERstruct - Official Python Implementation

A Python package for inferring the number of top informative PCs that capture population structure based on genotype information.

## Requirements for Data File
Data files must be of .mat format. The data matrix must with 0,1,2 and/or NaN (for missing values) entries only, the rows represent individuals and columns represent markers. If there are more than one data files, the data matrix inside must with the same number of rows.

## Examples
Example data files `test_chr21.py` and `test_chr22.py` can be found on the "sample_data" of [ERStruct GitHub repository](https://github.com/ecielyang/ERStruct).
```
ERStruct(2504, '.', ["test_chr21","test_chr22"], 10000, 1e-4)
```

## Other Details
Please refer to our paper
> *An Eigenvalue Ratio Approach to Inferring Population Structure from Whole Genome Sequencing Data*.
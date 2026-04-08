# CLIPer Dataset

This folder contains the datasets used in the CLIPer project
(*Contrastive Learning for Intrinsically disordered Protein linker classification*).

## Dataset Overview

- `disprot9.5/`: primary source data from DisProt, used for model development.
- `caid3/`: CAID3-related data, used as an external benchmark test set.

## Data Split Policy

To keep evaluation consistent with the CLIPer project design:

- **Training set**: from **DisProt**
- **Validation set**: from **DisProt**
- **Test set**: from **CAID3**

This setup is intended to evaluate generalization on a competition-style test distribution while training and tuning on curated DisProt annotations.

## Notes

- File names and intermediate artifacts are preserved according to the original project workflow.
- Additional preprocessing details and training scripts are documented in the repository root `README.md` and in the `CLAPE-SMB/` baseline folder.

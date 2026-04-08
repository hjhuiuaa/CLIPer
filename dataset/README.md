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

## File-level Notes

### `disprot9.5/`

- `disprot_202312_label.fasta`: DisProt sequence-level labels for **disorder** residues (binary string per sequence).
- `disprot_202312_linker_label.fasta`: DisProt sequence-level labels for **linker** residues (binary string per sequence).
- `error.txt`: cases where a disorder region is only partially covered by linker labels (records sequences/regions needing manual review).
- `merged1.fasta`: merged annotation view used during preprocessing; each record typically contains sequence + multiple label lines aligned by residue index.
- `prostt5_input.fasta`: sequence-only FASTA prepared as input for ProstT5 embedding extraction (no label lines).
- `esm_input.fasta`, `esm_input_train.fasta`, `esm_input_valid.fasta`: sequence-only FASTA files prepared for ESM-style embedding/inference workflow.
- `train.fasta`, `valid.fasta`: legacy split files from an earlier experiment setup (kept locally for traceability; not part of the active split policy).

### `caid3/`

- `linker.fasta`: CAID3 linker benchmark labels (used as test target in CLIPer).
- `binding.fasta`: CAID3 binding-site labels.
- `binding_idr.fasta`: CAID3 binding labels mapped to IDR-focused representation (contains alignment-style placeholders in some records).
- `disorder_nox.fasta`: CAID3 disorder labels in a sequence set with ambiguous `X` residues removed/normalized.
- `disorder_pdb.fasta`: CAID3 disorder labels in a PDB-mapped representation (contains alignment/padding placeholders for coordinate mapping).
- `predictions.zip`: archived CAID prediction outputs from prior runs.

## Notes

- Additional preprocessing details and training scripts are documented in the repository root `README.md` and in the `CLAPE-SMB/` baseline folder.

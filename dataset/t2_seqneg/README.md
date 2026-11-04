# T2 sequence-level negatives

Sequence-level 1:1 negatives (no residue undersampling).

1. Keep every DisProt protein that has at least one linker residue (after
   dropping CAID3 holdout and error.txt). All residues on those chains stay
   labeled 0/1/2/-; nothing is dropped to '-' by sampling.

2. Linker.fasta and disorder.fasta are the same 2896 DisProt IDs. Extra
   negatives are proteins with zero linker residues. Sample N_extra = N_linker
   sequences, preferring chains that contain both ordered (0) and other-IDR (1).

3. Split linker proteins 80/20 (seed 42) and extra proteins 80/20 independently
   so train/val each stay sequence-balanced.

4. Labels:
     0 = ordered (non-disorder); disorder-track '-' is also 0
     1 = other IDR (disorder, not linker)
     2 = linker
     - = unused here (no residue undersample)

5. CAID3 test is still the 31 linker sequences, labeled from linker + disorder-PDB
   (disorder '-' counted as ordered 0).

## Files

- `train.fasta` / `val.fasta`: 3-line FASTA with labels {0,1,2,-}, no residue undersample
- `train_val.fasta`: concatenation used by `cliper train`
- `test_caid3.fasta`: CAID3 31 sequences labeled 0/1/2 from linker + disorder-PDB
- `sequences_extract.fasta`: 2-line sequences for ProstT5 extract (train+val+test)
- `selection_stats.json`: counts and exclusion details

Split manifest: `artifacts/splits/t2_seqneg_split_seed42.json`

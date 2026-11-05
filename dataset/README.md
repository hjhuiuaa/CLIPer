# CLIPer dataset

Sequences used by the paper method. Labels are `0` ordered, `1` other disordered residue, and `2` linker.

## Paper split

`t2_seqneg/` is the split used for training, validation, and the two evaluation scenarios:

- `train.fasta`, `val.fasta`, `train_val.fasta`
- `test_caid3.fasta` — scenario 1, 31 CAID3 linker proteins
- `test_scene2_pdb62.fasta` — scenario 2, linker versus other disordered residues
- `error.txt` — proteins excluded from training and validation

Rebuild the training split and both test sets with `python scripts/prepare_dataset.py`.

## Source files for that rebuild

- `disprot_202312_linker_label.fasta` — DisProt linker labels
- `disprot_202312_label.fasta` — DisProt disorder labels
- `error.txt`
- `linker.fasta` — CAID3 linker reference
- `disorder_pdb.fasta` — CAID3 disorder-PDB reference

# CLIPer

CLIPer predicts disordered linker residues and separates them from other disordered residues. The public repository is [https://github.com/hjhuiuaa/CLIPer](https://github.com/hjhuiuaa/CLIPer).

The released model is a frozen ProstT5 encoder followed by one convolutional layer and one bidirectional GRU. Training uses binary cross-entropy on the linker logit plus ternary supervised contrastive learning (weight 0.2). The score for each residue is the linker probability. A residue is called a linker when that probability is at least 0.946. This threshold is the count-matched operating point on the training split: it is the score at which the number of predicted linker residues equals the number of labeled linker residues in `dataset/t2_seqneg/train.fasta`.

## Requirements

- Python 3.10 or later
- A CPU is enough to inspect the checkpoint. Predicting a sequence also downloads ProstT5 (`Rostlab/ProstT5` on Hugging Face, several gigabytes) and is much faster on a GPU.

```bash
pip install -r requirements.txt
```

## 1. Inspect the released classifier

This checks the architecture stored in the paper checkpoint. It does not download ProstT5.

```bash
python scripts/run_paper_example.py --check-only
```

The printed configuration should show one convolution with 128 channels, a bidirectional GRU with hidden size 64 and one layer, `linker_class` 2, and ternary contrastive weight 0.2. The checkpoint file is saved to `artifacts/paper/best.pt`. The same file is [the paper-supcon release](https://github.com/hjhuiuaa/CLIPer/releases/download/paper-supcon/best.pt).

## 2. Predict the example protein

`examples/example.fasta` is DP04381, the shortest protein in the Scene 1 test set (147 residues). The first run downloads ProstT5.

```bash
python scripts/run_paper_example.py --device cpu
```

Use `--device cuda` when a GPU is available. The command writes `artifacts/paper_example/linker/DP04381.caid` and `artifacts/paper_example/timings.csv`. Each `.caid` row is the residue, its linker probability, and the call at threshold 0.946.

To score another FASTA file:

```bash
python scripts/run_paper_example.py --fasta your_sequences.fasta --output-dir artifacts/your_run --device cuda
```

The FASTA file has two lines per protein: a header and the sequence. Ambiguous residues are accepted by the sequence cleaner used at prediction time.

## Data and the two evaluation scenes

Training and validation sequences are in `dataset/t2_seqneg/`. Labels are `0` ordered, `1` other disordered residue, `2` linker. The CAID3 linker proteins and the proteins listed in `dataset/t2_seqneg/error.txt` are excluded from training and validation.

| Split | File | What is scored |
| --- | --- | --- |
| Train | `dataset/t2_seqneg/train.fasta` | Used to fit the model and to set the 0.946 threshold |
| Validation | `dataset/t2_seqneg/val.fasta` | Model selection by Scene 1 AUROC |
| Scene 1 test | `dataset/t2_seqneg/test_caid3.fasta` | Linker versus every other residue, 31 CAID3 proteins |
| Scene 2 test | `dataset/t2_seqneg/test_scene2_pdb62.fasta` | Linker versus other disordered residues; ordered residues are not scored |

Scene 1 and Scene 2 use the same residue probabilities. They differ only in which residues enter the metric. AUROC and average precision do not use the 0.946 threshold.

Rebuild the split from the DisProt inputs with:

```bash
python scripts/build_t2_seqneg_split.py
```

Recompute the count-matched threshold from saved full-sequence embeddings with:

```bash
python scripts/count_matched_threshold.py
```

That script expects the training embeddings produced with the released checkpoint. It does not use the validation proteins.

## Training

The paper configuration is `configs/ablation/shared_supcon.yaml`. ProstT5 stays frozen. The trainable pieces are the convolution, the GRU, the classifier, and the contrastive projection.

```bash
python -m cliper.cli train --config configs/ablation/shared_supcon.yaml
```

Training reads `dataset/t2_seqneg/train_val.fasta` and writes under `artifacts/runs/ablation_shared_20261031/supcon`. A GPU is required for a full training run.

## Layout

- `cliper/` — model, data, training, and prediction
- `configs/ablation/shared_supcon.yaml` — paper training configuration
- `dataset/t2_seqneg/` — sequences used in the paper
- `examples/example.fasta` — one Scene 1 protein for a prediction check
- `scripts/run_paper_example.py` — download the checkpoint and run the check above

Notes from earlier internal experiments are in `docs/development_history.md`. Those stages are not the model reported in the paper.

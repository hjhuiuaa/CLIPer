# Disorder — residue-level binary classification

Standalone workflow under `disorder/` for **intrinsically disordered region (IDR)** prediction with ProstT5.

Two modes:

1. **Online** — `train` / `eval` run ProstT5 + head end-to-end.
2. **Offline features** — `extract_features` or `extract_sequence` write `.resfeat.txt`, then `train_features` / `eval_features` train only the head.

Entry point: `python -m disorder <command>`

---

## Single-sequence embedding (`extract_sequence`)

Extract **one protein** to a single `<protein_id>.resfeat.txt` (one line per residue, `[L, hidden_size]`).

| Sequence length | Encoding |
| --- | --- |
| `L <= 1024` | One full-sequence ProstT5 forward pass |
| `L > 1024` | Non-overlapping 1024-residue windows, concatenated to `[L, D]` (same final layout as `reextract_merge_nonoverlap`) |

### By id + sequence

```bash
python -m disorder extract_sequence \
  --protein-id P04637 \
  --sequence MKTAYIAKQRQISFVK... \
  --output-dir disorder/prostt5_features/one_off \
  --backbone /path/to/Rostlab-ProstT5 \
  --device cuda \
  --batch-size 1
```

### By 2-line FASTA (one record)

```bash
python -m disorder extract_sequence \
  --fasta disorder/data/one_protein.fasta \
  --output-dir disorder/prostt5_features/one_off \
  --backbone /path/to/Rostlab-ProstT5 \
  --device cpu
```

Alternative module entry:

```bash
python -m disorder.extract_sequence_embedding --help
```

Outputs:

- `{output_dir}/{protein_id}.resfeat.txt`
- `{output_dir}/manifest.json`

Implementation: `sequence_embedding.py` (core), `extract_sequence_embedding.py` (CLI).

---

## Batch embedding (`extract_features`)

2-line FASTA with many proteins → one `.resfeat.txt` per id (whole sequence encoded in one pass):

```bash
python -m disorder extract_features \
  --fasta disorder/data/train.fasta \
  --output-dir disorder/prostt5_features/train \
  --backbone /path/to/Rostlab-ProstT5 \
  --batch-size 1 --device cuda
```

For historical long-sequence pipelines (chunk → merge), see:

- `extract_features_chunked.py` — overlapping 1024 windows → `id_1.resfeat.txt`, `id_2.resfeat.txt`, …
- `chunk_existing_features.py` — re-extract long ids from FASTA into chunk files
- `reextract_merge_nonoverlap.py` — merge chunk stems back to one `id.resfeat.txt`

**Prefer `extract_sequence` for new one-off or long-sequence work** — it skips intermediate chunk files.

---

## Train on precomputed features

```bash
python -m disorder train_features \
  --config disorder/configs/disorder_feature_stage3_mlp5_example.yaml
```

Example config paths: `disorder/configs/disorder_feature_stage3_mlp5_example.yaml`

At train/eval time, the classifier reads the **full** `.resfeat.txt` and applies **1024 + overlap 256** sliding windows on embeddings (logit merge), independent of how embeddings were encoded.

---

## Other commands

| Command | Purpose |
| --- | --- |
| `prepare_data` / `prepare_split` | Train/val split manifests |
| `train` / `eval` | Online ProstT5 + head |
| `eval_features` | Evaluate a `train_features` checkpoint |

Place local FASTA splits under `disorder/data/` (gitignored except `__init__.py`).

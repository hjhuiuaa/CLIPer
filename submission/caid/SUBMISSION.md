# CLIPer CAID4 Submission Guide (Stage4)

Model: **stage4 ProstT5 CNN concat_window t1_regularized**  
Task: **linker** residue prediction (DisProt linker annotations)

Deadline reference: CAID4 submission closes **2026-05-31** ([CAID challenge page](https://caid.idpcentral.org/challenge)).

---

## What this package does

1. Reads a **2-line FASTA** (one protein id + sequence per record).
2. Loads **precomputed ProstT5 residue embeddings** (`.npy`, `.h5`, or CLIPer `.resfeat.txt`).
3. Runs the **frozen stage4 CNN classification head** on CPU.
4. Writes CAID outputs:
   - `{output_dir}/{flavor}/{protein_id}.caid`
   - `{output_dir}/timings.csv`

ProstT5 weights are **not** bundled. CAID organizers mount embeddings at runtime.

---

## 1. Prepare your checkpoint

Copy the best stage4 checkpoint from training:

```bash
# On training server (example path)
cp artifacts/runs/stage4_prostt5_cnn_concat_window_t1_regularized/exp0001/checkpoints/best.pt \
   submission/caid/models/best.pt
```

The checkpoint must contain:
- `model_state` with `classifier.*` weights
- `config.local_context` with `radius: 1`, `concat_window`
- `threshold` used for binary labels in `.caid` files

---

## 2. Local smoke test (before Docker)

Export or copy ProstT5 embeddings for a few proteins (shape `[L, 1024]` per protein):

```bash
python -m cliper.cli predict \
  --checkpoint artifacts/runs/stage4_prostt5_cnn_concat_window_t1_regularized/exp0001/checkpoints/best.pt \
  --fasta dataset/linker.fasta \
  --embeddings-dir /path/to/prostt5_embeddings \
  --output-dir artifacts/caid_smoke \
  --device cpu \
  --num-threads 4
```

Note: `dataset/linker.fasta` is a 3-line labeled file used internally for benchmarking. For CAID targets you will receive a standard 2-line FASTA. The `predict` command accepts 2-line FASTA only.

Convert labeled FASTA to 2-line for local tests:

```bash
awk 'NR%3==1 || NR%3==2' dataset/linker.fasta > /tmp/linker_seqonly.fasta
```

Embedding file naming (first match wins):

| FASTA header | Embedding file examples |
|---|---|
| `>P04637` | `embeddings/P04637.npy`, `embeddings/P04637.h5` |

---

## 3. Build Docker image

From repository root:

```bash
docker build -f submission/caid/Dockerfile -t cliper-caid:linker-stage4 .
```

Run:

```bash
docker run --rm \
  -v /path/to/targets.fasta:/input/targets.fasta:ro \
  -v /path/to/embeddings:/input/embeddings:ro \
  -v /path/to/best.pt:/models/best.pt:ro \
  -v /path/to/output:/output \
  -e NUM_THREADS=4 \
  cliper-caid:linker-stage4
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT` | `/models/best.pt` | Classifier checkpoint |
| `FASTA` | `/input/targets.fasta` | Input sequences |
| `EMBEDDINGS_DIR` | `/input/embeddings` | Per-protein embedding files |
| `OUTPUT_DIR` | `/output` | CAID output root |
| `FLAVOR` | `linker` | Output subdirectory name |
| `NUM_THREADS` | `4` | CPU threads (max 24) |
| `THRESHOLD` | _(from checkpoint)_ | Optional binary threshold override |

---

## 4. Output format

**`linker/PROTEIN.caid`**

```
>PROTEIN
1    M    0.892    1
2    E    0.813    0
...
```

**`timings.csv`**

```
# Running CLIPer linker predictor, started ...
sequence,milliseconds
PROTEIN,1827
```

---

## 5. CAID compliance checklist

| Requirement | Status |
|---|---|
| CPU-only inference | Yes (`--device cpu`, forced in Docker) |
| CLI, no interaction | Yes (`python -m cliper.cli predict`) |
| No internet at runtime | Yes (no model download) |
| ≤ 48 GB RAM | Head-only; verify on longest target |
| ≤ 24 threads | `--num-threads` capped at 24 |
| < 6 h / sequence | Profile longest target on CPU before submit |
| Precomputed PLM embeddings | Yes (`--embeddings-dir`) |
| Ambiguous residues (B,Z,J,U,O,X) | Sanitized to X; still outputs scores |
| Install/run instructions | This document |
| Dockerfile | `submission/caid/Dockerfile` |

**Confirm with organizers:** whether linker predictions use flavor directory `linker` or another CAID4 category name.

---

## 6. Submit to CAID

1. Push Docker image to a public registry (Docker Hub / GitHub Container Registry), or provide Dockerfile + build instructions.
2. Complete the **online submission form** on [caid.idpcentral.org](https://caid.idpcentral.org/challenge).
3. Document embedding mount path expected by your predictor (see `EMBEDDINGS_DIR` above).

---

## 7. Your next actions (checklist)

- [ ] Copy `best.pt` from your best multi-seed run (t1_regularized) into `submission/caid/models/`
- [ ] Ensure CAID will provide **ProstT5 embeddings** with hidden size **1024** and residue-aligned rows
- [ ] Run local `predict` smoke test on 2–3 proteins
- [ ] CPU-profile the **longest** sequence; confirm runtime < 6 hours
- [ ] Build and run Docker on a clean Linux machine (no GPU)
- [ ] Email organizers if flavor name should be `linker` vs `binding`/`disorder`
- [ ] Fill CAID online form before **2026-05-31**

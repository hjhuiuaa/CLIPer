# CLIPer-LinkCRNN — Source & Usage Guide (CAID4)

**Method name:** CLIPer-LinkCRNN  
**Task:** DisProt **linker** residue prediction (CAID flavor: `linker`)  
**Submitted checkpoint:** `stage7_prostt5_crnn_expl_a1_drop040` / `exp0005`  
**Docker image (recommended):** `crpi-fyyyvm7srevvnsix.cn-beijing.personal.cr.aliyuncs.com/cliper-caid4/cliper-linkcrnn:latest`

This archive contains **only the Stage 7 linker (CRNN) inference stack**. It does **not** include other training stages, the full `disorder/` IDR project, or training pipelines.

---

## 1. Method overview (technical)

### 1.1 Biological task

**CLIPer** predicts **linker residues** in intrinsically disordered proteins. Linkers are short disordered segments connecting folded domains. In CAID, the `linker` flavor is evaluated separately from general `disorder` or `binding`.

### 1.2 Backbone: ProstT5 (frozen)

- **Model:** [Rostlab/ProstT5](https://huggingface.co/Rostlab/ProstT5) protein language model.
- **Usage:** ProstT5 is **frozen**; we only use its per-residue embeddings (`hidden_size = 1024`).
- **Input:** amino-acid sequence (uppercase; ambiguous residues mapped to `X` so prediction never crashes).
- **Output:** one matrix per protein, shape **`[L, 1024]`** where `L` = sequence length.

### 1.3 Training data

- **Primary labels:** DisProt linker annotations (`disprot_202312_linker_label.fasta`, 3-line FASTA: id / sequence / linker labels).
- **Holdout:** CAID3 linker benchmark sequences are excluded from train/val splits.
- **Training objective:** binary linker vs non-linker (labels `0`/`1`); backbone not fine-tuned.

### 1.4 Feature pipeline (two stages)

**Stage A — ProstT5 embedding (encoding)**

| Case | Policy |
|------|--------|
| `L ≤ 1024` | Single full-sequence ProstT5 forward pass |
| `L > 1024` | **Non-overlapping** windows of 1024 residues; window embeddings concatenated along the residue axis → still `[L, 1024]` |

**Stage B — Classifier on embeddings (this submission)**

1. **Local context (`concat_window`, radius=1):** for each residue, concatenate embeddings of neighbors `[i-1, i, i+1]` → **3072-d** per residue (3 × 1024).
2. **Sliding evaluation windows:** for prediction, long sequences use overlapping windows (`window_size=1024`, `stride=512`, `top_k_heuristic=4`); window logits are merged back to full-length scores.
3. **CLIPer-LinkCRNN head:**
   - **CNN front-end:** 2 dilated 1D conv layers (channels 256→256, kernel 3, dilations 1, 2) for local motifs.
   - **RNN back-end:** **2-layer bidirectional GRU** (hidden 256) over the sequence to model **contiguous linker segments** and suppress isolated false positives.
4. **Decision:** sigmoid → per-residue probability; binary label via validation-tuned threshold (stored in checkpoint).

### 1.5 What is *not* in this bundle

- `disorder/` training scripts for general IDR prediction (separate research line).
- Stage 1–6 linker experiments, contrastive training code, or `cliper/pipeline.py` training loop.
- ProstT5 weights (~11 GB) — download separately (see §4).

---

## 2. Directory layout

```
CLIPer-LinkCRNN-source/
├── README.md                          ← this file
├── requirements-caid-full.txt         ← Python dependencies (CPU inference + ProstT5)
├── configs/
│   └── caid_stage7_predict.yaml       ← reference hyperparameters (runtime reads checkpoint)
├── models/
│   ├── checkpoint/
│   │   └── best.pt                    ← stripped CRNN classifier head only
│   └── Rostlab-ProstT5/
│       └── DOWNLOAD.md                ← how to obtain ProstT5 weights
├── cliper/                            ← linker CAID CLI & model code
│   ├── cli.py                         ← entry: predict, extract_embeddings
│   ├── caid_predict.py                ← linker inference from embeddings
│   ├── caid_io.py                     ← FASTA / embedding / .caid I/O
│   ├── extract_embeddings.py          ← ProstT5 embedding export
│   ├── modeling.py                    ← ProstT5 loader + CNN/CRNN heads
│   ├── windowing.py                   ← eval window starts & logit merge
│   └── data.py                        ← FASTA parsing helpers
├── disorder/                          ← minimal shared libs (NOT full IDR project)
│   ├── feature_modeling.py            ← DisorderFeatureClassifier (CRNN head)
│   ├── feature_io.py                  ← .resfeat.txt read/write
│   ├── sequence_embedding.py          ← ProstT5 encode (used by extract)
│   ├── fasta_parsing.py               ← 2-line FASTA helpers
│   └── data/__init__.py               ← re-exports parse_two_line_fasta
├── scripts/
│   └── strip_classifier_checkpoint.py ← utility: shrink best.pt for distribution
└── submission/caid/
    ├── extract_embeddings.sh          ← shell wrapper (MODE=extract)
    ├── predict.sh                     ← shell wrapper (MODE=predict)
    ├── entrypoint.sh                  ← Docker entrypoint
    ├── Dockerfile.full                ← rebuild full Docker image
    └── README-CLIPer-LinkCRNN.md      ← short Docker-focused readme
```

---

## 3. File reference (inputs / outputs)

### 3.1 CLI entry — `cliper/cli.py`

| Command | Purpose |
|---------|---------|
| `extract_embeddings` | Run ProstT5 → per-protein embedding files |
| `predict` | Run CRNN head on precomputed embeddings → CAID outputs |

Other commands (`train`, `eval`, `prepare_data`) exist in the parser but require training code **not shipped** in this bundle.

---

### 3.2 `cliper/extract_embeddings.py`

**Purpose:** Export ProstT5 residue embeddings for CAID / linker predict.

| | |
|--|--|
| **Input** | `--fasta` : 2-line FASTA (`>protein_id` + sequence); multiple records allowed |
| | `--backbone` : path to local ProstT5 dir or Hub id `Rostlab/ProstT5` |
| | `--window-size` : default `1024` |
| | `--format` : `npy` (recommended) or `resfeat` |
| | `--device` : `cpu` (CAID) or `cuda` |
| **Output** | `--output-dir` : one file per protein |
| | `{protein_id}.npy` : NumPy array `[L, 1024]` |
| | or `{protein_id}.resfeat.txt` : text matrix, one line per residue |
| | `manifest.json` : metadata (protein count, hidden size, paths) |

**Example:**

```bash
python -m cliper.cli extract_embeddings \
  --fasta targets.fasta \
  --output-dir embeddings/ \
  --backbone models/Rostlab-ProstT5 \
  --window-size 1024 \
  --format npy \
  --device cpu \
  --batch-size 1
```

---

### 3.3 `cliper/caid_predict.py`

**Purpose:** Linker prediction from precomputed embeddings (no ProstT5 in this step).

| | |
|--|--|
| **Input** | `--checkpoint` : `models/checkpoint/best.pt` (classifier head + config + threshold) |
| | `--fasta` : 2-line FASTA (sequence only, no labels) |
| | `--embeddings-dir` : directory with `{protein_id}.npy` (or `.h5` / `.resfeat.txt`) |
| | `--output-dir` : writable output directory |
| | `--flavor` : default `linker` |
| | `--num-threads` : CPU threads (max 24) |
| **Output** | `{output_dir}/linker/{protein_id}.caid` |
| | `{output_dir}/timings.csv` |
| | `{output_dir}/predict_summary.json` |

**`.caid` format (per protein):**

```
>DP03804
1    M    0.123    0
2    E    0.456    1
...
```

Columns: `position`, `residue`, `probability (0–1)`, `binary (0/1)`.

**`timings.csv`:**

```csv
sequence,milliseconds
DP03804,406
```

**Example:**

```bash
python -m cliper.cli predict \
  --checkpoint models/checkpoint/best.pt \
  --fasta targets.fasta \
  --embeddings-dir embeddings/ \
  --output-dir output/ \
  --flavor linker \
  --device cpu \
  --num-threads 4
```

---

### 3.4 `cliper/caid_io.py`

| Function | Input | Output |
|----------|-------|--------|
| `parse_caid_fasta` | 2-line FASTA path | `list[(protein_id, sequence)]` |
| `read_residue_embedding` | embeddings dir + protein_id | `torch.Tensor [L, D]` |
| `write_caid_file` | probabilities + threshold | `.caid` file |
| `write_timings_csv` | list of (id, ms) | `timings.csv` |

Ambiguous FASTA chars (`B,Z,J,U,O,X`) are sanitized so inference does not crash.

---

### 3.5 `cliper/modeling.py`

| Component | Role |
|-----------|------|
| `load_backbone_and_tokenizer` | Load ProstT5 from local dir (offline) |
| `encode_sequences` | Tokenize + forward ProstT5 |
| `CNNClassifierHead` | 1D CNN (used inside CRNN front-end) |
| `CRNNClassifierHead` | CNN → bidirectional GRU → per-residue logit |

---

### 3.6 `cliper/windowing.py`

| Function | Role |
|----------|------|
| `build_eval_window_starts` | Choose window start positions for long sequences |
| `merge_window_logits` | Merge overlapping window predictions to length `L` |
| `normalize_sequence` | Uppercase + map rare residues |

---

### 3.7 `disorder/feature_modeling.py`

**Purpose:** `DisorderFeatureClassifier` — wraps the classifier head when input is **precomputed** `[B, L, D]` embeddings (not raw sequence). Supports `type: crnn` for this submission.

**Note:** The `disorder/` folder name is historical. Only the files listed in §2 are included; this is **not** the separate IDR disorder prediction project.

---

### 3.8 `disorder/sequence_embedding.py` + `disorder/feature_io.py`

Used internally by `extract_embeddings` for ProstT5 encoding and file I/O. Same long-sequence windowing policy as §1.4.

---

### 3.9 Shell scripts (`submission/caid/`)

| Script | Env vars | Calls |
|--------|----------|-------|
| `extract_embeddings.sh` | `FASTA`, `EMBEDDINGS_DIR`, `PROSTT5_MODEL`, `NUM_THREADS` | `cli extract_embeddings` |
| `predict.sh` | `CHECKPOINT`, `FASTA`, `EMBEDDINGS_DIR`, `OUTPUT_DIR`, `FLAVOR` | `cli predict` |
| `entrypoint.sh` | `MODE=extract` or `MODE=predict` | routes to above |

---

### 3.10 `models/checkpoint/best.pt`

Stripped checkpoint containing **only** `classifier.*` weights plus:

- `config` (architecture, window settings, `local_context`)
- `threshold` (validation-tuned decision threshold)

ProstT5 weights are **not** inside `best.pt`.

---

## 4. Setup (from source, without Docker)

### 4.1 Install dependencies

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-caid-full.txt
export PYTHONPATH=/path/to/CLIPer-LinkCRNN-source
```

### 4.2 Download ProstT5 (~11 GB)

```bash
pip install huggingface_hub
huggingface-cli download Rostlab/ProstT5 --local-dir models/Rostlab-ProstT5
```

Required files: `pytorch_model.bin`, `config.json`, `spiece.model`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`.

### 4.3 Two-step workflow

```bash
# Step 1: embeddings
python -m cliper.cli extract_embeddings \
  --fasta targets.fasta \
  --output-dir embeddings/ \
  --backbone models/Rostlab-ProstT5 \
  --format npy --device cpu

# Step 2: predict
python -m cliper.cli predict \
  --checkpoint models/checkpoint/best.pt \
  --fasta targets.fasta \
  --embeddings-dir embeddings/ \
  --output-dir output/ \
  --device cpu
```

---

## 5. Docker workflow (recommended for CAID)

Organizers typically **`docker pull`** the image (not an HTTP file download):

```bash
docker pull crpi-fyyyvm7srevvnsix.cn-beijing.personal.cr.aliyuncs.com/cliper-caid4/cliper-linkcrnn:latest
```

```bash
# Extract embeddings
docker run --rm -e MODE=extract -e NUM_THREADS=4 \
  -v /path/to/targets.fasta:/input/targets.fasta:ro \
  -v /path/to/embeddings:/output/embeddings \
  crpi-fyyyvm7srevvnsix.cn-beijing.personal.cr.aliyuncs.com/cliper-caid4/cliper-linkcrnn:latest

# Predict
docker run --rm -e MODE=predict -e NUM_THREADS=4 \
  -v /path/to/targets.fasta:/input/targets.fasta:ro \
  -v /path/to/embeddings:/input/embeddings:ro \
  -v /path/to/output:/output \
  crpi-fyyyvm7srevvnsix.cn-beijing.personal.cr.aliyuncs.com/cliper-caid4/cliper-linkcrnn:latest
```

Image includes ProstT5, code, and `best.pt`. See `submission/caid/README-CLIPer-LinkCRNN.md` inside the image (`/app/README.md`).

---

## 6. CAID compliance

| Requirement | Status |
|-------------|--------|
| CPU-only | Yes (`--device cpu`) |
| CLI, no interaction | Yes |
| Offline | Yes (local ProstT5 + bundled deps) |
| ≤ 48 GB RAM | Yes |
| ≤ 24 threads | Yes (`NUM_THREADS`) |
| < 6 h / sequence | Yes (seconds per protein on CPU) |

---

## 7. Contact

Nankai University — CLIPer team. For CAID logistics: caid@idpcentral.org.

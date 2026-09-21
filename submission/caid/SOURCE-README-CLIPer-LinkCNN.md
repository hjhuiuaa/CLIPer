# CLIPer-LinkCNN — Source & Usage Guide (CAID4)

**Method name:** CLIPer-LinkCNN  
**Task:** DisProt **linker** residue prediction (CAID flavor: `linker`)  
**Submitted checkpoint:** `stage4_prostt5_cnn_concat_window_t1` / `exp0001`  
**Docker image (recommended):** `crpi-fyyyvm7srevvnsix.cn-beijing.personal.cr.aliyuncs.com/cliper-caid4/cliper-linkcnn:latest`

**Relation to CLIPer-LinkCRNN:** Same CLIPer linker pipeline (frozen ProstT5, DisProt training, `concat_window`, eval windows 1024/512). **Only the classifier head differs:** LinkCNN uses a **pure 1D CNN**; LinkCRNN adds a bidirectional GRU after CNN. This archive contains **Stage 4 CNN inference code only** — not a full training pipeline.

---

## 1. Method overview

- **Backbone:** Rostlab/ProstT5 (frozen), embeddings `[L, 1024]`
- **Local context:** `concat_window`, radius=1 → 3072-d per residue
- **Head:** 3-layer dilated **CNN** (256→256→256, kernel 3, dilations 1,2,4)
- **Long sequences:** non-overlapping ProstT5 windows at encode; overlapping eval windows at predict
- **Not included:** training data, `cliper/pipeline.py`, full `disorder/` IDR project, ProstT5 weights (~11 GB)

---

## 2. Directory layout

Same structure as CLIPer-LinkCRNN source bundle, with `configs/caid_stage4_predict.yaml` and CNN checkpoint reference.

Key files: `cliper/caid_predict.py`, `cliper/extract_embeddings.py`, `cliper/modeling.py` (CNN head), `disorder/feature_modeling.py`, `submission/caid/*.sh`.

---

## 3. Run from Docker (primary for CAID)

```bash
docker pull crpi-fyyyvm7srevvnsix.cn-beijing.personal.cr.aliyuncs.com/cliper-caid4/cliper-linkcnn:latest

docker run --rm -e MODE=extract -e NUM_THREADS=4 \
  -v /path/to/targets.fasta:/input/targets.fasta:ro \
  -v /path/to/embeddings:/output/embeddings \
  crpi-fyyyvm7srevvnsix.cn-beijing.personal.cr.aliyuncs.com/cliper-caid4/cliper-linkcnn:latest

docker run --rm -e MODE=predict -e NUM_THREADS=4 \
  -v /path/to/targets.fasta:/input/targets.fasta:ro \
  -v /path/to/embeddings:/input/embeddings:ro \
  -v /path/to/output:/output \
  crpi-fyyyvm7srevvnsix.cn-beijing.personal.cr.aliyuncs.com/cliper-caid4/cliper-linkcnn:latest
```

Outputs: `linker/{id}.caid`, `timings.csv`.

---

## 4. Optional source setup

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-caid-full.txt
export PYTHONPATH=/path/to/CLIPer-LinkCNN-source
# Download ProstT5: see models/Rostlab-ProstT5/DOWNLOAD.md
python -m cliper.cli extract_embeddings ...
python -m cliper.cli predict --checkpoint models/checkpoint/best.pt ...
```

See CLIPer-LinkCRNN SOURCE-README for full I/O tables (identical workflow, different head in checkpoint).

---

## 5. CAID compliance

CPU-only, offline, CLI, ≤48 GB RAM, ≤24 threads, per-sequence runtime ≪ 6 h.

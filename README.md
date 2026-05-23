# CLIPer
**CLIPer** stands for **Contrastive Learning for Intrinsically disordered Protein linker classification**.

Stage 1 provides a reproducible residue-level binary classifier pipeline for linker prediction:
- **Task:** residue-level linker classification (`0` non-linker, `1` linker)
- **Backbone:** pretrained **ProstT5** (frozen in v1)
- **Training source:** DisProt (`dataset/disprot_202312_linker_label.fasta`)
- **Holdout benchmark:** CAID3 (`dataset/linker.fasta`) used as strict final test only

Stage 2 adds supervised contrastive learning on top of the same residue-level binary task:
- **Objective:** `BCEWithLogits + SupCon` (joint loss)
- **Training mode:** from scratch joint training (no Stage 1 checkpoint required)
- **Backbone policy:** keep ProstT5 frozen by default, train classifier + projection head

Stage 3 validates pure architecture gain with a deeper classification head:
- **Objective:** `BCEWithLogitsLoss` only (no contrastive term)
- **Head options:** `classifier_head.type=mlp3` / `mlp5` / `mlp12` / `transformer`
- **MLP5 layout:** `hidden -> 1024 -> 256 -> 128 -> 64 -> 1`
- **MLP12 layout:** `hidden -> 1024 -> 1024 -> 768 -> 768 -> 512 -> 512 -> 256 -> 256 -> 128 -> 128 -> 64 -> 1`
- **Transformer layout:** residue-wise encoder (`num_layers=2`, `num_heads=4`, `ffn_dim=2048`) + per-residue linear classifier
- **Hidden blocks:** each hidden layer uses `ReLU + LayerNorm + Dropout(0.3)`
- **Backbone policy:** keep ProstT5 frozen; train classification head only

Stage 4 extends Stage 3 with local residue-context concatenation and alternative heads:
- **Objective:** `BCEWithLogitsLoss` only (contrastive forced off, same as Stage 3)
- **Local context:** `local_context.mode=concat_window` concatenates neighboring residue embeddings before the head
- **Head options:** `mlp3` / `mlp5` / `mlp12` / `transformer` / `cnn`
- **CNN head:** `conv_channels + dilations` Conv1d stack for residue-wise logits

Stage 5 extends Stage 4 with native PROSITE motif special tokens:
- **Objective:** `BCEWithLogitsLoss` only (contrastive forced off, same as Stage 3/4)
- **Backbone policy:** reuse ProstT5 encoder with a saved tokenizer extended from the base tokenizer
- **Motif source:** all `PATTERN` entries from `prosite.dat`; `MATRIX` entries are ignored
- **Alignment:** greedy longest-match span replacement, then broadcast token outputs back to residue-level logits

Stage 6 fuses ordinary ProstT5 tokenization with PROSITE special-token tokenization:
- **Objective:** two-branch weighted BCE, `0.5 * plain_bce + 0.5 * special_token_bce`
- **Backbone policy:** two independent ProstT5 encoder/classifier branches in one checkpoint
- **Fusion:** validation, CAID3, and `eval` use `0.5 * plain_logits + 0.5 * special_logits`
- **Default base:** Stage 4 T1 regularized CNN/local-context hyperparameters, plus Stage 5 PROSITE tokenizer assets

## Project Status — CAID4 Submission (2026-05)

**GitHub `main` is synced at commit `7156d3e` (`feat: add CAID stage4 predict CLI and submission bundle`).**

### What is done (code + docs)

| Item | Status | Location |
| --- | --- | --- |
| CAID4 predict CLI (`predict`) | Done | `cliper/cli.py`, `cliper/caid_predict.py`, `cliper/caid_io.py` |
| Stage4 CNN + `concat_window` inference path | Done | reuses training window merge logic; CPU-only |
| Precomputed embedding input (`.npy` / `.h5` / `.resfeat.txt`) | Done | `--embeddings-dir` |
| CAID output writers (`.caid`, `timings.csv`) | Done | `cliper/caid_io.py` |
| Reference predict config | Done | `configs/caid_stage4_predict.yaml` |
| Docker + submission guide | Done | `submission/caid/` (see [`SUBMISSION.md`](submission/caid/SUBMISSION.md)) |
| Smoke tests | Done | `tests/test_caid_predict.py` |

**Submission model (locked):** `stage4_prostt5_cnn_concat_window_t1_regularized`  
Do **not** use Stage 5/6 for the standard CAID flow — they need live ProstT5 + PROSITE special-token encoding, while CAID mounts **precomputed ProstT5 residue embeddings** at runtime.

Expected checkpoint on the training server:

```text
artifacts/runs/stage4_prostt5_cnn_concat_window_t1_regularized/exp0001/checkpoints/best.pt
```

(`best.pt` is gitignored; copy manually into `submission/caid/models/` when packaging Docker.)

### What teammates should do on the server (next steps)

```bash
cd /data/huggs/hujh/CLIPer
git pull origin main
git log -1 --oneline          # expect 7156d3e
python -m cliper.cli predict --help
```

1. **Smoke test** — 2-line FASTA + existing ProstT5 embeddings → `.caid` + `timings.csv` (see [§5 predict CLI](#5-caid4-predict-offline-embeddings) below).
2. **CPU profile** — run the **longest** target sequence; confirm runtime **< 6 h**, threads **≤ 24**, RAM **≤ 48 GB**.
3. **Docker** — `docker build -f submission/caid/Dockerfile ...` and run on Linux CPU (details in [`submission/caid/SUBMISSION.md`](submission/caid/SUBMISSION.md)).
4. **Submit** — push image / Dockerfile to CAID before **2026-05-31**; confirm flavor directory name `linker` with organizers.

### Still open

- [ ] Server repo updated (`git pull`)
- [ ] Real `best.pt` smoke test on server
- [ ] Longest-sequence CPU timing recorded
- [ ] Docker build verified on clean Linux
- [ ] CAID online form + organizer confirmation of `linker` flavor name

Full checklist and compliance table: [`submission/caid/SUBMISSION.md`](submission/caid/SUBMISSION.md).

## Stage 1 Pipeline
The implementation includes:
- strict 3-line FASTA parsing (`>id`, sequence, label string)
- deterministic protein-level train/val split (`seed=42`, `80/20`)
- `dataset/error.txt` exclusion for disagreement cases
- CAID3 holdout leakage guard
- long-sequence training crop (`window_size=1024`) centered on linker-positive residues
- long-sequence eval windows using sequence-only heuristics + coverage windows
- weighted BCE training (`pos_weight = neg/pos`)
- checkpoint selection by **validation AUPRC**
- threshold tuning on validation for F1/MCC reporting
- TensorBoard 可视化与训练日志持久化
- 自动实验编号（`exp0001`, `exp0002`, ...），每次训练 checkpoints 分开保存
- 频率参数：`save_every` / `print_every` / `eval_every`（按 global step 生效）
- 训练和评测会自动尝试启动 TensorBoard（可通过配置关闭）

## Installation
```bash
pip install -r requirements.txt
```

## CLI Commands
### 1) Prepare data split and exclusion report
```bash
python -m cliper.cli prepare_data \
  --fasta dataset/disprot_202312_linker_label.fasta \
  --error-file dataset/error.txt \
  --caid-fasta dataset/linker.fasta \
  --seed 42 \
  --val-ratio 0.2 \
  --output-split artifacts/splits/disprot_split_seed42.json \
  --output-exclusion artifacts/splits/exclusion_report_seed42.json
```

### 2) Train Stage 1 model (BCE baseline)
```bash
python -m cliper.cli train --config configs/stage1_prostt5.yaml
```

Resume training from a saved checkpoint (continues in the same experiment directory):
```bash
python -m cliper.cli train \
  --config configs/stage1_prostt5.yaml \
  --resume-checkpoint artifacts/runs/stage1_prostt5/exp0001/checkpoints/last.pt
```

### 2b) Train Stage 2 model (BCE + SupCon from scratch)
```bash
python -m cliper.cli train --config configs/stage2_prostt5_supcon.yaml
```

### 2c) Train Stage 3 model (MLP5 + BCE, no contrastive)
```bash
python -m cliper.cli train --config configs/stage3_prostt5_mlp5.yaml
```

### 2d) Train Stage 3 model (MLP12 + BCE, no contrastive)
```bash
python -m cliper.cli train --config configs/stage3_prostt5_mlp12.yaml
```

### 2e) Train Stage 3 model (Transformer head + BCE, no contrastive)
```bash
python -m cliper.cli train --config configs/stage3_prostt5_transformer.yaml
```

### 2f) Train Stage 4 model (CNN head + concat_window)
```bash
python -m cliper.cli train --config configs/stage4_prostt5_cnn_concat_window.yaml
```

### 2g) Build Stage 5 PROSITE motif library and tokenizer
```bash
python scripts/build_motif_library_from_profile.py \
  --prosite-dat prosite.dat \
  --out-json configs/prosite_motif_library_full.json

python scripts/train_prosite_tokenizer.py \
  --base-tokenizer /data/huggs/hujh/CLIPer/models/Rostlab-ProstT5 \
  --motif-json configs/prosite_motif_library_full.json \
  --out-dir configs/prosite_tokenizer
```

### 2h) Train Stage 5 model (PROSITE special tokens)
```bash
python -m cliper.cli train --config configs/stage5_prostt5_motif.yaml
```

### 2i) Train Stage 6 model (dual tokenizer weighted-logit fusion)
```bash
python -m cliper.cli train --config configs/stage6_prostt5_dual_tokenizer.yaml
```

### 2j) Prepare Stage 5/6 motif data pack (seed + coverage reports)
```bash
python scripts/prepare_motif_data_pack.py \
  --motif-json configs/prosite_motif_library_full.json \
  --matching prosite \
  --train-fasta dataset/disprot_202312_linker_label.fasta \
  --caid-fasta dataset/linker.fasta \
  --split-manifest artifacts/splits/disprot_split_seed42.json \
  --out-dir artifacts/motif \
  --max-per-residue 1 \
  --top-k 10
```

Outputs:
- `artifacts/motif/motif_coverage_summary.json`
- `artifacts/motif/motif_coverage_train.json`
- `artifacts/motif/motif_coverage_val.json`
- `artifacts/motif/motif_coverage_caid.json`

Stage 2 configuration keys:
- `stage: stage2`
- `contrastive.enabled`
- `contrastive.weight`
- `contrastive.temperature`
- `contrastive.proj_dim`
- `contrastive.max_samples_per_class`

When Stage 2 is enabled, TensorBoard will additionally track:
- `train/loss_bce`
- `train/loss_supcon`
- `train/loss_total`
- `train/contrastive_num_samples`

Stage 3 configuration keys:
- `stage: stage3`
- `freeze_backbone: true`
- `classifier_head.type: mlp5 | mlp12 | transformer`
- `classifier_head.hidden_dims` (mlp5 needs 4 dims; mlp12 needs 11 dims)
- `classifier_head.num_layers` (transformer)
- `classifier_head.num_heads` (transformer)
- `classifier_head.ffn_dim` (transformer)
- `classifier_head.use_positional_encoding` (transformer)
- `classifier_head.dropout`
- `classifier_head.activation`
- `contrastive.enabled: false`

Stage 4 additional configuration keys:
- `stage: stage4`
- `local_context.enabled`
- `local_context.radius`
- `local_context.mode: concat_window`
- `local_context.include_self`
- `classifier_head.type: cnn` (optional)
- `classifier_head.conv_channels` (cnn)
- `classifier_head.kernel_size` (cnn, odd integer)
- `classifier_head.dilations` (cnn; length must match `conv_channels`)

Stage 5 additional configuration keys:
- `stage: stage5`
- `tokenizer_name` (saved tokenizer extended from the base ProstT5 tokenizer)
- `motif.enabled`
- `motif.source` (PROSITE motif JSON with `motifs: [{id, ac, de, pa, token}]`)
- `motif.matching: prosite`
- `motif.tokenization: special_token`

Stage 6 additional configuration keys:
- `stage: stage6`
- `dual_tokenizer.enabled: true`
- `dual_tokenizer.fusion: weighted_logits`
- `dual_tokenizer.branches.plain.weight: 0.5`
- `dual_tokenizer.branches.plain.motif.enabled: false`
- `dual_tokenizer.branches.special.weight: 0.5`
- `dual_tokenizer.branches.special.tokenizer_name` (saved PROSITE tokenizer, e.g. `configs/prosite_tokenizer`)
- `dual_tokenizer.branches.special.motif.enabled: true`
- `dual_tokenizer.branches.special.motif.source`
- `dual_tokenizer.branches.special.motif.matching: prosite`
- `dual_tokenizer.branches.special.motif.tokenization: special_token`

Stage 3/4 runtime behavior:
- If `stage: stage3` or `stage: stage4` and config sets `contrastive.enabled: true`, pipeline will force it to `false`.
- Stage 3/4 loss remains pure BCE (`BCEWithLogitsLoss`) for controlled architecture-only validation.

Stage 5 runtime behavior:
- If `stage: stage5` and config sets `contrastive.enabled: true`, pipeline will force it to `false`.
- Stage 5 keeps pure BCE and trains motif special-token embeddings plus the classifier head while preserving residue-level labels and metrics.

Stage 6 runtime behavior:
- If `stage: stage6` and config sets `contrastive.enabled: true`, pipeline will force it to `false`.
- Each batch is encoded twice: once with ordinary residue tokenization, once with motif span replacement and PROSITE special tokens.
- The two branches keep independent backbone/head parameters; checkpoints store both branches in one model state.
- Training logs include `train/loss_plain_bce`, `train/loss_special_bce`, and `train/loss_fused_total`.
- Validation, CAID3, and standalone `eval` write the same prediction format as earlier stages, using weighted fused logits.

W&B（Weights & Biases）配置（按每次实验 YAML 生效参数记录）：
- `use_wandb`
- `wandb_entity`
- `wandb_project`
- `wandb_mode` (`online` / `offline` / `disabled`)
- `wandb_run_name`
- `wandb_group`
- `wandb_tags`
- `wandb_dir`（可选，默认写到实验目录下 `wandb/`）

当 `use_wandb: true` 时：
- 训练会自动创建 W&B run，`wandb.config` 写入当次实验实际生效配置（resolved YAML）
- 训练/验证/CAID3 指标会同步上报到 W&B
- `eval` 命令会创建独立的评测 run 并上报评测指标

训练时若 `auto_start_tensorboard: true`（默认），会在服务器上启动 TensorBoard（`tensorboard_host` / `tensorboard_port`，见 `configs/stage1_prostt5.yaml`）。日志目录在当次实验子目录下：`artifacts/runs/<output_dir>/expXXXX/tensorboard/`。

在**服务器上**手动查看（汇总该 `output_dir` 下所有 `expXXXX`）：
```bash
tensorboard --logdir artifacts/runs/stage1_prostt5 --host 0.0.0.0 --port 6006
```

在**本机浏览器**看远程 TensorBoard：把服务器端口转发到本地。

**Windows（PowerShell）**，与仓库脚本一致：
```powershell
powershell -ExecutionPolicy Bypass -File scripts/forward_tensorboard.ps1 `
  -RemoteHost <server_ip_or_domain> `
  -RemoteUser <username> `
  -RemotePort 6006 `
  -LocalPort 16006
```

**Linux / macOS / Git Bash**：
```bash
chmod +x scripts/forward_tensorboard.sh
bash scripts/forward_tensorboard.sh <username>@<server_host> 6006 16006
```

可选指定 SSH 私钥：
```bash
bash scripts/forward_tensorboard.sh <username>@<server_host> 6006 16006 ~/.ssh/id_ed25519
```

然后在本地浏览器打开 `http://127.0.0.1:16006`。

**说明**：训练进程自动起的 TensorBoard 若启动失败，请确认已安装依赖 `pip install -r requirements.txt`（含 `tensorboard`），并查看当次实验目录下 `logs/tensorboard_runtime.log`。多用户共用机器时若 `6006` 冲突，请在 YAML 里改 `tensorboard_port`。

### 3) Evaluate a checkpoint
```bash
python -m cliper.cli eval \
  --checkpoint artifacts/runs/stage1_prostt5/checkpoints/best.pt \
  --fasta dataset/linker.fasta
```
不传 `--output-dir` 时，评测结果会自动保存到 checkpoint 所属实验目录下：
`.../expXXXX/evaluations/evalXXXX/`

### 5) CAID4 predict (offline embeddings)

For **CAID4 submission**: reads a standard **2-line FASTA** (id + sequence, no labels), loads **precomputed ProstT5 embeddings** per protein, runs the **Stage 4 CNN head on CPU**, and writes CAID outputs.

```bash
python -m cliper.cli predict \
  --checkpoint artifacts/runs/stage4_prostt5_cnn_concat_window_t1_regularized/exp0001/checkpoints/best.pt \
  --fasta /path/to/targets.fasta \
  --embeddings-dir /path/to/prostt5_embeddings \
  --output-dir artifacts/caid_smoke \
  --flavor linker \
  --device cpu \
  --num-threads 4
```

Outputs under `--output-dir`:

- `{flavor}/{protein_id}.caid` — per-residue probability + binary label
- `timings.csv` — per-sequence runtime (milliseconds)
- `predict_summary.json` — run metadata for debugging

Embedding file naming (first match under `--embeddings-dir`):

| FASTA header | Examples |
| --- | --- |
| `>P04637` | `P04637.npy`, `P04637.h5`, `P04637.resfeat.txt` |

Local benchmark FASTA `dataset/linker.fasta` is **3-line labeled**; convert to 2-line for predict tests:

```bash
awk 'NR%3==1 || NR%3==2' dataset/linker.fasta > /tmp/linker_seqonly.fasta
```

Docker entrypoint: `submission/caid/predict.sh`. Build/run instructions: [`submission/caid/SUBMISSION.md`](submission/caid/SUBMISSION.md).

### 6) Batch structure visualization from `predictions.tsv`
RCSB experimental structures are used first. If unavailable or coverage is too low, the workflow falls back to AlphaFold (when `--fallback alphafold`).

```bash
python scripts/structure_viz_batch.py \
  --predictions-tsv artifacts/runs/stage3_prostt5_mlp5/exp0001/evaluations/eval0003/predictions.tsv \
  --fasta dataset/linker.fasta \
  --out-dir artifacts/structure_viz \
  --threshold 0.5 \
  --fallback alphafold
```

Outputs:
- `artifacts/structure_viz/annotated_pdb/{protein_id}.pdb`
- `artifacts/structure_viz/html/{protein_id}.html`
- `artifacts/structure_viz/manifest.tsv`
- `artifacts/structure_viz/summary.json`
- Notebook viewer: `notebooks/linker_structure_viz.ipynb`

Visualization behavior:
- Linker residues are highlighted in taro purple (`#BFA6E8`).
- Each HTML contains 3 panels:
  - Predicted linker (purple)
  - True linker (purple)
  - Prediction-vs-True comparison (TP/FP/FN/TN colors)

## Key Outputs
- Split manifest JSON and exclusion report JSON from `prepare_data`
- 每次训练自动生成独立实验目录：`artifacts/runs/.../expXXXX/`
- Best checkpoint: `artifacts/runs/.../expXXXX/checkpoints/best.pt`
- Last checkpoint: `artifacts/runs/.../expXXXX/checkpoints/last.pt`
- 周期 checkpoint: `artifacts/runs/.../expXXXX/checkpoints/step_*.pt`
- 训练日志：`artifacts/runs/.../expXXXX/logs/train.log`
- TensorBoard events：`artifacts/runs/.../expXXXX/tensorboard/`
- W&B 本地运行缓存（启用时）：`artifacts/runs/.../expXXXX/wandb/`
- 评测输出默认：`artifacts/runs/.../expXXXX/evaluations/evalXXXX/`
- Training and validation metrics JSON
- CAID3 metrics JSON
- Prediction TSV with columns:
  `protein_id`, `position_1based`, `probability`, `pred_label`

## Notes
- Default config in [`configs/stage1_prostt5.yaml`](configs/stage1_prostt5.yaml).
- Stage 2 config template: [`configs/stage2_prostt5_supcon.yaml`](configs/stage2_prostt5_supcon.yaml).
- Stage 3 config template: [`configs/stage3_prostt5_mlp5.yaml`](configs/stage3_prostt5_mlp5.yaml).
- Stage 3 (MLP12) config template: [`configs/stage3_prostt5_mlp12.yaml`](configs/stage3_prostt5_mlp12.yaml).
- Stage 3 (Transformer) config template: [`configs/stage3_prostt5_transformer.yaml`](configs/stage3_prostt5_transformer.yaml).
- Stage 4 (CNN + concat_window) config template: [`configs/stage4_prostt5_cnn_concat_window.yaml`](configs/stage4_prostt5_cnn_concat_window.yaml).
- Stage 4 (CNN T1 regularized) config template: [`configs/stage4_prostt5_cnn_concat_window_t1_regularized.yaml`](configs/stage4_prostt5_cnn_concat_window_t1_regularized.yaml).
- CAID4 predict reference config: [`configs/caid_stage4_predict.yaml`](configs/caid_stage4_predict.yaml).
- CAID4 submission bundle (Docker + checklist): [`submission/caid/SUBMISSION.md`](submission/caid/SUBMISSION.md).
- Stage 4 (CNN T3 balanced) config template: [`configs/stage4_prostt5_cnn_concat_window_t3_balanced.yaml`](configs/stage4_prostt5_cnn_concat_window_t3_balanced.yaml).
- Stage 5 PROSITE special-token config template: [`configs/stage5_prostt5_motif.yaml`](configs/stage5_prostt5_motif.yaml).
- Stage 6 dual-tokenizer fusion config template: [`configs/stage6_prostt5_dual_tokenizer.yaml`](configs/stage6_prostt5_dual_tokenizer.yaml).
- Stage 5 PROSITE motif library: `configs/prosite_motif_library_full.json`.
- Stage 5 PROSITE tokenizer artifact: `configs/prosite_tokenizer/`.
- Stage 5 seed motif library: [`configs/motif_library_seed_stage5.json`](configs/motif_library_seed_stage5.json).
- Motif library minimal example: [`configs/motif_library_example.json`](configs/motif_library_example.json).
- Motif data-pack generator script: [`scripts/prepare_motif_data_pack.py`](scripts/prepare_motif_data_pack.py).
- PROSITE motif-library builder from `prosite.dat`: [`scripts/build_motif_library_from_profile.py`](scripts/build_motif_library_from_profile.py).
- PROSITE tokenizer builder: [`scripts/train_prosite_tokenizer.py`](scripts/train_prosite_tokenizer.py).
- For local tests without downloading a backbone, use `backbone_name: dummy` in config.
- `save_every` / `print_every` / `eval_every` 默认单位是训练 step（global step）。
- TensorBoard 自动启动相关参数：`auto_start_tensorboard` / `tensorboard_host` / `tensorboard_port`。
- W&B 相关参数：`use_wandb` / `wandb_entity` / `wandb_project` / `wandb_mode` / `wandb_run_name` / `wandb_group` / `wandb_tags` / `wandb_dir`。
- 远程端口转发：`scripts/forward_tensorboard.ps1`（Windows）、`scripts/forward_tensorboard.sh`（Linux/macOS/Git Bash）。

## Current Model Comparison (2026-04-28)
Representative CAID3 results from recent Stage 4 runs (single-run snapshots):

| Model config | CAID3 AUROC | CAID3 AUPRC | CAID3 F1 | CAID3 MCC |
| --- | ---: | ---: | ---: | ---: |
| `stage4_prostt5_mlp5_concat_window` | `~0.917` | `~0.501` | `~0.542` | `~0.516` |
| `stage4_prostt5_transformer_concat_window_stable` | `~0.892` | `~0.488` | `~0.521` | `~0.501` |
| `stage4_prostt5_cnn_concat_window_t1_regularized` | `~0.926-0.927` | `~0.552-0.569` | `~0.546-0.568` | `~0.513-0.544` |

Current best-performing family in these experiments: **Stage 4 CNN + concat_window**, especially
`stage4_prostt5_cnn_concat_window_t1_regularized`.

**CAID4 submission uses this same Stage 4 T1 regularized CNN checkpoint** (not Stage 5/6). See [Project Status — CAID4 Submission](#project-status--caid4-submission-2026-05).

Notes:
- These are single-run / few-seed observations, not final multi-seed averages.
- Final model selection should use repeated seeds and report mean/std over the same split.

## Stage 3 Experiment Plan
- `E0`: Stage 1 linear head baseline (current best config).
- `E1`: Stage 3 MLP5 + BCE (primary comparison).
- `E2`: Stage 3 MLP5 + BCE with `classifier_head.dropout` grid `{0.2, 0.3, 0.4}`.
- `E3`: Stage 3 MLP5 + BCE with `lr` grid `{3e-4, 1e-4}`.
- `E4`: Stage 3 MLP12 + BCE (deeper-head comparison against E1).
- `E5`: Stage 3 Transformer + BCE (attention head comparison against E1/E4).
- Selection rule: choose the checkpoint with highest validation AUPRC; if gain `< 0.005`, move to class-balanced focal loss (still without contrastive learning).

## Stage 5 Experiment Plan
- `M0`: Stage 4 best checkpoint/config baseline.
- `M1`: Stage 5 PROSITE special-token primary run.
- `M2`: Stage 5 PROSITE special-token with local-context radius sweep.
- `M3`: Stage 5 PROSITE special-token without local context (`local_context.enabled=false`) for interaction ablation.
- `M4`: Optional adapter/LoRA backbone ablation if new-token-only training underfits.
- Selection rule: prefer the model with best validation AUPRC, and report paired AUROC/F1/MCC on CAID3.

## Stage 6 Experiment Plan
- `D0`: Stage 4 T1 regularized CNN baseline.
- `D1`: Stage 5 PROSITE special-token baseline.
- `D2`: Stage 6 dual-tokenizer `0.5 / 0.5` weighted-logit fusion primary run.
- `D3`: Optional branch-weight ablation after D2, e.g. `0.25 / 0.75` and `0.75 / 0.25`, only if D2 beats both single-branch baselines.
- Selection rule: prefer validation AUPRC, then report CAID3 AUPRC/AUROC/F1/MCC using the fused logits from the best checkpoint.

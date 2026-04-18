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
- **Head options:** `classifier_head.type=mlp5` / `mlp12` / `transformer`
- **MLP5 layout:** `hidden -> 1024 -> 256 -> 128 -> 64 -> 1`
- **MLP12 layout:** `hidden -> 1024 -> 1024 -> 768 -> 768 -> 512 -> 512 -> 256 -> 256 -> 128 -> 128 -> 64 -> 1`
- **Transformer layout:** residue-wise encoder (`num_layers=2`, `num_heads=4`, `ffn_dim=2048`) + per-residue linear classifier
- **Hidden blocks:** each hidden layer uses `ReLU + LayerNorm + Dropout(0.3)`
- **Backbone policy:** keep ProstT5 frozen; train classification head only

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

Stage 3 runtime behavior:
- If `stage: stage3` and config sets `contrastive.enabled: true`, pipeline will force it to `false` and log:
  `"[stage3] contrastive.enabled=true ignored; forced to false for stage3."`
- Stage 3 loss remains pure BCE (`BCEWithLogitsLoss`) for controlled architecture-only validation.

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

### 4) Batch structure visualization from `predictions.tsv`
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
- For local tests without downloading a backbone, use `backbone_name: dummy` in config.
- `save_every` / `print_every` / `eval_every` 默认单位是训练 step（global step）。
- TensorBoard 自动启动相关参数：`auto_start_tensorboard` / `tensorboard_host` / `tensorboard_port`。
- W&B 相关参数：`use_wandb` / `wandb_entity` / `wandb_project` / `wandb_mode` / `wandb_run_name` / `wandb_group` / `wandb_tags` / `wandb_dir`。
- 远程端口转发：`scripts/forward_tensorboard.ps1`（Windows）、`scripts/forward_tensorboard.sh`（Linux/macOS/Git Bash）。

## Stage 3 Experiment Plan
- `E0`: Stage 1 linear head baseline (current best config).
- `E1`: Stage 3 MLP5 + BCE (primary comparison).
- `E2`: Stage 3 MLP5 + BCE with `classifier_head.dropout` grid `{0.2, 0.3, 0.4}`.
- `E3`: Stage 3 MLP5 + BCE with `lr` grid `{3e-4, 1e-4}`.
- `E4`: Stage 3 MLP12 + BCE (deeper-head comparison against E1).
- `E5`: Stage 3 Transformer + BCE (attention head comparison against E1/E4).
- Selection rule: choose the checkpoint with highest validation AUPRC; if gain `< 0.005`, move to class-balanced focal loss (still without contrastive learning).

# CLIPer
**CLIPer** stands for **Contrastive Learning for Intrinsically disordered Protein linker classification**.

Stage 1 provides a reproducible residue-level binary classifier pipeline for linker prediction:
- **Task:** residue-level linker classification (`0` non-linker, `1` linker)
- **Backbone:** pretrained **ProstT5** (frozen in v1)
- **Training source:** DisProt (`dataset/disprot_202312_linker_label.fasta`)
- **Holdout benchmark:** CAID3 (`dataset/linker.fasta`) used as strict final test only

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

### 2) Train Stage 1 model
```bash
python -m cliper.cli train --config configs/stage1_prostt5.yaml
```

训练后可用 TensorBoard 查看：
```bash
tensorboard --logdir artifacts/runs/stage1_prostt5
```

远程训练时可用 PowerShell 转发端口到本地：
```powershell
powershell -ExecutionPolicy Bypass -File scripts/forward_tensorboard.ps1 `
  -RemoteHost <server_ip_or_domain> `
  -RemoteUser <username> `
  -RemotePort 6006 `
  -LocalPort 16006
```
然后在本地浏览器访问 `http://127.0.0.1:16006`。

### 3) Evaluate a checkpoint
```bash
python -m cliper.cli eval \
  --checkpoint artifacts/runs/stage1_prostt5/checkpoints/best.pt \
  --fasta dataset/linker.fasta
```
不传 `--output-dir` 时，评测结果会自动保存到 checkpoint 所属实验目录下：
`.../expXXXX/evaluations/evalXXXX/`

## Key Outputs
- Split manifest JSON and exclusion report JSON from `prepare_data`
- 每次训练自动生成独立实验目录：`artifacts/runs/.../expXXXX/`
- Best checkpoint: `artifacts/runs/.../expXXXX/checkpoints/best.pt`
- Last checkpoint: `artifacts/runs/.../expXXXX/checkpoints/last.pt`
- 周期 checkpoint: `artifacts/runs/.../expXXXX/checkpoints/step_*.pt`
- 训练日志：`artifacts/runs/.../expXXXX/logs/train.log`
- TensorBoard events：`artifacts/runs/.../expXXXX/tensorboard/`
- 评测输出默认：`artifacts/runs/.../expXXXX/evaluations/evalXXXX/`
- Training and validation metrics JSON
- CAID3 metrics JSON
- Prediction TSV with columns:
  `protein_id`, `position_1based`, `probability`, `pred_label`

## Notes
- Default config in [`configs/stage1_prostt5.yaml`](configs/stage1_prostt5.yaml).
- For local tests without downloading a backbone, use `backbone_name: dummy` in config.
- `save_every` / `print_every` / `eval_every` 默认单位是训练 step（global step）。
- TensorBoard 自动启动相关参数：`auto_start_tensorboard` / `tensorboard_host` / `tensorboard_port`。
- 远程端口转发脚本：[`scripts/forward_tensorboard.ps1`](scripts/forward_tensorboard.ps1)。

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

### 3) Evaluate a checkpoint
```bash
python -m cliper.cli eval \
  --checkpoint artifacts/runs/stage1_prostt5/checkpoints/best.pt \
  --fasta dataset/linker.fasta \
  --output-dir artifacts/eval/caid3
```

## Key Outputs
- Split manifest JSON and exclusion report JSON from `prepare_data`
- Best checkpoint: `artifacts/runs/.../checkpoints/best.pt`
- Training and validation metrics JSON
- CAID3 metrics JSON
- Prediction TSV with columns:
  `protein_id`, `position_1based`, `probability`, `pred_label`

## Notes
- Default config in [`configs/stage1_prostt5.yaml`](configs/stage1_prostt5.yaml).
- For local tests without downloading a backbone, use `backbone_name: dummy` in config.

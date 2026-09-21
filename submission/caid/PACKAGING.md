# CLIPer CAID4 完整打包指南（ProstT5 + extract_embeddings + predict）

两个提交模型：

| 方法名（建议） | Checkpoint |
|----------------|------------|
| CLIPer-stage4-CNN-t1 | `artifacts/runs/stage4_prostt5_cnn_concat_window_t1/exp0001/checkpoints/best.pt` |
| CLIPer-stage7-CRNN-a1 | `artifacts/runs/stage7_prostt5_crnn_expl_a1_drop040/exp0005/checkpoints/best.pt` |

CAID 截止：**2026-06-30**

---

## 性能（你已测过）

最长序列 **DP04198（3010 aa）** CPU predict：

| 模型 | 耗时 |
|------|------|
| Stage4 | **167 ms** |
| Stage7 | **406 ms** |

远低于 CAID 单条 **6 小时** 限制。

---

## 包内包含什么

| 组件 | 路径 | 作用 |
|------|------|------|
| ProstT5 权重 | `models/Rostlab-ProstT5/` | 特征提取 backbone |
| `extract_embeddings` | `cliper/extract_embeddings.py` + CLI | FASTA → `{id}.npy` |
| `predict` | `cliper/caid_predict.py` + `predict.sh` | embedding → `linker/*.caid` |
| 分类头权重 | `submission/caid/models/best.pt` | 每个镜像一个 checkpoint |

容器两种模式（环境变量 `MODE`）：

- `MODE=extract`：FASTA → ProstT5 → `/output/embeddings/*.npy`
- `MODE=predict`：读 `/input/embeddings` → 输出 `.caid` + `timings.csv`

---

## 一、服务器上准备文件

```bash
cd /data/huggs/hujh/CLIPer
conda activate hupy

# 1. 分类头 checkpoint（两个模型各一份）
mkdir -p submission/caid/models

cp artifacts/runs/stage4_prostt5_cnn_concat_window_t1/exp0001/checkpoints/best.pt \
   submission/caid/models/stage4_t1_exp0001_best.pt

cp artifacts/runs/stage7_prostt5_crnn_expl_a1_drop040/exp0005/checkpoints/best.pt \
   submission/caid/models/stage7_a1_exp0005_best.pt

# 2. 确认 ProstT5 在构建上下文内
ls models/Rostlab-ProstT5/pytorch_model.bin
```

---

## 二、构建完整 Docker 镜像（含 ProstT5）

### Stage4 镜像

```bash
cp submission/caid/models/stage4_t1_exp0001_best.pt submission/caid/models/best.pt

docker build -f submission/caid/Dockerfile.full \
  -t cliper-caid-full:linker-stage4-cnn-t1 .
```

### Stage7 镜像

```bash
cp submission/caid/models/stage7_a1_exp0005_best.pt submission/caid/models/best.pt

docker build -f submission/caid/Dockerfile.full \
  -t cliper-caid-full:linker-stage7-crnn-a1 .
```

> 镜像体积较大（含 ProstT5 ~2GB），构建需几分钟。

---

## 三、测试完整流程

### 3a. 容器内提取 embedding

```bash
mkdir -p /tmp/caid_full_test/embeddings /tmp/caid_full_test/predict

docker run --rm \
  -e MODE=extract \
  -e NUM_THREADS=4 \
  -v /tmp/longest.fasta:/input/targets.fasta:ro \
  -v /tmp/caid_full_test/embeddings:/output/embeddings \
  cliper-caid-full:linker-stage4-cnn-t1

ls /tmp/caid_full_test/embeddings/
```

应出现 `DP04198.npy`（修复命名 bug 后）或 `DP04198.resfeat.npy`（旧代码）。

### 3b. 容器内 predict

```bash
docker run --rm \
  -e MODE=predict \
  -e NUM_THREADS=4 \
  -v /tmp/longest.fasta:/input/targets.fasta:ro \
  -v /tmp/caid_full_test/embeddings:/input/embeddings:ro \
  -v /tmp/caid_full_test/predict:/output \
  cliper-caid-full:linker-stage4-cnn-t1

cat /tmp/caid_full_test/predict/timings.csv
ls /tmp/caid_full_test/predict/linker/
```

### 3c. 一条龙（主办方若从 FASTA 开始）

```bash
# Step 1: extract
docker run --rm -e MODE=extract \
  -v /tmp/linker_seqonly.fasta:/input/targets.fasta:ro \
  -v /tmp/caid_emb:/output/embeddings \
  cliper-caid-full:linker-stage4-cnn-t1

# Step 2: predict
docker run --rm -e MODE=predict \
  -v /tmp/linker_seqonly.fasta:/input/targets.fasta:ro \
  -v /tmp/caid_emb:/input/embeddings:ro \
  -v /tmp/caid_out:/output \
  cliper-caid-full:linker-stage4-cnn-t1
```

---

## 四、推送镜像

```bash
docker tag cliper-caid-full:linker-stage4-cnn-t1 YOUR_USER/cliper-caid-full:linker-stage4-cnn-t1
docker tag cliper-caid-full:linker-stage7-crnn-a1 YOUR_USER/cliper-caid-full:linker-stage7-crnn-a1
docker login
docker push YOUR_USER/cliper-caid-full:linker-stage4-cnn-t1
docker push YOUR_USER/cliper-caid-full:linker-stage7-crnn-a1
```

---

## 五、打 tar 包（除 Docker 外另交源码+权重）

```bash
cd /data/huggs/hujh/CLIPer

tar -czvf CLIPer_caid4_stage4_bundle.tgz \
  cliper/caid_predict.py \
  cliper/caid_io.py \
  cliper/extract_embeddings.py \
  cliper/cli.py \
  cliper/modeling.py \
  cliper/windowing.py \
  disorder/sequence_embedding.py \
  disorder/feature_io.py \
  disorder/feature_modeling.py \
  disorder/fasta_parsing.py \
  submission/caid/predict.sh \
  submission/caid/extract_embeddings.sh \
  submission/caid/entrypoint.sh \
  submission/caid/Dockerfile.full \
  submission/caid/requirements-caid-full.txt \
  submission/caid/PACKAGING.md \
  models/Rostlab-ProstT5 \
  submission/caid/models/stage4_t1_exp0001_best.pt

# Stage7 同理，换 checkpoint 文件名
```

---

## 六、交给主办方的说明（可贴进表单）

**Method:** CLIPer linker predictor (ProstT5 + CNN or CRNN head)

**Workflow:**

1. `MODE=extract`: 2-line FASTA → ProstT5 residue embeddings `{protein_id}.npy`, shape `[L, 1024]`. Long sequences (L>1024): non-overlapping 1024-residue windows.
2. `MODE=predict`: embeddings + FASTA → `linker/{id}.caid` + `timings.csv`. Eval sliding windows (1024, stride 512) applied inside predict.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `predict` | `extract` or `predict` |
| `FASTA` | `/input/targets.fasta` | Input sequences |
| `PROSTT5_MODEL` | `/app/models/Rostlab-ProstT5` | Bundled ProstT5 |
| `EMBEDDINGS_DIR` | extract: `/output/embeddings`; predict: `/input/embeddings` | Embedding I/O |
| `CHECKPOINT` | `/models/best.pt` | Classifier head |
| `OUTPUT_DIR` | `/output` | CAID outputs |
| `FLAVOR` | `linker` | Output subdirectory |
| `NUM_THREADS` | `4` | CPU threads (max 24) |

---

## 七、提交 checklist

- [x] 最长序列 CPU 耗时 < 6 h
- [ ] 全量 31 条 smoke（extract + predict）
- [ ] Stage4 / Stage7 各 build 一个 `Dockerfile.full` 镜像
- [ ] Docker 内 extract + predict 测试通过
- [ ] 镜像 push 或 tar 包上传
- [ ] 在线表单提交 2 次（两个模型）

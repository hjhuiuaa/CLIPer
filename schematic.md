# CLIPer
* 本项目完整信息参考 `README.md`

## Stage 1: ProstT5 链接子残基二分类基线
* 分类标签：`0` 非 linker，`1` linker
* 特征主干：预训练 **ProstT5**（Stage 1 冻结主干，仅训练分类头）
* 数据集说明见 `dataset/README.md`
  - 训练/验证：`dataset/disprot_202312_linker_label.fasta`（当前已清洗为仅保留含 linker 的蛋白）
  - 最终测试（严格 holdout）：`dataset/linker.fasta`（CAID3）
* 关键流程：
  - 固定随机种子进行蛋白级 8:2 划分
  - 排除 `dataset/error.txt` 中的不一致样本
  - 长序列训练使用 linker 中心裁剪，评估使用无标签启发式窗口 + 覆盖窗口
  - 以验证集 AUPRC 选择最优 checkpoint
  - TensorBoard 记录训练/验证曲线，训练日志写入 `logs/train.log`
  - 每次实验自动编号并独立保存 checkpoint（`expXXXX/checkpoints`）
  - `save_every` / `print_every` / `eval_every` 控制保存、日志打印与验证频率
  - 训练与评测阶段自动尝试启动 TensorBoard
  - 评测输出默认写入 checkpoint 对应实验目录：`expXXXX/evaluations/evalXXXX`

## Stage 2: 引入正负样本对比学习
* 训练目标：`BCEWithLogits + SupCon`（联合损失）
  - `total_loss = bce_loss + contrastive.weight * supcon_loss`
* 训练策略：从零开始联合训练（不依赖 Stage 1 checkpoint）
* 主干策略：默认继续冻结 **ProstT5**，训练分类头与投影头（projection head）
* 当前 SupCon 实现（与 `cliper/pipeline.py` 一致）：
  - 输入特征：模型前向时返回 `residue_embeddings`（由 projection head 输出）
  - 有效位筛选：仅保留 `labels >= 0` 的残基；`-`（掩码位）直接丢弃
  - 类内采样：对标签 `0/1` 分别采样，每类最多 `contrastive.max_samples_per_class`
  - 退化保护：若采样后总样本 `< 2` 或类别数 `< 2`，返回 `supcon_loss=0`
  - 相似度计算：对 embedding 做 L2 normalize，使用
    `logits = (z @ z^T) / temperature`
  - 自对比排除：用对角 mask 去掉样本与自身配对
  - 正样本定义：同标签样本互为正对（监督对比）
  - 数值稳定：每行减去该行最大值后再做 `exp/log`
  - 损失形式：对每个 anchor 取正样本对数概率均值，再取负号与 batch 均值
  - 非有限值保护：若损失出现 NaN/Inf，直接抛错中止
* 配置入口：
  - Stage 1 模板：`configs/stage1_prostt5.yaml`
  - Stage 2 模板：`configs/stage2_prostt5_supcon.yaml`
  - 关键字段：`stage`、`contrastive.*`、`use_wandb`、`wandb_*`
* 可视化与日志：
  - TensorBoard 指标：`train/loss_bce`、`train/loss_supcon`、`train/loss_total`、`train/contrastive_num_samples`
  - W&B：当前代码中已注释停用（返回 `disabled`，不实际上报）
* 产物与元数据：
  - 训练目录：`artifacts/runs/.../expXXXX/`
  - 评测目录：`artifacts/runs/.../expXXXX/evaluations/evalXXXX/`
  - `run_metadata.json` / `evaluation_metadata.json` 中包含 `wandb_service` 与运行信息

## Stage 3: MLP / Transformer 分类头容量增强（不使用对比学习）
* 目标：在 Stage1/Stage2 基本持平前提下，优先通过分类头容量提升 AUPRC。
* 训练目标：`BCEWithLogitsLoss`（不引入 SupCon 项）
* 主干策略：继续冻结 **ProstT5**，仅训练分类头。
* 分类头结构（`classifier_head.type=mlp5 | mlp12 | transformer`）：
  - 5 层全连接：`hidden -> 1024 -> 256 -> 128 -> 64 -> 1`
  - 12 层全连接：`hidden -> 1024 -> 1024 -> 768 -> 768 -> 512 -> 512 -> 256 -> 256 -> 128 -> 128 -> 64 -> 1`
  - Transformer 头：逐残基 TransformerEncoder（默认 `2` 层、`4` 头、`ffn_dim=2048`）后接线性输出
  - 每个隐层后使用：`ReLU + LayerNorm + Dropout(0.3)`
  - 输出层仅线性映射到单 logit
* Stage3 约束：
  - `stage: stage3` 时，pipeline 强制 `contrastive.enabled=false`
  - 若配置误开 `contrastive.enabled=true`，会在日志写明被强制关闭
* 配置入口：
  - Stage 3 模板：`configs/stage3_prostt5_mlp5.yaml`
  - Stage 3 (MLP12) 模板：`configs/stage3_prostt5_mlp12.yaml`
  - Stage 3 (Transformer) 模板：`configs/stage3_prostt5_transformer.yaml`
  - 关键字段：`stage`、`freeze_backbone`、`classifier_head.*`、`contrastive.enabled`
* Stage3 小规模实验计划：
  - `E0`：Stage1 线性头基线
  - `E1`：Stage3 MLP5 + BCE（核心对照）
  - `E2`：Stage3 MLP5 + BCE，`dropout={0.2,0.3,0.4}`
  - `E3`：Stage3 MLP5 + BCE，`lr={3e-4,1e-4}`
  - `E4`：Stage3 MLP12 + BCE（深层头对照）
  - `E5`：Stage3 Transformer + BCE（注意力分类头对照）
  - 选择标准：验证集 AUPRC 最高；若提升 `< 0.005`，下一步改用 class-balanced focal（仍不加对比学习）

## 数据清洗策略（当前口径）
* 原始文件备份：`dataset/disprot_202312_linker_label_not_cleaned.fasta`
* 清洗后训练文件：`dataset/disprot_202312_linker_label.fasta`
* 清洗规则：移除标签全 `0` 的蛋白，仅保留标签中至少含一个 `1` 的样本
* 清洗统计报告：`artifacts/splits/disprot_linker_clean_report.json`
* split 需基于清洗后 FASTA 重新生成：
  - `artifacts/splits/disprot_split_seed42.json`
  - `artifacts/splits/exclusion_report_seed42.json`

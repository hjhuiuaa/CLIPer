# CLIPer
* 本项目完整信息参考 `README.md`

## Stage 1: ProstT5 链接子残基二分类基线
* 分类标签：`0` 非 linker，`1` linker
* 特征主干：预训练 **ProstT5**（Stage 1 冻结主干，仅训练分类头）
* 数据集说明见 `dataset/README.md`
  - 训练/验证：`dataset/disprot_202312_linker_label.fasta`
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
* 对比学习样本构造：
  - 以残基标签 `0/1` 在 batch 内构造监督对比学习正负对
  - 无效标签位（`-`）不参与对比损失
  - 若 batch 内单类或有效样本不足，则该 step 的 `supcon_loss=0`，仅优化 BCE
* 配置入口：
  - Stage 1 模板：`configs/stage1_prostt5.yaml`
  - Stage 2 模板：`configs/stage2_prostt5_supcon.yaml`
  - 关键字段：`stage`、`contrastive.*`、`use_wandb`、`wandb_*`
* 可视化与日志：
  - TensorBoard 指标：`train/loss_bce`、`train/loss_supcon`、`train/loss_total`、`train/contrastive_num_samples`
  - W&B（可选）：训练/验证/CAID3 与评测指标自动上报
  - 每次实验实际生效参数（resolved YAML）写入 `wandb.config`
* 产物与元数据：
  - 训练目录：`artifacts/runs/.../expXXXX/`
  - 评测目录：`artifacts/runs/.../expXXXX/evaluations/evalXXXX/`
  - `run_metadata.json` / `evaluation_metadata.json` 中包含 `wandb_service` 与运行信息

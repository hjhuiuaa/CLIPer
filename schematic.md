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

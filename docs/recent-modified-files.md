# 最近修改的文件清单

统计时间：2026-09-21。范围：约最近 10 天（9-18 起 T2/T3 协议评测与 Stage7 三分类）。  
仓库 `main` 相对 `origin/main`：**ahead 1 / behind 10**。最近提交：`49d1fac`（2026-09-20，`feat: T1 T2 T3 settings established`）。

未列入：`.idea/`、ProstT5 特征、`tmp/` 下 embedding / `.caid`、checkpoint 大文件、更早的未跟踪 Stage 4–7 探索 yaml。

## 代码与配置

| 时间 | 文件 | 说明 |
| --- | --- | --- |
| 9-21 13:25 | `scripts/build_t2_ternary_fasta.py` | 三分类 FASTA 构建 |
| 9-21 13:25 | `scripts/eval_protocol_t2_t3.py` | T2 / T3 协议评测 |
| 9-21 13:25 | `scripts/build_t3_negatives_fasta.py` | T3 负蛋白 |
| 9-21 13:25 | `configs/stage7_expl/stage7_prostt5_crnn_expl_a1_t2_ternary.yaml` | 三分类训练配置 |
| 9-20 19:27 | `cliper/pipeline.py` | CE、SupCon、val T2 选模 |
| 9-20 19:27 | `cliper/modeling.py` | CRNN `num_classes=3` |
| 9-20 19:27 | `cliper/data.py` | 标签字符含 `2` |
| 9-20 19:27 | `cliper/windowing.py` | 多类 logit 合并 + P(Linker) |
| 9-20 19:27 | `docs/experiment-analysis.md` | §8 T1 / T2 四项指标 |
| 9-18 17:14 | `docs/disordered-linker-survey.md` | 领域调研 / SOTA 表 |

## 数据

| 时间 | 文件 |
| --- | --- |
| 9-18 22:31 | `dataset/disprot_202312_t2_ternary.fasta` |
| 9-18 22:31 | `dataset/disprot_202312_t2_ternary_stats.json` |
| 9-18 14:33 | `artifacts/splits/disprot_t3_negatives_seed42.json` |
| 9-18 14:33 | `artifacts/splits/disprot_t3_negatives_seed42.fasta` |

## 评测产物

| 时间 | 目录 | 说明 |
| --- | --- | --- |
| 9-18 | `artifacts/protocol_eval/linkcnn_t1_exp0001/` | LinkCNN T2 / T3 |
| 9-18 | `artifacts/protocol_eval/linkcrnn_a1_exp0005/` | 二分类 LinkCRNN T2 / T3 |
| 9-18 夜 | `artifacts/protocol_eval/linkcrnn_a1_t2_ternary/` | 三分类 T1 val/CAID3 + T2 val/CAID3（含 `summary.json`） |

训练 run：`artifacts/runs/stage7_prostt5_crnn_expl_a1_t2_ternary/exp0002/`（含 `best.pt` 等，体积大，不逐条列出）。

# 任务型神经元、DR-ISR 与 KAN：结果边界

核对日期：2026-11-04。

## 已有论文结果，尚非本项目复现

来源：[Fan 等，No One-Size-Fits-All Neurons，v2，第 V-A 节、表 IV](https://arxiv.org/html/2405.02369v2#S5.T4)。作者报告训练/测试 8:2 划分，重复 5 次，报告均值；回归指标为 MSE，分类指标为准确率。

下表摘取标准 KAN 与 TN 两列，不将其他 KAN 变体混入同一列。回归越低越好，分类越高越好；不得跨不同数据集直接比较 MSE 数值。

| 数据集 | 指标 | KAN | 原论文 TN | 本项目复现状态 |
|---|---|---:|---:|---|
| California Housing | MSE | 0.2093 | 0.0553 | 未复现 |
| House Sales | MSE | 0.1207 | 0.0139 | 未复现 |
| Airfoil Self Noise | MSE | 0.1731 | 0.0170 | 未复现 |
| Diamonds | MSE | 0.0181 | 0.0069 | 未复现 |
| Abalone | MSE | 0.6677 | 0.0246 | 未复现 |
| Bike Sharing | MSE | 0.0640 | 0.0076 | 未复现 |
| Space GA | MSE | 0.3318 | 0.0055 | 未复现 |
| Airlines Delay | MSE | 0.9288 | 0.1690 | 未复现 |
| Credit | Accuracy | 0.7367 | 0.7498 | 未复现 |
| Heloc | Accuracy | 0.7230 | 0.7262 | 未复现 |
| Electricity | Accuracy | 0.8145 | 0.8119 | 未复现 |
| Phoneme | Accuracy | 0.8501 | 0.8651 | 未复现 |
| Magic Telescope | Accuracy | 0.8734 | 0.8531 | 未复现 |
| Vehicle | Accuracy | 0.7686 | 0.8000 | 未复现 |
| Orange Vs Grapefruit | Accuracy | 0.9555 | 0.9813 | 未复现 |
| Eye Movements | Accuracy | 0.5826 | 0.6080 | 未复现 |

按上述表中均值，TN 对标准 KAN 为 8/8 回归任务较好、6/8 分类任务较好。这是对作者表格的统计，不是显著性判断或普适优越性证明；对所有 KAN 变体的最优结果比较，分类并非全部领先，例如 Orange Vs Grapefruit 上 ChebyKAN 为 0.9835。

## v0.1 时的结果边界（历史记录）

v0.1 仅测表达式树 GP 搜索机制，没有将搜索结构参数化为神经元并训练网络。因此当时不能给出“DR-ISR 改进 TN 优于/劣于 KAN”的结论。GP 对 GP 的 150 次实验也不能代替 TN 对 KAN 的端到端实验。

已发现可复用资源：本机 G:/Task_based_neurons 中有作者实现、合成数据、一个单次 TN/MLP 记录和十种子 TN/MLP CSV；这些历史文件没有 KAN 列，不作为 TN/KAN 对照证据。原文件保持不变。

后续端到端实验由 [固定协议](neural_protocol.md) 和独立验收报告记录。原论文的数字始终与本项目的实测数字分开：小规模复现、结构限制或官方实现改动均必须单列说明。

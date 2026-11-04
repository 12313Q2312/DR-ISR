# DR-ISR

**Data- and Residual-Guided Incremental Symbolic Regression — 科研想法验证原型。**

当前交付：普通表达式树 GP、残差引导子树交叉、置乱残差对照；相同算子、数据、初始种群和显式求值预算。**这是最小实验，不是已经证明有效的新算法，也不是完整的增量符号回归/神经网络系统。**

## 本次验收与实际结果

已完成 **150 次**主实验（5 任务 × 10 种子 × 3 组），每次 3000 个计费求值单位。残差引导对普通 GP 的成对测试误差为 **11 胜 / 14 平 / 25 负**，本版**未显示整体优势**。不根据结果删掉失败种子。

- [中文验收报告与演示步骤](docs/acceptance_zh.md)
- [主实验完整结果](evidence/benchmark-v1/report.md)
- [下载后可直接打开的离线演示](evidence/benchmark-v1/demo.html)（GitHub 文件页面显示源码，下载后用浏览器打开）
- [普通 GP 基线试验](evidence/baseline-pilot-v1/report.md)

复核保存结果：`python verify_evidence.py evidence/benchmark-v1`。源码和实验均由 AI 协助生成，需结合报告中的局限解释。[GitHub Linux 自动测试与快速实验已通过](https://github.com/12313Q2312/DR-ISR/actions/runs/41836393020)。

## 一键运行

需要 Python 3.11+。已在 Windows / Python 3.14 验证；不需要 GPU、API Key、付费服务或下载数据集。测试用标准库 unittest。

```powershell
# 在仓库目录打开终端。新环境可放 G 盘（已有依赖可跳过安装）。
py -m venv G:\tools\venvs\dr-isr
G:\tools\venvs\dr-isr\Scripts\python.exe -m pip install -r requirements.txt
G:\tools\venvs\dr-isr\Scripts\python.exe run.py
```

如果当前 Python 已有所需依赖，只需：

```powershell
python run.py
```

程序先执行测试，再跑 5 个任务 × 10 个种子 × 3 种方法，每次预算 3000。默认保存到带时间戳的 `results/` 子目录，拒绝覆盖旧结果。可先快速检查：

```powershell
python run.py --config configs/smoke.json
python run.py --test-only
```

结果包含 `runs.jsonl`（每次运行的完整公式、误差、计数与时间）、`traces.csv`（搜索轨迹）、`summary.json`、`report.md`、学习曲线图、`demo.html`、全部数据 CSV、环境/配置/源码哈希清单和文件校验和。搜索时间不含数据生成、测试集预测和画图。固定种子可复现算法数值轨迹；不同机器的时间与浮点末位可以不同。

## 核心机制

普通 GP 随机挑选合法供体子树；残差引导方法对少量供体计算：

`r = y - f(X)`，`delta = J * (t-s)`，`score = mean(2*r*delta - delta²)`。

`s` 是待替换子树，`t` 是供体，`J` 是外层表达式对该子树的局部导数。例如两个符号模块串联 `sin(g(x))` 时，`J=cos(g(x))`。评分只是局部近似，实际训练误差仍通过完整子代求值确认。变异机制相同；没有额外系数拟合。

**预算包含引导成本**：完整适应度、供体探针、上下文导数各消耗一个训练集向量求值单位。另报节点×样本工作量和时间，因为相同单位不代表相同浮点运算数。详情见 [实验协议](docs/protocol.md)。

## 目录与继续研究

- `dr_isr/tree.py`：共享树结构、保护运算、序列化、上下文导数。
- `dr_isr/gp.py`：共享 GP 搜索引擎和预算账本。
- `dr_isr/data.py`：训练、插值测试和外推任务。
- `dr_isr/experiment.py`：成对种子实验、报告和曲线。
- `tests/`、`.github/workflows/ci.yml`：本地检查与 GitHub CI。
- [设计说明与材料勘误](docs/design_notes_zh.md)：两个模块串联、残差诊断、动态记忆及后续消融。

先看结果和失败模式，再决定是否加入更多机制。普通 GP 是初步研究基线；本版不能代表与 PySR、SRBench 中现代方法的比较，也不证明论文创新。涉及相关工作见设计说明中的原始资料。

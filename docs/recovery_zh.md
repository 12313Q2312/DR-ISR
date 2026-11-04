# 恢复与复验

完整代码、配置和基准结果均已纳入 Git。先从 GitHub 克隆，安装 requirements.txt 后运行 `python run.py --test-only` 和 `python verify_evidence.py evidence/benchmark-v1`。

交付压缩包同时包含 `DR-ISR-history.bundle`，它保存开发阶段的完整本地 Git 记录。可在希望保存项目的目录运行：

```powershell
git clone DR-ISR-history.bundle DR-ISR-restored
cd DR-ISR-restored
python run.py --test-only
python verify_evidence.py evidence/benchmark-v1
```

基准主实验 manifest 中的提交是运行时源码 `e17274c`；后续提交增加报告、演示和跨系统文本格式处理，没有修改 GP 搜索机制或已测数值。原始运行目录仍保留在本次工作区 results/ 中，Git 记录包含归档前后版本。文件文本换行处理不改变数据数值，归档校验和已重新计算。

GitHub 发布与本地提交可能采用不同提交编号；以文件内容、保存的核心源码哈希、数据指纹和 bundle 为复核依据。新实验默认创建新的时间戳目录，不覆盖已有结果。

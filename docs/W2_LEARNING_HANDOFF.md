# W2 多种子成对实验：学习交接包

## A. 一句话目标

在完全相同的数据边界和训练规则下，用五个预注册seed比较SimpleCNN与TemporalCNN，判断观察到的差距是否只是一次幸运初始化。

## B. 四个核心知识点

### 1. Multi-seed experiment

- 是什么：同一配置在多个预注册`run_seed`下重复训练。
- 为什么需要：单次结果会受初始化、Dropout和batch顺序影响。
- 不这样做：可能把一次幸运或倒霉运行当成模型能力。
- 与W1关系：W1固定数据划分，W2才可以只改变训练随机性。
- 面试追问：为什么相同seed不代表不同架构逐参数初始化相同？

### 2. Mean与sample std

- 是什么：mean描述五次结果中心，sample std用`n-1`分母描述观察到的波动。
- 为什么需要：同时回答“平均表现怎样”和“结果稳不稳定”。
- 不这样做：只报最高分会隐藏模型对随机性的敏感程度。
- 与W1关系：只有五次运行使用同一固定validation，统计才可比较。
- 面试追问：为什么五个seed不能支持统计显著性结论？

### 3. Paired delta

- 是什么：对同一个seed先计算`A1-A0`，再汇总五个差值。
- 为什么需要：把同一随机性口径下的两模型结果配成一对。
- 不这样做：只比较两个独立均值，会丢失同seed对应关系。
- 与W1关系：同seed配对依赖固定manifest和相同DataLoader顺序。
- 面试追问：有一个seed的一侧失败时，为什么不能补零或换seed？

### 4. 指标口径

- 是什么：Accuracy、Macro F1、per-SNR和best epoch报告mean±sample std；参数量等确定性量直接报告原值。
- 为什么需要：随机训练结果与模型固有属性不是同一类量。
- 不这样做：给参数量加std会制造没有意义的“波动”。
- 与上一阶段关系：W1固定数据，W2固定统计对象。
- 面试追问：为什么Macro F1可能高于或低于Accuracy？

## C. 源码学习路径

1. `scripts/run_v2_paired_experiments.py`
   - 先读`W2_MATRIX`和`create_w2_model`。
   - 再读`run_one_experiment`，最后读`main`的seed×模型循环。
   - 重点观察：每次怎样重置run seed、加载相同manifest、保存best checkpoint和原始JSON。
2. `src/signal_modulation/v2_statistics.py`
   - 先读`descriptive_summary`。
   - 再读`_paired_summary`和`summarize_w2_results`。
   - 重点观察：sample std、`A1-A0`方向、缺失配对为什么报错。
3. `src/signal_modulation/evaluation.py`
   - 读`calculate_classification_metrics`和`calculate_snr_metrics`。
   - 重点观察：per-SNR Macro F1如何从每个SNR自己的混淆矩阵计算。
4. `tests/test_w2_artifacts.py`
   - 重点观察：怎样重新计算summary、核对checkpoint哈希并逐个加载checkpoint。

## D. 必须能口述的链路

```text
固定split manifest
→ 取一个预注册run_seed
→ A0设置全局随机状态和同seed训练batch顺序
→ 训练、validation、最低validation loss保存best checkpoint
→ 保存A0原始JSON
→ 同一个run_seed重新设置随机状态
→ A1使用相同manifest和batch顺序
→ 保存A1 checkpoint与原始JSON
→ 对五个seed重复
→ 每模型计算mean与sample std
→ 同seed计算A1-A0 paired delta
→ 输出n_pairs=5与per-SNR汇总
```

## E. 简化手写目标

```python
for seed in frozen_run_seeds:
    for config in ("A0", "A1"):
        set_seed(seed)
        train_loader = make_loader(frozen_train, seed=seed, shuffle=True)
        result = train_and_validate(config, train_loader, validation_loader)
        save_raw_result(config, seed, result)
```

```python
deltas = [a1[seed] - a0[seed] for seed in frozen_run_seeds]
mean_delta = statistics.fmean(deltas)
sample_std = statistics.stdev(deltas)  # n-1
```

## F. 三道面试题与标准答案

### 问题1：代码怎样保证同seed的A0和A1看到相同batch顺序？

答案：每次模型运行都重新创建训练DataLoader，并给它独立的`torch.Generator`，generator使用相同`run_seed`。DataLoader的随机数与模型全局随机数分开，所以两个模型在对应epoch得到相同样本排列。相同seed不能让不同架构逐参数初始化相同，因为两种架构参数形状和随机数消耗不同。

### 问题2：为什么要计算paired delta，不能只比较两个模型的均值？

答案：paired delta先在同seed内计算`A1-A0`，保留了两边共同的随机性口径。本次五个Macro F1差值全部为正，均值为`+15.23`个百分点，sample std为`3.35`个百分点；它支持“本协议五个seed下方向一致”的描述，但不等于统计显著或普遍规律。

### 问题3：哪些量应该报告mean±std，哪些不应该？

答案：由随机训练产生的Accuracy、Macro F1、per-SNR指标和best epoch报告mean±sample std。数据集大小、split数量、参数量、配置和哈希是确定性量，直接报告原值。本次A0参数量为`11,499`，A1为`224,587`，不能给它们加跨seed std。

## G. 面试资产

- 技术判断：在当前固定validation和五个seed下，A1相对A0的Accuracy和Macro F1 paired delta方向全部为正。
- 稳定性数据：A1 Accuracy为`54.81% ± 0.64`个百分点，A0为`42.26% ± 2.08`个百分点。
- Trade-off：A1有`224,587`个参数，明显大于A0的`11,499`个；W2是整体架构比较，不能把差距归因于某一层。
- 低SNR限制：`-20`至`-14 dB`两者都接近机会水平，说明强噪声仍是共同瓶颈。
- 工程证据：10份原始JSON、10个可加载checkpoint、五组完整配对和零失败记录。
- 未获得的结论：没有V2 test结果、没有显著性检验、没有真实空口泛化证据。

## H. 三个遗忘风险

1. `mean±std`不能替代每个seed的原始结果。
2. paired delta必须来自同seed的完整配对，缺失时不能补零或换seed。
3. A1优于A0是整体架构比较，不能提前解释为“时序聚合一定有效”；这一归因要等W3容量受控消融。

# A2-L 预注册假设与实验锁

> 预注册阶段：W-C0  
> 状态：模型代码与任何A2-L结果产生前冻结  
> 数据范围：V2固定train/validation；禁止访问V1 test

## 1. 待检验假设

A2-L在高SNR段（`SNR≥10 dB`）的validation Macro F1优于A2-T和A2-G；在低SNR段（`SNR≤-10 dB`）无明显差异或略差。

机制假设：高SNR时符号级时序结构较清晰，递归聚合可能利用32步序列中的顺序；低SNR时噪声主导、可建模时序信息不足，LSTM增加的优化难度可能抵消表达能力。该机制是待检验解释，不是预先写好的结论。

## 2. 结果前冻结的判定标准

主判据使用五个同seed paired delta，方向固定为`A2-L - comparator`。主指标是高SNR段Macro F1；Accuracy、总体Macro F1和class×SNR只作辅助诊断。

“支持假设”必须同时满足：

1. 对A2-T和A2-G，高SNRMacro F1平均paired delta都至少为`+1.0`个百分点；
2. 两个比较各自至少`4/5`个seed方向为正；
3. 对两个比较，低SNRMacro F1平均paired delta都落在`[-1.0,+0.5]`个百分点，符合“无明显差异或略差”的预期。

“部分支持”仅用于以下情况：高SNR条件只对一个比较满足，或高SNR平均差值位于`[+0.5,+1.0)`个百分点且至少`4/5`方向为正。其他结果均判为“不支持”。如果总体均值提高但高SNR预注册条件不满足，也不能改写成支持原假设。

阈值依据：现有A2-T−A2-G高SNRMacro F1 paired delta为`-0.34 ± 2.48`个百分点，方向不一致；项目既有模型选择规则把小于`0.5`个百分点视为实践上接近。因此本实验使用更严格的`+1.0`个百分点与`4/5`同向作为描述性支持门槛。五个seed不做显著性检验。

## 3. 冻结结构

```text
shared TemporalCNN backbone
  Conv1d/BN/ReLU/MaxPool: 2 -> 64, length 128 -> 64
  Conv1d/BN/ReLU/MaxPool: 64 -> 128, length 64 -> 32
  Conv1d/BN/ReLU:         128 -> 128, length stays 32
  output (B, 128, 32)
-> transpose to (B, 32, 128)
-> LSTM(input_size=128, hidden_size=127, num_layers=1,
        batch_first=True, bidirectional=False, bias=True)
-> final time step (B, 127)
-> Dropout(p=0.3)
-> Linear(127, 11)
```

Dropout放在Linear之前，用于保持A2-T/A2-G的classifier正则化口径；它不增加参数。backbone只冻结**结构与初始state**，训练时仍参与梯度更新。若把backbone的`requires_grad`关闭，会改变训练问题并破坏与A2-T/A2-G的公平比较，禁止这样实现。

## 4. 容量与计算预期

- backbone参数：`91,968`；
- LSTM参数：`130,556`；
- Linear参数：`1,408`；
- 预期总可训练参数：`223,932`；
- A1/A2-T参数：`224,587`；
- 预期差值：`-655`，即`-0.292%`，满足目标`≤1%`。

实现后必须实测参数量、head参数量和MACs/FLOPs。LSTM MACs的计算口径必须单独说明，不能沿用只统计Conv1d/Linear的旧函数后声称是完整计算量。

## 5. 冻结训练与随机性

- run seeds：`20260901`至`20260905`，不得增删或挑最好seed；
- split：只加载`manifests/v2/`固定train/validation；
- 每个seed加载与A2-T/A2-G相同的已提交初始backbone state；
- backbone初始state相同不表示LSTM/head能与不同架构逐参数同初始化；
- train batch顺序、batch size、Adam、`lr=0.001`、20 epochs上限、`patience=4`、`min_delta=0.0001`、最低validation loss checkpoint规则全部沿用；
- 现有协议中没有“10,000次评估预算”这一可执行定义，因此不引入该口径，也不以它替代冻结的epoch/early-stopping预算；
- effective模型配置必须显式记录`dropout=0.3`，不能只序列化一个未被模型使用的通用默认值。

## 6. 产物与失败规则

- 新产物只写`experiments/v2/w5/`，不覆盖W2/W3任何历史文件；
- 每个seed保存原始validation JSON、best checkpoint和run manifest；
- 保存mean±sample std、与A2-T/A2-G的paired delta、best epoch、失败记录和参数/计算量；
- 基础设施失败只允许相同seed、相同协议重跑并保留失败记录；模型数值失败不换seed掩盖；
- 结论必须逐条对照本文件判定。零结果或反向结果按原样报告。

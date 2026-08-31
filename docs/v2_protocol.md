# Wireless V2 实验协议（v6 / W0 冻结版）

> 协议版本：`wireless-v2-w0-v1`
> 冻结阶段：W0 · 协议对齐
> 冻结日期：2026-08-31
> 上位依据：`2026 年 9 月 双项目升级执行协议（v6 · 定稿）`
> 状态：预注册；本文件提交时尚未产生 W1–W4 正式训练结果。

本协议是 Wireless 九月 W1–W4 的唯一仓库内执行基线。任何旧 V2 计划与本文件冲突时，以本文件为准。实验结果可以支持任意方向的结论；在运行前不预设某个模型或消融必然更好。

## 1. 数据身份与开发边界

### 1.1 冻结数据身份

```text
dataset: RadioML 2016.10A
pickle_sha256: b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c
sample_count: 220000
sample_shape: (2, 128)
stratification: modulation × SNR
train_fraction: 0.70
validation_fraction: 0.15
test_fraction: 0.15
expected_counts: 154000 / 33000 / 33000
split_seed: 20260812
split_algorithm: sorted strata + numpy.random.default_rng(seed).permutation
```

W1 必须按上述身份确定性重建 V1 索引、保存 split manifest，并记录三组索引哈希、每个 modulation×SNR 计数、NumPy 版本和划分实现的 Git commit。正式运行只能加载冻结 manifest，不得临时重新划分。

### 1.2 `split_seed` 与 `run_seed`

- `split_seed=20260812`：只决定固定 train/validation/test 索引，不参与模型初始化、Dropout 或训练批次顺序。
- `run_seed`：只控制 Python、NumPy、PyTorch/CUDA 随机状态、模型初始化、Dropout 和训练 DataLoader 顺序，不得改变 split manifest。
- 同一个 `run_seed` 下的成对配置必须使用相同 split manifest 和相同训练样本顺序。
- 不同架构参数数量和随机数消耗不同；相同 `run_seed` 不能被解释为不同架构逐参数初始化完全相同。

### 1.3 V1 test 边界

> 从同一份 RadioML 归档内部重新划分得到的集合，只能作为内部同分布 holdout，不能作为历史独立测试证据。真正独立的 V2 最终测试需要新的数据来源或独立生成的数据。

九月不重新划分。V2 Core 只使用固定 train/validation：train 用于梯度更新，validation 用于 early stopping、模型比较、消融和错误分析。V1 test 只允许以索引、样本数和哈希形式参与集合互斥检查；禁止：

- 用 test index 调用 Dataset `__getitem__`；
- 创建 V1 test `Subset` 或 DataLoader；
- 读取 V1 test 对应信号和标签；
- 对 V1 test forward、重算指标或生成新图；
- 把 V1 test 指标用于任何 V2 选择或结论。

V1 的 Accuracy 56.38% 和 Macro F1 56.26% 只保留为 2026-08-12 的一次性历史验收记录。V2 W1–W4 只报告固定 validation 上的开发阶段结果，不称为 test 或独立泛化证据。

## 2. 预注册随机种子

正式 `run_seed` 冻结为：

```text
20260901
20260902
20260903
20260904
20260905
```

四条纪律：

1. seed 数量必须在查看任何正式结果前决定；
2. 全部 seed 的原始结果必须完整保存和报告；
3. 不删除低分 seed，不挑最高 seed 作为主结果；
4. `n=5` 只用于描述性统计，不进行显著性检验，不外推总体统计规律。

W0 的一次训练仅用于确认运行预算，不属于正式矩阵；其 stdout/stderr 被屏蔽，临时 checkpoint/JSON 在计时后删除，任何指标均不读取、不记录、不使用。如果计时显著超过约 71 秒，只能在查看正式结果前通过新协议提交调整 seed 数量，且所有配置必须使用同一列表。

## 3. 冻结实验矩阵

| 配置 | 结构与目的 | Run seeds | 正式训练数 |
| --- | --- | --- | ---: |
| A0 | 现有 `SimpleCNN1D` 整体基线 | 全部 5 个 | 5 |
| A1 / A2-T | 现有 `TemporalCNN1D`，保留 8 个时序位置；同时作为时序聚合侧 | 全部 5 个 | 5 |
| A2-G | 与 A1 共享卷积 backbone，改为全局池化和容量受控 head | 全部 5 个 | 5 |
| A3 | A1 只把 Dropout 从 `p=0.3` 改为 `p=0` | 全部 5 个 | 5 |

正式矩阵共 20 次训练。A2-T 与 A1 是同一配置，直接复用 A1 的对应 checkpoint 和指标，不重复训练。

矩阵之外的 temporal bins、Transformer、RNN、强化学习、GNU Radio、外部数据、信道扰动和广泛超参数搜索都不属于 W1–W4。不得在看到结果后向主矩阵增加配置。

## 4. 共同训练条件

以下值来自 W0 对当前实现的核对，并冻结为公平对照的共同条件：

```text
input: raw float32 I/Q, shape (batch, 2, 128)
normalization: none
augmentation: none
loss: torch.nn.CrossEntropyLoss defaults
optimizer: torch.optim.Adam
learning_rate: 0.001
betas: (0.9, 0.999)
eps: 1e-8
weight_decay: 0
amsgrad: false
scheduler: none
train_batch_size: 256
evaluation_batch_size: 512
epochs: 20
patience: 4
min_delta: 0.0001
num_workers: 0
AMP: false
deterministic_algorithms: true
cudnn_benchmark: false
checkpoint_rule: lowest validation loss
```

只有当 `validation_loss < best_validation_loss - min_delta` 时才替换 checkpoint；相等或未超过 `min_delta` 时保留更早的 best checkpoint。所有配置使用相同训练预算，不允许逐个模型调学习率、epoch、early stopping 或其他超参。

如后续发现任何上述事实与实际实现不一致，必须在正式结果产生前修正协议并提交；正式结果产生后，可能改变数值的修正必须提升协议版本，并重跑全部受影响配置与 seed。

## 5. 汇总与配对口径

### 5.1 必须保存的原始结果

每次运行至少保留 validation Accuracy、Macro F1、per-SNR Accuracy/F1、best epoch、训练历史、失败状态、checkpoint 和可追溯 manifest。汇总报告不能只保留均值。

### 5.2 报 mean ± sample std 的量

- validation Accuracy；
- validation Macro F1；
- per-SNR Accuracy 和 Macro F1；
- best epoch；
- 同 seed paired delta。

sample std 使用 `n-1` 分母。必须同时报告实际样本数 `n`；缺失配对不得补零或插值。

配对差值方向冻结为：

- 整体架构：`A1 - A0`；
- 聚合方式：`A2-T - A2-G`；
- Dropout：`A1(p=0.3) - A3(p=0)`。

### 5.3 直接报告原值的确定性量

- 数据集和各 split 样本数；
- 模型、backbone 和 head 参数量；
- MACs/FLOPs；
- 固定配置值与哈希。

确定性量不报告跨 seed 标准差。运行耗时可以逐次记录并描述运行环境，但不得与随机指标混用口径。

## 6. W3 两条独立一维消融

### 6.1 聚合方式：A2-T 与 A2-G

共享 backbone 冻结为 TemporalCNN 在最终聚合层之前的三层 Conv1d、三层 BatchNorm、ReLU 和前两次 MaxPool；对固定 128 点输入，其输出为 `(B, 128, 32)`。

```text
A2-T / A1:
  backbone -> AvgPool1d(kernel_size=4, stride=4) -> (B, 128, 8)
  Flatten(1024)
  Linear(1024, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 11)

A2-G:
  same backbone -> AdaptiveAvgPool1d(1) -> (B, 128, 1)
  Flatten(128)
  Linear(128, 947) -> ReLU -> Dropout(0.3) -> Linear(947, 11)
```

按层定义计算：A2-T head 为 132,619 个参数，A2-G head 为 132,591 个参数，相差 28 个；A1 当前全模型为 224,587 个参数，预计 A2-G 为 224,559 个。W3 实现后必须用代码重新实测。

- 可训练参数差异目标 `≤1%`，硬上限 `2%`；
- 如超限，最多调整 head 隐藏宽度 3 次；仍超限则发出告警、披露实际差值后继续，不得隐藏；
- 同时报告全模型参数量、head 参数量和 MACs/FLOPs；
- 每个 seed 的 A2-T 与 A2-G 应从同一份初始 backbone state 加载，两个 head 按冻结随机性规则初始化；
- 安全表述固定为：“在共享 backbone 与近似参数预算下的容量受控比较”。

参数量接近不等于函数表达能力完全一致，因此不得称为严格单变量实验，也不得把结果绝对归因于“保留时序位置”。

### 6.2 Dropout：A1 与 A3

A3 只把 A1 的 `Dropout(p=0.3)` 改成 `Dropout(p=0)`，其余数据、结构、初始化规则和训练条件不变。Dropout 不包含可训练参数，因此两边参数量天然相同。

聚合轴和 Dropout 轴是两个独立的一维比较，禁止合并成 2×2 网格。

## 7. 失败、修复和结果表述

- 基础设施中断只允许用相同配置 hash 和相同 seed 重跑，失败记录必须保留；
- 不得用新 seed 替代失败 seed；
- 影响输入、batch 顺序、模型计算、训练动态、checkpoint 选择或指标计算的修改属于协议变更；
- 日志文案、展示格式和路径标签等不影响数值的修复可以作为无害修复；
- 判断不清时按协议变更处理；
- 所有结论只适用于当前 RadioML 2016.10A、固定 validation、冻结配置和 5 个 seed。

禁止提前写“验证某方案有效”。正式报告必须根据原始结果描述方向、波动、成本和限制。

## 8. 阶段映射与硬闸门

- W1：实现 `split_seed` / `run_seed` 解耦、固定 manifest 和 seed 可审计入口；
- W2：完成 A0/A1 多 seed 成对实验及 mean±std；
- W3：完成 A2-T/A2-G 容量受控比较与 A1/A3 Dropout 比较；
- W4：完成 run manifest、汇总报告、README 和复现验收。

每阶段必须独立提交、更新 `docs/DEV_STATE.md` 并停止。`⑤ 拥有权验证`不是“已完成”时，禁止开始下一阶段。

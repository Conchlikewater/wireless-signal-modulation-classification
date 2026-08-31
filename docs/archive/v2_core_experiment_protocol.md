# Wireless V2 Core 实验协议

> 状态：阶段 0 冻结协议；尚未训练、评测或读取 V1 测试信号。  
> 目标：让多随机种子和消融结论可复现、可配对、不可通过换划分或挑 Seed 美化。

## 1. 开发数据边界：必须沿用 V1 train/validation

V2 Core **必须**确定性重建并沿用 V1 的原 train/validation，不再使用“优先沿用”措辞，也不重新划分全量 220,000 条数据。

冻结身份如下：

```text
RadioML pickle SHA-256:
b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c

stratification: modulation × SNR
train_fraction: 0.70
validation_fraction: 0.15
test_fraction: 0.15
split_seed: 20260812
algorithm: sorted strata + numpy.random.default_rng(seed).permutation
expected counts: 154000 / 33000 / 33000
```

阶段 1 必须把三组索引保存成 manifest，并记录索引数组 SHA-256、样本数、每个 modulation×SNR 计数、NumPy 版本和划分实现版本。后续运行加载 manifest，不在训练时重新计算划分。确定性重建结果与 manifest hash 不一致时直接失败。

V1 test 只允许以索引/哈希形式参与互斥检查，信号内容永久退出 V2 开发流程。

## 2. Run Seed：P0 固定三个，不根据结果扩展

P0 正式 `run_seed` 固定为：

```text
17, 43, 97
```

这些数字没有性能含义，只是开工前预注册的三个不同随机流。每一个 Seed 必须成对运行所有必跑模型/消融，只控制 Python/NumPy/PyTorch 初始化、Dropout 和训练 DataLoader 的样本顺序，不得改变 split manifest。

P0 不根据前三个结果决定要不要“再跑两个”。如果以后做五 Seed 扩展，追加 Seed 已预注册为 `131, 197`，并且必须对全部必跑实验补齐，单独标记为 P1 replication；不能只给喜欢的模型补跑。

## 3. Checkpoint 与跨模型选择规则

### 3.1 单次运行

继续沿用现有训练纪律：按 **最低 validation loss** 保存 best checkpoint，`patience=4`、`min_delta=0.0001`。最终汇总只读取该 best checkpoint 的 validation 指标，不使用最后一轮。

### 3.2 跨模型选择

- 唯一主指标：三个预注册 Seed 的 **mean validation Macro F1**；
- 辅助解释指标：sample std、同 Seed paired delta、Overall Accuracy、validation loss、per-class F1、per-SNR Macro F1、参数量和推理延迟；
- 辅助指标不能推翻主指标后挑一个更好看的结论。

并列规则预先固定：

1. mean Macro F1 相差至少 0.5 个百分点时，选择更高者；
2. 小于 0.5 个百分点视为工程上近似并列，优先参数量更少者；
3. 参数量相差小于 5% 时，优先同硬件上平均推理延迟更低者；
4. 仍并列时选择结构更简单的模型。

0.5 个百分点是预注册的工程容差，不宣称统计显著性。三 Seed 样本不足以做强统计结论，报告必须给原始每次结果。

## 4. 失败运行和配对差值

- 基础设施/瞬时错误（断电、磁盘、显存抢占）允许只重跑 **同一个 Seed、同一个配置 hash**，原失败 manifest 保留；
- 代码、数据或协议错误会影响公平性时，先修复并提升 `experiment_protocol_version`，然后重跑所有受影响模型和 Seed，不能只补分低的一次；
- 禁止用另一个 Seed 替换失败 Seed；
- paired delta 只对 A/B 两边都成功的同 Seed 配对计算，不插值、不补零；
- 报告必须显示 `n_pairs` 和缺失原因；少于 3 个完整配对时，不给“稳定优于”的正式结论，只能报告探索性结果。

## 5. A2 的精确结构与表述

A0 SimpleCNN 与 A1 TemporalCNN 是整体架构基线，不是严格单变量实验。

A2 定义为“**共享 TemporalCNN backbone 的容量受控聚合比较**”，不称为严格单变量消融：

### A2-T：保留时序分箱

```text
共享 backbone 输出 (B, 128, 32)
AdaptiveAvgPool1d(8)
Flatten: 1024
Linear(1024, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 11)
```

### A2-G：全局池化容量对照

```text
同一个共享 backbone 输出 (B, 128, 32)
AdaptiveAvgPool1d(1)
Flatten: 128
Linear(128, 947) -> ReLU -> Dropout(0.3) -> Linear(947, 11)
```

两者 backbone、输入、训练协议和 Seed 完全相同。按当前层定义，A2-T 与 A2-G 的 head 参数量分别约为 132,619 和 132,591，差 28 个；加上共享 backbone 后差异远小于 1%。实现后必须由 `count_trainable_parameters()` 实测，最终容差上限为 **1%**；超出就阻止正式运行。

该设计控制总参数量，但隐藏层宽度不同，仍不能把差异完全归因于“时序位置”。安全结论只能是：在共享 backbone 和近似相同参数预算下，两种聚合/head 设计的表现差异。

A3 仍是 A1 只把 Dropout 从 `0.3` 改为 `0`，其余不变。A4 temporal bins 只作为 exploratory，并披露由此产生的参数变化。

## 6. V1 Test 技术 Guard

V2 Core 允许读取以下元信息，用于证明隔离：

- V1 test 索引文件的路径、样本数和 SHA-256；
- split manifest 中 test index 的集合，用于与 train/validation 做互斥检查；
- 历史文档里已经公开的 V1 结果，只能作为带日期的历史背景。

V2 Core 禁止：

- 使用 V1 test index 调用 Dataset `__getitem__`；
- 创建 test `Subset`/`DataLoader`；
- 对 V1 test 做推理、画新图或重新计算指标；
- 把 V1 test 指标加入 V2 模型选择、消融汇总或配置调整。

Guard 的验收不是“代码里没有 test 字样”，而是训练入口只接收 train/validation manifest，报告 schema 不包含 V1 test result，测试用一个会在 test index 被取样时抛错的 Dataset 证明禁读。

## 7. External/P1 的 checkpoint 规则

External 阶段不挑一个最好 Seed。先依据 V2 Core 冻结规则选定模型结构，然后对该结构的 **17、43、97 三个预注册 checkpoint 全部**运行 `external_frozen`，报告每个原始结果和 mean ± sample std。

如果任一 Seed checkpoint 缺失，必须按第 4 节规则修复同 Seed；三个都齐全前不发布正式 external 结论。外部集结果不能反过来更换模型结构或 checkpoint。

## 8. External Frozen 的预注册和隔离

P1 生成最终数据前，必须另写 `external_generation_protocol.md`，一次性冻结并提交：

- fixture 与 frozen 各自不重叠的 `external_generation_seed` 列表；
- 5 个真实标签、每类/每 SNR/每扰动层级样本数；
- SNR 档位及精确定义；
- 采样率、samples-per-symbol、窗口长度、脉冲成形、roll-off、归一化；
- 频偏、相偏、定时偏差、多径的数值范围；
- E0/E1/E2 组合规则；
- 11 类输出下的 Accuracy、Macro F1、per-class、per-SNR 和跨出 5 类错误口径；
- 生成代码 commit、环境、配置 hash 和数据 manifest schema。

工程 fixture 与 `external_frozen` 必须使用不同生成 run、不同 Seed、不同符号序列。生成后对原始符号序列 hash 和窗口内容 hash 做交集检查，交集必须为 0。任何查看 frozen 指标后的调参都会使其降级为 external validation，并需要另建真正冻结集。

具体数值范围在 GNU Radio 环境和信号生成方式完成技术核验后冻结；在该文件提交前不得生成或查看正式 `external_frozen`。这是 P1 开工门禁，不是可以边跑边决定的参数。

## 9. P0 必跑矩阵与完成门禁

```text
A0 SimpleCNN          × seeds [17, 43, 97]
A1 TemporalCNN        × seeds [17, 43, 97]
A2-G capacity control × seeds [17, 43, 97]
A3 no-dropout         × seeds [17, 43, 97]
```

正式矩阵共 12 次运行。A2-T 与 A1 结构相同，复用 A1 的对应 checkpoint/指标，不重复训练。只有 split hash、配置 hash、全部原始结果、失败记录、mean/std 和完整 paired delta 都可追溯，V2 Core 才算完成。


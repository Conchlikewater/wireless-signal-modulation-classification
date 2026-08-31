# 2026 年 9 月升级规格：可复现的调制识别与外部鲁棒性评估

> 状态：审计修订版计划稿，尚未实施。  
> 启动条件：完成当前项目学习、能解释 Dataset/DataLoader、CNN、训练循环、早停和评测后。  
> 当前基线：RadioML 2016.10A 上的 SimpleCNN/TemporalCNN V1。本文中的 V2 功能和指标不能提前写入简历。

## 1. 当前基线与问题

V1 已完成数据安全审计、固定分层划分、PyTorch 训练、简单基线、受控模型改进、验证集选择、一次封存测试和 84 项自动化测试。

已记录结果：

| 模型/数据范围 | Accuracy | Macro F1 |
| --- | ---: | ---: |
| SimpleCNN 验证集 | 40.89% | 38.20% |
| TemporalCNN 验证集 | 56.35% | 56.23% |
| TemporalCNN V1 封存测试集 | 56.38% | 56.26% |

0 dB 以上准确率约 79%–83%，但极低 SNR 接近随机猜测，QAM16 Recall 约 4.07%，大量被识别成 QAM64。V1 能证明完整机器学习实验闭环，但仍有四个主要限制：

1. 主要结果来自单一随机种子，无法判断训练波动；
2. V1 测试集已经启封，后续不能继续用它选择模型；
3. 当前结果只属于 RadioML 2016.10A 同分布数据；
4. 公开仓库没有本机历史 checkpoint/JSON，历史数字可审查但当前不能直接重算。

V2 的目标不是“加一个更潮的网络”，而是让实验结论更可靠：

> 在固定协议下，以多随机种子、单变量消融、低 SNR/分类别错误分析和独立生成的外部信道数据，验证 TemporalCNN 的收益、稳定性和泛化边界。

## 2. 核心研究问题

V2 只回答以下问题：

1. TemporalCNN 相对 SimpleCNN 的提升，在不同随机种子下是否稳定？
2. 提升究竟来自“保留时序位置”，还是仅来自参数量增大？
3. 哪些模块和超参数真正有效，哪些没有收益？
4. 低 SNR、QAM16/QAM64、WBFM/AM-DSB 混淆的主要模式是什么？
5. 模型面对独立生成、含频偏/相偏/定时偏差/多径等扰动的数据时还能保留多少能力？

## 3. 数据分层与防泄漏

### 3.1 V1 历史结果

现有 V1 划分和最终测试结果只作为历史基线。V1 测试集已经在 2026-08-12 启封，V2 不能称它为“未见测试集”，也不能根据它继续选结构和超参数。

### 3.2 V2 RadioML 开发协议

V2 不从全部 220,000 条样本重新洗牌来制造一个所谓“全新测试集”。原 V1 test 已经启封，必须永久退出 V2 开发流程；原 V1 train/validation 已参与开发，也不能通过换名字变成历史未见测试。

V2 Core 只使用固定的 RadioML 开发边界：

- 必须沿用 V1 的原始 train/validation 索引；数据集 SHA-256、70/15/15、`split_seed=20260812` 和原划分算法按 `docs/v2_core_experiment_protocol.md` 固定；
- train 只用于梯度更新，validation 用于早停、模型选择、消融和多 seed 比较；
- 原 V1 test 索引保存为禁读清单，V2 训练与汇总入口不得加载；
- split manifest 保存样本索引、`split_seed`、每组计数和 SHA-256，所有候选模型完全共用；
- 如果未来在同一归档中保留额外内部 holdout，只能称为“同分布内部 holdout”，不能称为历史未见测试或外部泛化证据。

RadioML 2016.10A 没有原始发射序列 ID，无法完全排除相邻窗口相关性，这是数据集固有限制，必须继续披露。V2 的新最终证据来自过去未参与开发的独立来源或独立生成数据。

### 3.3 独立 GNU Radio 外部鲁棒性集

DeepSig 官方明确说明 2016/2017 年开放数据存在已知勘误，主要用于历史和教育，并建议新研究自己用 MATLAB/GNU Radio 生成数据或使用真实空口数据。因此 V2 External/P1 新建一个小型、可复现的“独立生成合成域外鲁棒性集”，不把它包装成真实空口数据，也不把 RadioML 2018.01A 直接混作当前测试集。

第一版外部集只覆盖两个模型都能支持的重叠数字调制：

- BPSK、QPSK、8PSK、QAM16、QAM64；
- 固定输出 `(2, 128)` float32 I/Q；
- 多档 SNR；
- 可控 AWGN、归一化频偏、相位偏差、采样时钟偏差；
- 可选简单多径/FIR taps；
- 每个生成 run 保存独立 seed 和 channel config。

模型仍保持 11 类输出；评测只统计这 5 个真实标签样本，但预测成其余 6 类同样计为错误，不能临时重定义成更容易的 5 类分类器。固定采样率、samples-per-symbol、脉冲成形、roll-off、幅度归一化、SNR 定义和扰动范围，并按以下层次生成：

- E0：尽量对齐的干净/AWGN 基线；
- E1：一次只加入频偏、相偏、定时偏差或多径中的一种；
- E2：组合扰动。

划分必须按“生成 run/`external_generation_seed`/channel config”分组，而不是把同一段长信号切成窗口后随机分散。先建立可反复运行的工程调试 fixture；最终 `external_frozen` 在生成统计、标签和信号级测试通过后冻结，只运行最终鲁棒性报告。如果查看结果后继续改模型，它就降级为 external validation，不能再作为最终证据。生成器正确性由统计、标签和信号级测试验证，不能用模型准确率反推生成器正确。

外部数据目录同样 Git 忽略；仓库只保留生成脚本、配置、manifest、hash、统计和可公开的小型测试 fixture。

## 4. 实验协议

### 4.1 固定条件

对照实验必须固定：

- 数据索引与预处理；
- batch size、优化器、基础学习率策略；
- epoch 上限、早停规则、最佳模型选择标准；
- 输入 shape、类别顺序；
- 指标实现与报告脚本；
- 硬件、PyTorch/CUDA 版本和确定性设置。

不得为每个模型私下使用不同数据或不透明调参，再把结果写成公平对照。

### 4.2 多随机种子

随机性职责必须拆开，不能让一个 `seed` 同时改变数据划分和训练过程：

- `split_seed`：固定一次，只决定开发索引；
- `run_seed`：P0 预先冻结为 `17, 43, 97`，只控制模型初始化、Dropout 和训练数据顺序；如以后做 P1 五 Seed replication，追加值固定为 `131, 197`，不得根据前三次分数决定；
- `external_generation_seed`：只控制外部数据生成。

同一个 `run_seed` 必须成对运行 SimpleCNN 与 TemporalCNN。每个模型报告：

- Validation Accuracy/Macro F1 的 mean ± std；
- 最佳 epoch、训练耗时和参数量；
- per-SNR Accuracy 的 mean ± std；
- per-class Precision/Recall/F1 的 mean ± std；
- 每次运行的配置、checkpoint 和结果 hash。

同时报告同 seed 下的 `TemporalCNN - SimpleCNN` 差值、best epoch 和失败运行。多种子不是为了挑最高分，而是展示结果是否稳定；主结论使用均值、样本标准差和配对差值，不使用单次最好结果。

### 4.3 单变量消融

启动前冻结必跑矩阵、run seed 列表和停止条件；额外实验统一标为 exploratory，避免看到结果后选择性报告。A0/A1 是整体基线对比，并非严格单变量实验，因为它们同时改变通道数、卷积层数、BatchNorm、Dropout、Pooling 和参数量。建议矩阵：

| 实验 | 变化与控制方式 | 要回答的问题 |
| --- | --- | --- |
| A0 | SimpleCNN 整体基线 | 历史基线能否在固定开发协议重现 |
| A1 | TemporalCNN 整体基线 | 整体架构变化是否稳定优于 A0；不把差异只归因于时序位置 |
| A2 | 共享 TemporalCNN 卷积 backbone，只替换聚合/head，并尽量匹配参数量 | 聚合方式的影响是否仍存在 |
| A3 | TemporalCNN 仅将 dropout `p=0.3` 改为 `p=0` | 正则化对稳定性和过拟合的影响 |
| A4 | 改变 temporal bins，并补偿 classifier 宽度或明确披露参数变化 | 过早压缩与保留时序信息的取舍 |

如果 A2/A4 难以做到严格参数量匹配，应报告实际参数差异，不得用“完全公平”包装。V2 Core 必跑 A0、A1、A2、A3；A4 属于时间允许时的 exploratory 实验。

上述划分身份、Seed、best checkpoint、主指标、并列规则、失败重跑、A2 精确结构、V1 test guard 和 External checkpoint 规则已冻结在 `docs/v2_core_experiment_protocol.md`，实施时以该协议为准。

### 4.4 是否加入新模型

V2 不强制 Transformer。当前项目已经能证明 PyTorch、CNN、训练与评测能力。只有在完成多种子和消融后仍有时间，才增加一个轻量候选：

- 1D ResNet：验证残差连接是否改善深层时序特征；或
- 小型 CLDNN/CNN-RNN：验证卷积局部特征加序列建模的收益；或
- 有明确论文假设的轻量 attention。

三者只选一个。加入前先写假设和资源预算，保持相同协议。没有稳定收益就诚实保留负向实验。

强化学习不属于当前分类问题。RL 需要环境、动作、状态和奖励，例如做自适应调制或资源分配；那将是另一个项目，不能把 DQN 名称硬加进调制分类器。

## 5. 训练与产物管理

V2 所有实验从头训练，不依赖缺失的历史 checkpoint。每个运行目录必须唯一且拒绝覆盖，至少保存：

```text
run_id/
├── config.json
├── environment.json
├── split_manifest.json
├── train_history.json
├── best_checkpoint.pt        # 本地/GitHub Release，不进普通 Git
├── checkpoint.sha256
├── validation_result.json
├── metrics_by_snr.json
├── metrics_by_class.json
├── confusion_matrix.npy
└── run_summary.md
```

总报告只能读取这些机器可验证产物生成，不能手工抄写结果。checkpoint 太大时放 GitHub Release 或外部对象存储；仓库保留 hash、下载/重训说明和版本对应关系。

需要新增一个 experiment registry/manifest。每个 run 独立写入 manifest，全部完成后再汇总，避免并行写同一个 registry；记录模型、`split_seed`、`run_seed`、Git commit、数据 hash、配置 hash、运行状态、开始结束时间、最佳 epoch、产物路径和失败原因。

## 6. 评测与错误分析

### 6.1 固定指标

- Overall Accuracy；
- Macro Precision、Recall、F1；
- 每类别 Precision/Recall/F1；
- 混淆矩阵；
- 每 SNR Accuracy 和 Macro F1；
- 参数量、模型大小、训练时间和推理延迟；
- 多种子 mean ± std。

### 6.2 重点错误切片

报告必须单独分析：

- 极低 SNR 接近随机水平是否符合信息受噪声主导的预期；
- QAM16 被预测为 QAM64 的比例和可能原因；
- WBFM 与 AM-DSB 的混淆；
- AM-SSB 高 Recall、低 Precision 的含义；
- 总 Accuracy 是否掩盖某些类别失败；
- 不同 seed 下困难类别是否一致。

不能只说“准确率提高了”，必须区分：整体收益、困难类收益、低 SNR 收益、稳定性和成本。

### 6.3 最终验收顺序

1. 只用固定 RadioML train/validation 开发边界完成所有设计；
2. 原 V1 test 始终不进入 V2 数据加载、模型选择和报告生成；
3. 冻结必跑矩阵、run seeds、checkpoint hash 和最终协议；
4. V2 Core 报告 validation 多 seed、配对差值、消融和错误分析，不伪造一个“新 V2 test”；
5. V2 External 完成工程 fixture 后冻结 `external_frozen`，再运行一次最终外部鲁棒性评估；
6. RadioML 开发结果与 5 类外部子集结果分表报告，禁止混成一个数字或直接比较总体 Accuracy。

## 7. 测试与质量保证

在现有 84 项测试上新增：

- V2 开发 split manifest 稳定性、互斥、hash 和 V1 test 禁读测试；
- 分组划分防止同一生成 run 泄漏的测试；
- 多 seed 配置冻结和批量实验编排测试；
- 新模型 shape、梯度、参数量和确定性测试；
- experiment registry、产物完整性、拒绝覆盖测试；
- mean/std、per-SNR、per-class 汇总测试；
- checkpoint/config/data hash 绑定测试；
- external final guard，防止未冻结或开发阶段读取 `external_frozen`；
- GNU Radio 生成配置、标签、shape、有限值、类别与 SNR 统计测试；
- 图表和 Markdown 报告从 JSON 生成的测试。

CI 仍不依赖完整 RadioML 数据、GPU 或数小时训练。CI 只用小型合成 fixture 验证代码路径；真实训练通过手动、可审计的实验流程完成。

## 8. 实施顺序

### 阶段 0：协议和 Git 边界

- 让 V1 tag 指向最后一个纯 V1 提交 `f9668db`，实施前再次核对 commit；
- 保存当前代码、数据和历史结果边界；
- 固定开发索引，建立 V1 test 禁读清单，不重新洗牌全部 220,000 条；
- 按 `docs/v2_core_experiment_protocol.md` 冻结 `split_seed=20260812`、P0 `run_seed=[17,43,97]`、12 次必跑矩阵、训练/模型选择和失败重跑规则；
- 建立 V2 config、run manifest 和产物完整性 schema。

### 阶段 1：固定 manifest、V2 训练入口与产物链路

- 保存/加载固定 split manifest，拒绝每次运行重新划分；
- 新增独立 `train_radioml_v2.py`，不破坏 V1 脚本；
- 让训练脚本显式接收 run ID、`split_seed` 和 `run_seed`；
- 保存完整配置、环境、hash 和失败原因；
- 每个 run 独立写 manifest，最后生成 registry。

### 阶段 2：V2 Core/P0——多种子、消融与错误分析

- 以相同 run seeds 成对重训 SimpleCNN 与 TemporalCNN；
- 生成 mean、样本标准差和 paired delta 报告；
- 完成 A2 参数控制与 A3 Dropout 单变量消融；A4 仅作为 exploratory；
- 自动生成混淆矩阵、分类别与 per-SNR 报告；
- 整理稳定 bad cases 和负向结果。

到这里构成 **V2 Core/P0**：可以开始投递，不需要等待 GNU Radio。

### 阶段 3：V2 External/P1——外部数据生成

- 取得许可后安装 GNU Radio 或采用可审计的 GNU Radio 环境；
- 固定波形、采样、信道、`external_generation_seed` 和数据 manifest；
- 先建立工程 fixture，再冻结独立 `external_frozen`；
- 按生成 run 分组；
- 完成 E0/E1/E2 外部鲁棒性集及信号级测试。

### 阶段 4：V2 External/P1——冻结评估

- 根据 RadioML validation 多种子结果先冻结模型结构，再对该结构的 `17, 43, 97` 三个 checkpoint 全部运行外部评测，不挑最好 Seed；
- 保持 11 类输出运行 `external_frozen`，预测到另外 6 类计为错误；
- 分表报告 RadioML 开发结果与 5 类外部子集，不把外部集用于继续调参；
- 时间足够时只增加一个轻量新模型，并重新走完整协议。

### 阶段 5：作品集交付

- 更新 README、实验矩阵、复现命令和数据许可；
- 提供一键生成结果表和图表的脚本；
- 给出 checkpoint 获取/重训方式；
- 准备 1 分钟、3 分钟项目介绍和技术追问；
- 只把真实运行并保存报告的数字写进简历。

## 9. 完成定义

完成分为两个里程碑。

### 9.1 V2 Core/P0 完成定义

1. SimpleCNN/TemporalCNN 至少 3 seeds 完整重训并保存配置、hash 和结果；
2. `split_seed` 与 `run_seed` 职责分离，同 seed 成对运行并报告 mean、sample std 和 paired delta；
3. 完成一个共享 backbone 的参数/聚合控制实验和一个 Dropout 单变量消融；
4. 每个结论同时有整体、分类别、per-SNR 和混淆矩阵证据；
5. 原 V1 test 被技术 guard 和实验协议排除，报告不声称存在新的 RadioML 历史未见 test；
6. checkpoint、配置、数据和结果 hash 可以互相追溯；
7. 自动化测试和 CI 通过，仓库不包含受限原始数据；
8. 用户能不依赖 AI 解释多 seed、消融、公平对照、低 SNR、混淆矩阵和数据泄漏。

### 9.2 V2 External/P1 完成定义

1. 独立生成的合成域外集按生成 run 隔离，fixture 与 `external_frozen` 分离；
2. E0/E1/E2 配置、标签、shape、统计和信号级测试通过；
3. 评测保持 11 类输出，5 类外部子集中的跨出类别预测计为错误；
4. 外部集不参与训练、早停或参数选择；
5. RadioML 开发结果与外部结果分开报告，真实边界说明完整；
6. 用户能解释独立生成合成数据能证明什么、不能证明什么。

## 10. 明确不做

- 不把 V1 已启封测试集继续用于模型选择；
- 不从同一份 RadioML 2016.10A 全量重新洗牌并包装成“全新未见测试集”；
- 不只运行更多 epoch 作为“升级”；
- 不同时堆 Transformer、ResNet、RNN、Attention；
- 不做强化学习/DQN；
- 不把 RadioML 2018.01A 直接当成当前 11 类模型的可比测试集；
- 不从未知网盘下载 Pickle；
- 不提交 RadioML 原始数据或违反 CC BY-NC-SA 4.0；
- 不把 GNU Radio 合成数据描述成真实空口生产数据；
- 不以一次最好 seed 或只截取高 SNR 区间夸大效果。

## 11. 面试必须能回答的决策

1. 为什么 V1 的 56.38% 不能继续用于 V2 调参？
2. 多随机种子解决什么问题，为什么不能挑最好的一次？
3. TemporalCNN 的提升为什么可能只是参数量增加？如何做控制？
4. 为什么 QAM16 容易与 QAM64 混淆？现有证据能说明到什么程度？
5. 为什么 Overall Accuracy 不够，Macro F1 和 per-SNR 有什么作用？
6. 为什么外部数据要按生成 run/seed 划分，而不是随机窗口划分？
7. RadioML validation 多种子结果和 GNU Radio 外部测试分别能证明什么？
8. 为什么暂时不需要 Transformer，RL 又为什么不属于这个分类任务？
9. checkpoint 没有提交 Git 时如何保证结果可追溯？
10. DeepSig 已提示历史数据有勘误，为什么还使用 RadioML 2016.10A？

## 12. 官方依据（实施时再次核验版本）

- DeepSig Datasets：<https://www.deepsig.ai/datasets/>
- GNU Radio Channel Model：<https://wiki.gnuradio.org/index.php/Channel_Model>
- RadioML 2016.10A 生成代码：<https://github.com/radioML/dataset/blob/master/generate_RML2016.10a.py>

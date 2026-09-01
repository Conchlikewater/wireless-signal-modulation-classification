# Wireless Signal Modulation Classification

基于PyTorch和RadioML 2016.10A的无线信号自动调制识别项目，重点展示固定数据边界、多随机种子成对实验、容量受控消融、SNR错误分析和可审计复现。

## 当前状态

V2 Core的W0–W4开发、复现验收和学习交接已完成，当前进行最终冻结前审计。项目使用固定的154,000条train和33,000条validation进行开发；V1阶段已经启封的33,000条test永久退出V2模型选择和调参流程。

V2正式运行包含4个配置×5个预注册run seed：

- A0：SimpleCNN；
- A1/A2-T：保留8个粗粒度时序分箱的TemporalCNN；
- A2-G：共享TemporalCNN backbone、改用全局池化和容量受控head；
- A3：仅把A1的Dropout从`p=0.3`改为`p=0`。

所有原始JSON、best checkpoint、初始化backbone、汇总和20份run manifest均已提交。135项离线自动化测试通过。

## V2 Core结果

下表全部是固定validation上的五seed `mean ± sample std`，不是新的test成绩，也不表示统计显著性。

| 配置 | Validation Accuracy | Validation Macro F1 | 参数量 | 技术判断 |
|---|---:|---:|---:|---|
| A0 SimpleCNN | 42.26% ± 2.08 pp | 40.24% ± 2.52 pp | 11,499 | 教学基线 |
| A1/A2-T TemporalCNN | 54.81% ± 0.64 pp | 55.47% ± 1.05 pp | 224,587 | 当前默认模型 |
| A2-G 全局池化 | 55.00% ± 0.63 pp | 55.73% ± 0.64 pp | 224,559 | 差异小于波动，不promote |
| A3 无Dropout | 53.91% ± 2.30 pp | 54.40% ± 3.13 pp | 224,587 | 均值下降、波动增大，不promote |

主要成对结果：

- `A1−A0` Macro F1 paired delta：`+15.23 ± 3.35`个百分点，`n_pairs=5`，五个seed方向全部为正；
- `A2-T−A2-G` Macro F1 paired delta：`-0.25 ± 1.38`个百分点，方向不一致；
- `A2-T(p=0.3)−A3(p=0)` Macro F1 paired delta：`+1.08 ± 3.54`个百分点，方向不完全一致。

安全结论是保留A1/A2-T。A2-G的平均分略高，但约`0.25`个百分点的差异小于跨seed波动，不能包装成稳定提升；移除Dropout也没有得到可靠收益。

完整原始结果、per-SNR数据、参数/MACs、延迟、限制和证据哈希见[`docs/V2_CORE_REPORT.md`](docs/V2_CORE_REPORT.md)。

## 核心技术取舍清单

1. **固定数据身份，不重新洗牌制造“新test”**：V2沿用V1 train/validation边界；同一RadioML归档内部重新切分不能冒充真正独立测试证据。
2. **分离`split_seed`与`run_seed`**：`split_seed=20260812`只决定固定索引；五个`run_seed`只控制模型初始化、Dropout和训练batch顺序。
3. **结果产生前预注册seed和矩阵**：Git提交早于正式结果，避免看到成绩后挑seed、删失败运行或改变比较规则。
4. **跨模型看Macro F1 mean±sample std和paired delta**：不使用单次最高Accuracy选择模型；五个seed只做描述性统计。
5. **A2只称容量受控比较**：A2-T/A2-G共享backbone且全模型参数只差28个，但head函数结构不同，不能称严格单变量实验。
6. **不为“新模型”强行promote**：A2-G没有稳定优势，A3无Dropout没有可靠收益，因此默认仍保留A1/A2-T。
7. **区分理论计算量和真实延迟**：参数量/MACs是确定性结构量；延迟受硬件、算子和运行环境影响。
8. **manifest保证可审计，不夸大为绝对复现**：运行身份、配置、环境和文件哈希可以重建；跨硬件bitwise一致不作保证。

## 项目能力范围

项目能够证明：

- NumPy/PyTorch数据处理、Dataset与DataLoader；
- `(B,2,128)` I/Q序列上的Conv1d模型和Tensor shape管理；
- CrossEntropy、backward、Adam、validation、early stopping和best checkpoint；
- Accuracy、Macro F1、混淆矩阵、per-SNR指标；
- 固定分层划分、数据泄漏边界和测试集纪律；
- 多seed成对实验、sample std、controlled ablation；
- 参数量、MACs/FLOPs和固定硬件推理延迟分析；
- 原始实验产物、哈希、run manifest、自动化测试和CI。

项目不能证明：

- 真实空口、生产部署或跨设备泛化；
- SOTA、统计显著的算法创新或研究型新模型；
- Transformer、强化学习或大型网络经验；
- V2新的独立最终test成绩。

## 数据与边界

RadioML 2016.10A包含220,000条样本、11种调制方式和20个SNR条件。单条样本是`(2,128)`：2表示I/Q两个通道，128表示时间采样点。

固定分层划分：

| split | 样本数 | V2用途 |
|---|---:|---|
| train | 154,000 | 梯度更新 |
| validation | 33,000 | early stopping、模型比较、消融和错误分析 |
| V1 test | 33,000 | 仅保留历史一次性验收；V2禁止重新使用 |

- 数据SHA-256：`b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c`；
- split manifest SHA-256：`48ad195d5552e3ec4e5a6d1bc4fc0f20099df8dc70f8eb78a80df95e7f5297a7`；
- 分层单元：modulation×SNR，共220个，每单元`700/150/150`；
- V2开发入口不返回test索引、Subset或DataLoader。

RadioML 2016.10A来自[DeepSig官方数据页面](https://www.deepsig.ai/datasets/)，许可为CC BY-NC-SA 4.0。仓库MIT许可证只覆盖本项目代码，不改变数据集许可证。仓库不重新分发原始数据；使用者需从官方渠道取得数据，并放置为：

```text
data/raw/RML2016.10a/RML2016.10a_dict.pkl
```

安全加载、归档审计和已知限制见[`docs/DATASET_AUDIT.md`](docs/DATASET_AUDIT.md)。不要对来源不明的Pickle直接调用普通`pickle.load`。

## 环境安装

已验证正式实验环境：Python 3.12.10、NumPy 2.5.2、PyTorch 2.12.1+cu130、CUDA 13.0、cuDNN 9.2、NVIDIA GeForce RTX 4060 Laptop GPU。

CPU环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Windows + NVIDIA GPU环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

顶层依赖已固定版本；manifest还保存了实际Python、NumPy、PyTorch、CUDA、cuDNN、GPU和依赖规格文件哈希。它不是包含驱动与全部传递依赖的跨平台容器镜像，因此不承诺任意机器逐位相同。

## 自动化测试与CI

全部135项离线测试不需要RadioML数据、GPU或API密钥：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

GitHub Actions在Python 3.12上安装项目、运行同一套离线测试，并解析全部Python源码。产物测试会重算W2/W3汇总、加载checkpoint、核对初始化backbone以及验证20份manifest的依赖和历史文件哈希。

## 从run manifest复现一次历史运行

[`experiments/v2/run_manifests/catalog.json`](experiments/v2/run_manifests/catalog.json)登记了4个配置×5个seed共20次历史运行。每份manifest包含数据、split、seed、训练配置、优化器、环境、Git commit、原始JSON、checkpoint和依赖哈希。

先做安全dry-run：

```powershell
.\.venv\Scripts\python.exe scripts\replay_v2_run.py `
  experiments\v2\run_manifests\w3-a2-g-20260901.json `
  --data-file data\raw\RML2016.10a\RML2016.10a_dict.pkl `
  --output-directory artifacts\replay\w3-a2-g-20260901
```

dry-run只验证哈希并打印可执行命令，不反序列化数据、不训练、不访问V1 test。

确实需要重新训练时，检查预计GPU时间和磁盘空间，然后在新输出目录上显式增加：

```text
--execute
```

脚本只重新拉起manifest指定的单个配置和单个seed。它仍只创建train/validation DataLoader；输出目录已存在时拒绝覆盖。本仓库W4验收只执行了dry-run，没有重新训练。

## 训练与评测链路

```text
RadioML Pickle + SHA-256
-> restricted loader / schema audit
-> frozen split manifest
-> train/validation Dataset
-> seeded DataLoader
-> CNN forward: (B,2,128) -> logits (B,11)
-> CrossEntropy
-> zero_grad -> backward -> Adam step
-> validation loss
-> early stopping / best checkpoint
-> fixed-validation Accuracy + Macro F1
-> confusion matrix + per-SNR
-> raw JSON + checkpoint hash + run manifest
```

## V1一次性历史验收

V1在模型与协议冻结后曾对33,000条test执行一次最终验收：

| 历史模型/范围 | Accuracy | Macro F1 |
|---|---:|---:|
| V1 SimpleCNN validation | 40.89% | 38.20% |
| V1 TemporalCNN validation | 56.35% | 56.23% |
| V1 TemporalCNN一次性历史test | 56.38% | 56.26% |

V1 test结果没有用于当时继续调参，但test已经启封，因此V2禁止重新用它选择模型。V2的五seedvalidation结果与V1历史test结果必须分开表述。

![V1 TemporalCNN历史test SNR曲线](docs/images/final_test/final_test_accuracy_by_snr.svg)

![V1 TemporalCNN历史test混淆矩阵](docs/images/final_test/final_test_confusion_matrix.svg)

## 结果限制

- 极低SNR接近11分类机会水平；A2-T在`-20 dB`的五seed平均Accuracy约9.28%；
- A2-T在`0 dB`及以上平均Accuracy约77%–82%，但仍存在QAM16/QAM64和WBFM/AM-DSB等混淆；
- 所有V2结果只适用于RadioML 2016.10A固定validation和五个预注册seed；
- 没有真实空口、外部独立数据或生产服务结论；
- 参数量接近不等于函数表达能力相同；
- 五个seed不支持统计显著性结论。

## 目录

```text
src/signal_modulation/        核心数据、模型、训练、统计与manifest代码
scripts/                      审计、训练、报告和单次manifest复现入口
tests/                        135项离线自动化测试
manifests/v2/                 固定split manifest与索引
experiments/v2/w2/            A0/A1五seed原始JSON与checkpoint
experiments/v2/w3/            A2-G/A3结果、共享初始backbone与W3汇总
experiments/v2/run_manifests/ 20份单次运行manifest与catalog
docs/                         协议、技术报告、开发状态和学习材料
data/                         原始数据不提交Git
artifacts/                    临时重放和本地运行输出，不覆盖正式产物
.github/workflows/            离线CI
```

## 学习路径

基础链路：

1. `docs/LEARNING_01_IQ_SNR.md`
2. `docs/LEARNING_02_DATA_SPLIT.md`
3. `docs/LEARNING_03_DATASET_DATALOADER.md`
4. `docs/LEARNING_04_CNN_FORWARD.md`
5. `docs/LEARNING_05_TRAINING_LOOP.md`
6. `docs/LEARNING_06_BEST_MODEL_EARLY_STOPPING.md`
7. `docs/LEARNING_07_RADIOML_DATASET.md`
8. `docs/LEARNING_08_RADIOML_DATALOADER.md`
9. `docs/LEARNING_09_SMOKE_TRAINING.md`
10. `docs/LEARNING_10_EVALUATION_METRICS.md`

V2重点：

- [`docs/v2_protocol.md`](docs/v2_protocol.md)：冻结协议；
- [`docs/W2_LEARNING_HANDOFF.md`](docs/W2_LEARNING_HANDOFF.md)：多seed成对实验；
- [`docs/W3_LEARNING_HANDOFF.md`](docs/W3_LEARNING_HANDOFF.md)：容量受控消融；
- [`docs/V2_CORE_REPORT.md`](docs/V2_CORE_REPORT.md)：完整结果与限制；
- [`docs/DEV_STATE.md`](docs/DEV_STATE.md)：阶段、证据和决策记录。

## 暂不包含

- Transformer、强化学习或大型网络；
- GNU Radio外部集或真实空口采集；
- Web服务、实验数据库、仪表盘或模型注册服务；
- 未经独立数据验证的泛化和性能宣传。

未来P1若继续，优先使用预先冻结的独立生成或独立来源数据验证域外鲁棒性；在此之前不继续查看V1 test，也不为了刷一次最高分扩大模型复杂度。

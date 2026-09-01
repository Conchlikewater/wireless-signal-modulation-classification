# Wireless 开发状态

## 当前阶段

- 当前阶段：W4 · run manifest、报告、README与复现验收（已获用户授权）
- 真实开工日期：2026-08-31
- ⑤ 拥有权验证：未完成
- 已完成阶段：W0、W1、W2、W3

## 每阶段关键决策及理由

### W0

- 采用《2026 年 9 月 双项目升级执行协议 v6》作为唯一上位基线，旧计划只作归档背景。
- 正式 `run_seed` 预注册为 `20260901` 至 `20260905`；五个 seed 只做描述性统计，避免结果产生后挑 seed。
- 固定沿用 V1 的 train/validation 开发边界；V1 test 仅允许索引/哈希互斥检查，禁止读取信号、标签或重新评测。
- 实验矩阵冻结为 A0、A1/A2-T、A2-G、A3 共 4 个配置 × 5 seed；A2-T 复用 A1，不重复训练。
- W3 只保留聚合方式和 Dropout 两条独立一维比较，不扩展 temporal bins 或外部数据。
- 2026-08-31 仅计时运行耗时 88.45 秒；20次矩阵线性估算约29.5分钟，仍保留5个正式 seed。计时输出被屏蔽，临时产物已删除，未读取或使用指标。
- 旧 V2 文档整体移入 `docs/archive/`，不逐份改写，也不再作为执行依据。
- 预注册协议提交：`3cfc1c4`；旧文档归档提交：`6157c72`。两者都早于 W1–W4 的任何正式结果。

### W1

- 保留 V1 的 `BaselineExperimentConfig.seed` 和原训练入口，避免为 V2 改坏历史复现；V2 使用独立的 `V2ExperimentConfig`。
- `split_seed=20260812` 只校验固定 split manifest；五个 `run_seed` 只进入全局训练随机状态和训练 DataLoader 顺序。
- 固定划分保存为 `manifests/v2/split_manifest.json` 与 `split_indices.npz`。后续正式运行必须加载该产物，不得运行时重新划分。
- V2 数据入口只返回 train/validation DataLoader；test 只保留数量和索引哈希，不返回 test 索引、`Subset` 或 DataLoader。
- split manifest 生成脚本只依据数据文件哈希、分组键、组大小、调制/SNR 元数据和冻结算法构造索引；未调用任何 test 样本的 `__getitem__`，未训练或评测模型。
- 划分实现提交：`3892b3b`。它晚于 W0 预注册提交 `3cfc1c4`，且早于任何 W2–W4 正式训练结果。

### W2

- 只运行预注册的 A0 `SimpleCNN1D` 与 A1 `TemporalCNN1D`，每个模型完整运行 `20260901` 至 `20260905` 五个 seed，共10次。
- 每个同 seed 的 A0/A1加载同一split manifest，并由独立同seed DataLoader generator产生相同训练batch顺序；未宣称不同架构逐参数初始化相同。
- 每次运行按最低validation loss保留best checkpoint，沿用20 epochs、patience 4和min_delta 0.0001；没有为某个模型单独调参。
- 10次运行全部成功，零失败、零重跑、零seed替换。原始JSON、best checkpoint、matrix lock和汇总均保存在`experiments/v2/w2/`。
- W2训练与统计实现提交：`616921f`；原始JSON提交：`8805511`；checkpoint提交：`f15428f`。实验实现提交早于全部正式结果。
- 全部W2指标都是固定validation开发结果；未创建test DataLoader，未运行V1 test推理，也未把历史test指标加入汇总。

### W3

- 只新增 A2-G 和 A3 各五个预注册 seed，共10次正式训练；A2-T严格复用W2的A1五次结果，没有重复训练或扩成2×2网格。
- A2-G与A2-T共享三层Conv1d/BatchNorm backbone；每个seed保存并加载确定性重建的同一份初始backbone。A3与A1的完整初始state hash相同，只把Dropout从`p=0.3`改为`p=0`。
- A2-T/A2-G全模型参数量为`224,587 / 224,559`，相差28个（`0.0125%`）；head参数为`132,619 / 132,591`，满足目标`≤1%`。A2仍只称“共享backbone与近似参数预算下的容量受控比较”，不称严格单变量实验。
- 10次运行全部成功，零失败、零重跑、零seed替换；原始JSON、best checkpoint、初始化backbone、矩阵锁与汇总保存在`experiments/v2/w3/`。
- W3实现提交`596900a`早于全部正式结果；原始JSON提交`a4cd7ea`，checkpoint和初始backbone提交`3d68864`。
- 全部W3指标都是固定validation开发结果；未创建test DataLoader、未运行V1 test推理，也未把历史test指标加入汇总。

## W2 描述性结果

以下均为五个seed的`mean ± sample std`，只适用于当前RadioML固定validation，不表示统计显著性：

- A0 Accuracy：`42.26% ± 2.08`个百分点；Macro F1：`40.24% ± 2.52`个百分点。
- A1 Accuracy：`54.81% ± 0.64`个百分点；Macro F1：`55.47% ± 1.05`个百分点。
- 同seed `A1-A0` Accuracy paired delta：`+12.55 ± 2.57`个百分点，`n_pairs=5`。
- 同seed `A1-A0` Macro F1 paired delta：`+15.23 ± 3.35`个百分点，`n_pairs=5`。
- 五个paired delta方向全部为正；安全结论仅为“在本协议和五个seed下观察到一致方向”，不做显著性或普适性外推。
- A0 best epoch：`20.0 ± 0.0`；A1 best epoch：`9.0 ± 2.74`。
- 参数量是确定性原值：A0 `11,499`，A1 `224,587`，不对参数量报告std。
- 在`-20`至`-14 dB`两者都接近机会水平；从`-12 dB`开始差距增大，A1在`0 dB`及以上平均Accuracy约`77%–82%`。这是数据条件下的描述，不是所有无线环境的结论。
- 10次训练记录的累计GPU训练与validation时间为`623.91`秒，平均`62.39`秒/次；W2产物约`13.84 MB`。
- `w2_summary.json` SHA-256：`7c5f2513ae78903abad0781801e4f5577ec009a7203cd7a56ac5503bbaa1c00f`。

## W3 描述性结果与技术判断

以下均为五个seed的固定validation结果，只作描述性比较，不表示统计显著性：

- A2-T Accuracy：`54.81% ± 0.64`个百分点；Macro F1：`55.47% ± 1.05`个百分点。
- A2-G Accuracy：`55.00% ± 0.63`个百分点；Macro F1：`55.73% ± 0.64`个百分点。
- 同seed `A2-T - A2-G` Accuracy paired delta：`-0.18 ± 0.57`个百分点；Macro F1 paired delta：`-0.25 ± 1.38`个百分点，`n_pairs=5`。
- A2聚合比较的差异很小且Macro F1方向不一致（3个seed为A2-T更高、2个seed为A2-G更高）。当前证据不支持把平均值的微小差异解释成稳定结构优势，因此不promote A2-G。
- A3 Accuracy：`53.91% ± 2.30`个百分点；Macro F1：`54.40% ± 3.13`个百分点。
- 同seed `A2-T(p=0.3) - A3(p=0)` Accuracy paired delta：`+0.91 ± 2.41`个百分点；Macro F1 paired delta：`+1.08 ± 3.54`个百分点，`n_pairs=5`。
- 去除Dropout后均值下降且波动明显增大，但seed方向仍不完全一致；安全结论是本协议下没有证据支持去掉Dropout，不能外推为“Dropout普遍有效”。
- Conv1d/Linear MACs：A2-T/A3为`4,441,472`，A2-G为`4,440,625`；按`2 FLOPs/MAC`约为`8,882,944 / 8,881,250` FLOPs。统计不含BatchNorm、激活、池化、Dropout和bias加法。
- 固定512样本全零输入、RTX 4060、单个预注册参考seed checkpoint的平均批次延迟：A2-T `1.487 ms`、A2-G `1.514 ms`、A3 `1.472 ms`。这是单次环境测量，不报告跨seed std，也不据此宣称普遍速度优势。
- 10次新增训练累计记录时间`428.40`秒，平均`42.84`秒/次；W3全部产物约`29.28 MB`。
- `w3_summary.json` SHA-256：`b0edff36b715f68599ceb8076e371a80994ac0af4ae02e4d7af67d357eaeccc8`。

## W1 固定划分产物

- 数据 SHA-256：`b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c`
- 样本数：train `154000` / validation `33000` / sealed test metadata `33000`
- 分层单元：220 个 modulation×SNR 单元，每个单元 `700 / 150 / 150`
- manifest SHA-256：`48ad195d5552e3ec4e5a6d1bc4fc0f20099df8dc70f8eb78a80df95e7f5297a7`
- train index SHA-256：`8b605c73b4c31f49047189e13c99a77218881244756003422913fc376b2c2e69`
- validation index SHA-256：`ef315218d7678da5d550b834379b9f8aa16ca6f326a2ffc857c6901e495b6641`
- test index SHA-256：`16253cdc095d2b2283159b14a57aeb3ab4ef83e0038ea630f13e6f32f6a9ca7f`
- indices archive SHA-256：`aae52254ba088464ca60e60c9c97a6f1c8ed64d285f086d99fc597cd61f324bb`

## 未决问题

- W3 开发、训练和产物验收没有阻塞项；⑤拥有权验证仍为“未完成”，因此W4尚未开始。
- 独立 `.venv` 没有第三方 `pytest`，但仓库测试实际使用 Python 标准库 `unittest`；无需安装依赖即可执行。W1 完整回归共 96 项。
- `compileall` 额外检查因现有 `__pycache__` 写权限被系统拒绝，未作为通过项；96 项测试已经实际导入并执行全部新增模块。

## 下一阶段入口条件

只有同时满足以下条件，才能由用户明确下令开始 W4：

1. [x] A2-G/A3各五个预注册seed全部运行，A2-T只复用W2结果；
2. [x] 两组同seed配对均完整，`n_pairs=5`；
3. [x] 参数量、head参数、MACs/FLOPs、固定批次延迟和初始化哈希已记录；
4. [x] 10个新checkpoint及5个初始backbone哈希匹配、可加载；
5. [x] W3产物完整性测试及127项全量离线测试通过；
6. [x] W3学习交接包及三道就业面试题标准答案已写入`docs/W3_LEARNING_HANDOFF.md`；
7. [ ] 用户完成W3学习确认，本文件的W3“⑤拥有权验证”改为“已完成”；
8. [ ] 用户在第7项完成后明确下令开始W4。

## W0 ⑤拥有权验证记录

- 完成日期：2026-08-31
- 用户能够区分 `split_seed` 控制数据划分，`run_seed` 控制初始化与训练顺序。
- 用户能够说明seed和矩阵必须在结果前提交，否则会形成结果先于规则的选择偏差。
- 用户能够区分随机指标需要mean ± sample std，而参数量、FLOPs等确定性量直接报告原值。
- 按用户最新要求，本阶段不包含亲手修改代码或生成个人 `git diff`。

## W1 ⑤拥有权验证记录

- 状态：已完成。
- 完成日期：2026-08-31。
- 用户选择由 Codex 直接给出三道源码题的面试版答案和代码依据，并在讲解后明确下令继续开发；按用户决定，本阶段不要求亲手修改代码或生成个人 `git diff`。
- 该记录只表示用户决定完成本阶段学习交付并进入 W2，不夸大为用户已经能够脱离提示独立回答全部问题；相关知识仍应加入后续复习。

## W2 ⑤拥有权验证记录

- 状态：已完成。
- 完成日期：2026-09-01。
- 按用户最新学习偏好，阶段结束时直接提供三道就业面试相关源码题及标准答案，不要求用户盲猜，也不要求亲手修改代码。
- 用户已明确确认学习完成并下令继续；相关标准答案保存在`docs/W2_LEARNING_HANDOFF.md`。

## W3 ⑤拥有权验证记录

- 状态：已完成。
- 完成日期：2026-09-01。
- 已提供三道就业面试相关源码题及标准答案，不要求用户盲猜，也不要求亲手修改代码。
- 用户已明确确认学习完成并下令继续；相关标准答案保存在`docs/W3_LEARNING_HANDOFF.md`。

## 环境与依赖变更

- W0 未安装、升级或删除任何依赖。
- W0 不修改业务代码和训练实现。
- W0 核对环境：Windows 11、Python 3.12.10、NumPy 2.5.2、PyTorch 2.12.1+cu130、CUDA 13.0、cuDNN 9.2、NVIDIA GeForce RTX 4060 Laptop GPU。
- W0 测试说明：`.venv\Scripts\python.exe -m pytest -q` 因环境未安装 `pytest` 而未进入测试收集；没有用其他环境冒充项目环境，也没有安装新依赖。
- W1 未安装、升级或删除依赖；未训练模型、未运行 frozen test 评测、未读取或记录模型指标。
- W1 测试命令：`.venv\Scripts\python.exe -m unittest discover -s tests -v`，结果为 96 项全部通过。
- W2 未安装、升级或删除依赖；使用已验证的RTX 4060 Laptop GPU完成10次正式训练。
- W2没有访问V1 test信号、标签或模型指标，全部新指标来自固定validation。
- W2测试命令：`.venv\Scripts\python.exe -m unittest discover -s tests -v`，结果为106项全部通过；其中3项会重新计算W2 summary并逐个加载10个checkpoint。
- W3未安装、升级或删除依赖；使用同一RTX 4060 Laptop GPU完成10次新增正式训练。
- W3没有访问V1 test信号、标签或模型指标；10次新评测和5次复用结果全部来自固定validation。
- W3测试命令：`.venv\Scripts\python.exe -m unittest discover -s tests -v`，结果为127项全部通过；其中4项会重算W3汇总、逐个加载10个新checkpoint并核对5个共享初始backbone。

## 给接手者的上下文

- 本阶段尝试过但放弃的方案，以及放弃原因：旧协议的 `17, 43, 97` seed、12次矩阵、temporal bins 和 GNU Radio 计划与 v6 不一致，不进入九月 W1–W4。
- V1 的 `BaselineExperimentConfig.seed` 仍保持历史行为；W2 必须使用新的 `V2ExperimentConfig` 和 manifest 入口，不能误用 V1 训练入口。
- V1 的 `RadioMLPartitions` 仍保留 test `Subset` 以维持历史兼容；V2 的 `DevelopmentPartitions` 刻意不提供 test 入口，两套 API 不得混用。
- W0 的耗时测量不是正式实验，不得保存、读取或引用其中的验证指标。
- W1 不得重新打开 `docs/archive/` 里的旧 seed 或扩展范围；需要变更协议时必须新提交并说明尚未产生哪些结果。
- W1 已确认仓库测试基于标准库 `unittest`，不需要为缺少 `pytest` 安装新依赖；此前关于“必须先解决 pytest”的判断已被实际测试入口纠正。
- W2已严格复用`manifests/v2/`固定索引与五个预注册run seed；W1阶段当时没有提前实现或启动多seed训练dispatcher。
- W2已完成A0/A1多seed对照；W3只能新增A2-G与A3，A2-T必须直接复用W2的A1结果，禁止重复训练A1或把两个一维消融扩成2×2。
- W2结果说明A1在当前五个seed和固定validation下方向一致地高于A0，但不能据此宣称统计显著、真实空口泛化或所有信道条件下普遍更优。
- W3两条消融均未形成足够稳定的新优势：A2-G只取得小于波动的平均差异，A3均值更低且波动更大；默认仍保留A1/A2-T，不为追求“新模型”强行promote变体。
- W4只能整理run manifest、汇总报告、README和复现验收，不得借整理阶段重新训练、补挑seed或重开V1 test。

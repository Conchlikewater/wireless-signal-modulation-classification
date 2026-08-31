# Wireless 开发状态

## 当前阶段

- 当前阶段：W1 · `split_seed` / `run_seed` 解耦（开发与代码验收已完成，等待学习验收）
- 真实开工日期：2026-08-31
- ⑤ 拥有权验证：未完成
- 已完成阶段：W0

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

- W1 开发与自动化验收没有阻塞项。
- 独立 `.venv` 没有第三方 `pytest`，但仓库测试实际使用 Python 标准库 `unittest`；无需安装依赖即可执行。W1 完整回归共 96 项。
- `compileall` 额外检查因现有 `__pycache__` 写权限被系统拒绝，未作为通过项；96 项测试已经实际导入并执行全部新增模块。

## 下一阶段入口条件

只有同时满足以下条件，才能由用户明确下令开始 W2：

1. [x] W0 seed 列表和实验矩阵已在结果前提交；
2. [x] `split_seed` 与 `run_seed` 已在 V2 代码入口解耦；
3. [x] 固定 split manifest 已保存、哈希已锁定并通过互斥检查；
4. [x] V2 入口不返回 test Dataset、Subset、索引数组或 DataLoader；
5. [x] 96 项离线测试通过；
6. [ ] W1 学习讲解和三道具体源码题已完成；
7. [ ] 本文件的“⑤ 拥有权验证”已改为“已完成”；
8. [ ] 用户明确发出 W2 开工指令。

## W0 ⑤拥有权验证记录

- 完成日期：2026-08-31
- 用户能够区分 `split_seed` 控制数据划分，`run_seed` 控制初始化与训练顺序。
- 用户能够说明seed和矩阵必须在结果前提交，否则会形成结果先于规则的选择偏差。
- 用户能够区分随机指标需要mean ± sample std，而参数量、FLOPs等确定性量直接报告原值。
- 按用户最新要求，本阶段不包含亲手修改代码或生成个人 `git diff`。

## W1 ⑤拥有权验证记录

- 状态：未完成。
- 按用户最新要求，先提供短知识点，再一次给出三道与 W1 真实源码直接相关的问题；不要求亲手修改代码或生成个人 `git diff`。
- 完成问答和纠错前，不得把本节改为“已完成”，也不得开始 W2。

## 环境与依赖变更

- W0 未安装、升级或删除任何依赖。
- W0 不修改业务代码和训练实现。
- W0 核对环境：Windows 11、Python 3.12.10、NumPy 2.5.2、PyTorch 2.12.1+cu130、CUDA 13.0、cuDNN 9.2、NVIDIA GeForce RTX 4060 Laptop GPU。
- W0 测试说明：`.venv\Scripts\python.exe -m pytest -q` 因环境未安装 `pytest` 而未进入测试收集；没有用其他环境冒充项目环境，也没有安装新依赖。
- W1 未安装、升级或删除依赖；未训练模型、未运行 frozen test 评测、未读取或记录模型指标。
- W1 测试命令：`.venv\Scripts\python.exe -m unittest discover -s tests -v`，结果为 96 项全部通过。

## 给接手者的上下文

- 本阶段尝试过但放弃的方案，以及放弃原因：旧协议的 `17, 43, 97` seed、12次矩阵、temporal bins 和 GNU Radio 计划与 v6 不一致，不进入九月 W1–W4。
- V1 的 `BaselineExperimentConfig.seed` 仍保持历史行为；W2 必须使用新的 `V2ExperimentConfig` 和 manifest 入口，不能误用 V1 训练入口。
- V1 的 `RadioMLPartitions` 仍保留 test `Subset` 以维持历史兼容；V2 的 `DevelopmentPartitions` 刻意不提供 test 入口，两套 API 不得混用。
- W0 的耗时测量不是正式实验，不得保存、读取或引用其中的验证指标。
- W1 不得重新打开 `docs/archive/` 里的旧 seed 或扩展范围；需要变更协议时必须新提交并说明尚未产生哪些结果。
- W1 已确认仓库测试基于标准库 `unittest`，不需要为缺少 `pytest` 安装新依赖；此前关于“必须先解决 pytest”的判断已被实际测试入口纠正。
- W2 只能复用 `manifests/v2/` 的固定索引与五个预注册 run seed；W1 没有实现或启动多 seed 训练 dispatcher。

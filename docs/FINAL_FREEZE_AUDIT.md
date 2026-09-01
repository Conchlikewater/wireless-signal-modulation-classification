# Wireless V2 Core最终冻结审计

> 审计日期：2026-09-01  
> 审计范围：唯一正式仓库的W0–W4代码、协议、Git历史、数据身份、固定划分、训练产物、统计汇总、manifest复现、测试、CI和文档  
> 当前结论（2026-09-02）：原“通过、可以冻结”结论撤销；开发阶段完成，发布纠错进行中。

## 1. 审计边界

本次没有重新训练、没有创建V1 test DataLoader、没有读取V1 test信号或标签、没有执行V1 test推理，也没有安装依赖。对640,919,653字节的RadioML Pickle只计算SHA-256；代表性复现验收只执行默认dry-run。

实际证据优先级为：代码与已提交产物、自动化测试、Git历史、冻结协议、说明文档。归档在`docs/archive/`的旧计划不是当前执行依据。

## 2. 总体结论

Wireless V2 Core的训练与实验开发阶段已经完成：数据边界固定，训练随机性与划分随机性解耦，20次预注册validation运行全部保留。发布审计随后发现A3历史JSON和run manifest把`config.dropout`误记为`0.3`，实际模型工厂使用`0.0`。因此本文件原先的“最终冻结通过”判断不再有效，必须完成勘误、语义测试、训练级复现和纠正版发布审计。

它适合用于通信AI、智能信号处理、PyTorch/初级ML工程实习面试，但不是研究型算法创新、真实空口系统或生产部署项目。当前不需要继续堆模型；继续扩张P0的边际价值低于转入第二个项目R0。

## 3. 证据核对表

| 项目 | 审计结果 | 证据 | 结论 |
|---|---|---|---|
| 数据身份 | 通过 | Pickle SHA-256为`b29ccc25...09e8c`，大小640,919,653字节 | 与冻结协议一致 |
| 固定划分 | 通过 | manifest SHA-256为`48ad195d...97a7`；索引归档SHA-256为`aae52254...24bb` | 154,000/33,000/33,000，220个modulation×SNR分层单元 |
| Test隔离 | 通过 | V2入口只返回train/validation；测试验证无test索引、Subset或DataLoader | V1 test仅保留数量和索引哈希元数据 |
| 预注册顺序 | 通过 | 协议提交`3cfc1c4`早于W1–W4实现与全部结果提交 | seed、矩阵和口径不是看结果后决定 |
| W2主对照 | 通过 | 10份JSON、10个checkpoint、`w2_summary.json` | A0/A1各5 seed，零失败、零替换 |
| W3消融 | 指标产物通过、A3元数据有勘误 | 10份新JSON、10个checkpoint、5个初始backbone、`w3_summary.json`、`A3/ERRATA.json` | A2-T复用A1；A2-G/A3各5 seed；A3实际`p=0`但记录误写`0.3` |
| 统计口径 | 通过 | 汇总测试从原始JSON重算mean、sample std、paired delta | 没有只报最好seed；`n=5`只作描述性统计 |
| 复现清单 | 纠错中 | 20份历史run manifest保持原字节，catalog登记A3勘误 | 覆盖4配置×5 seed；读取A3时必须应用勘误 |
| Checkpoint完整性 | 通过 | 20个正式checkpoint均逐一哈希并可加载 | V2指标有原始产物支持 |
| 共享初始化 | 通过 | 5个backbone文件与state hash；A3/A1初始state hash相同 | A2/A3公平性证据存在 |
| 启动前审计 | 通过 | A2-G/20260901 dry-run核对8项仓库文件和数据SHA-256 | 不加载信号、不训练、不评测test |
| 自动化测试 | 当前通过 | 本地140项`unittest`全部通过 | 新增effective配置语义与67个历史产物字节保护 |
| CI | 配置通过、远端待同步 | GitHub Actions使用Python 3.12运行同套测试并解析源码 | 当前V2尚未推送，远端CI还未验证最终HEAD |
| Git对象 | 通过 | `git fsck --full`没有可达对象损坏 | 少量dangling对象不属于当前历史损坏 |

## 4. 结果可信度

V2的下列结论可以安全写进项目说明和简历，但必须带上“固定validation、5个预注册seed”的限定：

- A0 Macro F1：`40.24% ± 2.52`个百分点；
- A1/A2-T Macro F1：`55.47% ± 1.05`个百分点；
- `A1−A0` Macro F1 paired delta：`+15.23 ± 3.35`个百分点，`n_pairs=5`，方向全部为正；
- A2-G与A2-T差异小于波动且方向不一致，不支持替换当前A1；
- A3去掉Dropout后均值下降、波动增大，但方向不完全一致，只能说当前证据不支持移除Dropout；
- 低SNR接近机会水平，0 dB以上约进入77%–82%的平台，并存在QAM16/QAM64、WBFM/AM-DSB等混淆。

不能写成：V2 final test、统计显著提升、SOTA、真实空口鲁棒性、跨设备泛化或生产性能。

V1的56.38% Accuracy与56.26% Macro F1仍可表述为“一次性历史test记录”，但原V1 checkpoint和JSON不在当前仓库产物中，因此不能声称现在能从仓库独立重算。V2的20次validation结果不存在这个缺口。

## 5. 最终审计修复

审计发现W4 dry-run原先只核对结果JSON、checkpoint、split manifest、协议和依赖规格；真正执行A2-G还依赖共享初始backbone，真正建立DataLoader还依赖`split_indices.npz`。如果这两个文件缺失或被替换，旧dry-run可能先显示可执行、随后才在正式启动时报错。

修复后：

1. A0/A1必须没有W3初始化字段；
2. A2-G必须提供共享backbone文件、文件SHA-256和state SHA-256；
3. A3必须声明与配对A1初始state一致，两个SHA-256必须相同；
4. manifest初始化记录必须与原始结果JSON一致；
5. dry-run会核对固定split索引归档；
6. A2-G dry-run会额外核对共享初始backbone。

这是复现安全检查，不改变输入样本、batch顺序、模型计算、训练动态、checkpoint选择或指标计算。历史JSON、checkpoint和20份manifest均未覆盖，因此无需重训。

## 6. 剩余风险与限制

### 仍阻塞纠正版发布

- A3元数据勘误已经登记且有语义测试，但训练级`--execute`复现、validation同集选择偏差声明和纠正版最终发布审计尚未完成。
- 旧`wireless-v2-core` tag创建于缺陷发现之前，不得推送或用作纠正版发布标记。

### 不单独阻塞发布

- 顶层NumPy和PyTorch版本已固定，但没有完整锁定驱动与全部传递依赖；manifest记录了历史运行环境，仍不承诺跨机器bitwise一致。
- CI工作流存在，本地等价测试已通过；因V2提交尚未推送，远端CI和远端备份仍停在V1。
- 训练和延迟只在一台RTX 4060 Laptop GPU环境验证，不能外推所有硬件。
- RadioML 2016.10A是公开合成数据，固定validation不是独立外部证据。

### 未来P1才处理

1. `temporal occlusion sensitivity`：低成本解释模型依赖哪些时间区域；只作解释，不用来宣称因果。
2. `Channel Robustness Lab`：按结果前冻结的样本与扰动规则演示相位、频偏、时移、多径敏感性；不是独立test。
3. 独立外部数据：另写生成协议，冻结后只评一次；这是补足域外证据最有价值的方向。

Transformer、强化学习、大型网络、Web仪表盘、实验数据库和刷榜式调参均不建议现在开发。

## 7. 冻结与后续决策

- Wireless训练开发阶段已完成，但发布纠错尚有必须项；默认模型暂时保持A1/A2-T。
- 此后不修改正式W2/W3 JSON、checkpoint、汇总或manifest，不继续查看V1 test。
- 如未来开展P1，必须创建新协议版本、新目录和新提交链，不覆盖V2 Core。
- 本地V1基线应标记在纯V1提交`f9668db`；不移动、也不推送缺陷发现前创建的`wireless-v2-core` tag，纠正版通过后使用新tag。
- 在进入R0前，只剩版本标记和可选的`origin/main`远端同步；远端推送属于发布/备份动作，不影响本地审计结论。

因此，当前先完成已经批准的发布纠错、错误分析、A2-L预注册实验和交付打包；在新的发布审计通过前，不再声称Wireless V2 Core已经最终冻结。

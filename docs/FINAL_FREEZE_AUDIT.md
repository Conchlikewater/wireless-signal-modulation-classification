# Wireless V2 纠正版发布审计

> 审计日期：2026-09-02
>
> 审计范围：W-A发布纠错、W-B错误分析、W-C A2-L扩展、W-D交付打包，以及既有W0–W4全部证据
>
> 当前结论：**本地纠正版发布审计通过；首次远端CI发现跨平台校验缺陷，修复后的最终HEAD仍需远端复验。**

## 1. 审计边界

本轮没有重新划分数据，没有创建V1 test DataLoader，没有读取V1 test信号或标签，没有执行V1 test推理，也没有把V1历史test指标用于模型选择。除计划内5次A2-L正式train/validation和一次预先设阈值的A3 replay外，没有重训A0、A1、A2-T、A2-G或A3。

证据优先级为：已提交代码与机器可读产物、自动化测试、Git历史、冻结协议、说明文档。`docs/archive/`只保留旧计划背景，不是当前执行依据。

## 2. 发布门结论

| 发布门 | 状态 | 可验证证据 |
|---|---|---|
| A3 effective配置与实际模型一致 | 通过 | `A3/ERRATA.json`；语义测试实例化模型并检查`nn.Dropout.p` |
| 原始20组W2/W3历史产物不变 | 通过 | 67个受保护文件全部与`c05d7ef`基线SHA-256一致 |
| 语义测试能抓住人为错误 | 通过 | 把A3 effective值临时改回0.3时，5个seed均出现`[0.0] != [0.3]`，恢复后全绿 |
| 一次训练级复现 | 通过 | A3/20260901的best epoch、loss、Accuracy、Macro F1和checkpoint SHA全部与历史一致 |
| validation乐观偏差披露 | 通过 | README与技术报告均在主结果附近明确说明 |
| 错误分析 | 通过 | 四arm×五seed固定validation推理；SNR曲线、高/低SNR混淆、class×SNR热力图与书面发现齐全 |
| A2-L预注册顺序 | 通过 | 假设提交`bb7c6ec`是模型提交`5372c20`的祖先 |
| A2-L容量与产物 | 通过 | 223,932参数；与A2-T差0.292%；5个JSON/checkpoint均可加载并通过哈希校验 |
| A2-L结论 | 通过 | 联合假设按预注册规则判“不支持”；首次错误汇总保留并登记勘误 |
| run manifest | 通过 | catalog含5配置×5 seed共25份；W5代表manifest核对9项依赖并成功生成dry-run命令 |
| 最小推理demo | 通过 | 构造checkpoint与仓库真实A1 checkpoint均完成单条`(2,128)`输入预测 |
| 完整测试 | 通过 | 加入发布可移植性回归检查后，本地169项`unittest`全部通过 |
| Python源码解析 | 通过 | `src/`、`scripts/`、`tests/`共79个Python文件均通过AST解析 |
| README链接 | 通过 | README中的本地文件/图表链接全部存在 |
| Git对象 | 通过 | `git fsck --full`无可达对象损坏；dangling对象不属于当前提交历史损坏 |
| 远端发布与CI | 纠错后待复验 | 远端CI在Linux暴露浅克隆缺少基线commit，以及历史JSON与manifest依赖文本换行表示不同；模型和实验产物测试未显示数值缺陷 |

## 3. 数据与test边界

- 数据SHA-256：`b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c`；
- 固定split manifest SHA-256：`48ad195d5552e3ec4e5a6d1bc4fc0f20099df8dc70f8eb78a80df95e7f5297a7`；
- 样本：train 154,000 / validation 33,000 / V1 test metadata 33,000；
- 220个modulation×SNR分层单元，每单元`700/150/150`；
- V2入口只返回train/validation；test只允许样本数、索引hash和互斥检查；
- V1历史56.38% Accuracy与56.26% Macro F1仍只是一次性历史验收，不能作为V2结果或当前可独立重算的指标。

本轮不执行A-5的可选test再次启封，因为用户没有给出单独的人类决策，且当前实验目标不需要用test量化validation乐观偏差。

## 4. 产物完整性

当前正式V2产物：

- 25份`validation_result.json`；
- 25个best checkpoint；
- 5个共享初始backbone；
- W2、W3历史汇总各1份；
- W5原汇总、勘误和纠正版汇总；
- 25份单次run manifest和1份catalog；
- 0个`failure.json`。

受保护历史范围仍是纠错前20份JSON、20个checkpoint、5个初始backbone、2份W2/W3 summary和20份W2/W3 run manifest，共67个文件。保护脚本逐一从基线Git对象重建字节并核对SHA-256，结果全部匹配。

## 5. 两项勘误

### 5.1 A3配置元数据

5份A3历史JSON与run manifest的`config.dropout`记录为0.3，实际模型工厂使用0.0。原始文件不覆盖；`experiments/v2/w3/A3/ERRATA.json`声明effective值。影响仅限记录语义，不改变训练模型、checkpoint或指标。

### 5.2 A2-L汇总判定

首次`w5_summary.json`把联合假设误判为`partially_supported`。预注册规则要求：两组高SNR条件都通过、但任一低SNR边界失败时应判`not_supported`。原汇总SHA-256固定为`9beca76b...d5c61a`并保留；`SUMMARY_ERRATA.json`和`w5_summary_corrected.json`给出正确结论。只重跑确定性汇总，没有重训模型。

这两个案例都展示同一工程原则：不能为了让仓库“看起来干净”改写历史证据；应保留原文件、增加机器可读勘误，并让测试保护正确解释。

## 6. 结果与安全表述

可以写进简历，但必须带“RadioML 2016.10A固定validation、五个预注册seed”的限定：

- A1相对A0的Macro F1 paired delta：`+15.23 ± 3.35 pp`，5/5同向；
- A2-T/A2-G容量受控比较是零结果：`A2-T−A2-G=-0.25 ± 1.38 pp`，方向不一致；
- 去Dropout没有可靠收益：`A2-T−A3=+1.08 ± 3.54 pp`，方向不完全一致；
- A2-L总体Macro F1为`58.28% ± 2.61 pp`，但联合假设不支持：高SNR改善伴随低SNR相对A2-G超边界退化；
- 错误分析观察到低SNR预测塌缩，以及高SNRQAM16/QAM64、WBFM/AM-DSB残余混淆。

不能写成：V2 final test、统计显著提升、SOTA、真实空口鲁棒性、跨设备泛化或生产性能。每次checkpoint与报告指标使用同一validation，存在模型选择造成的乐观偏差。

## 7. 工程与复现边界

manifest冻结并校验数据、split、seed、配置/effective语义、环境、Git commit、原始JSON、checkpoint、协议、依赖规格和共享初始化。dry-run不反序列化信号、不训练；只有显式`--execute`才启动单次train/validation replay，并拒绝覆盖已有输出。

这些机制支持“身份可审计、运行可重启”，不保证任意硬件和驱动上的逐位一致。顶层依赖版本已固定，但没有Docker、驱动镜像或完整传递依赖锁；这是计划内能力边界。

## 8. 剩余风险

### 发布前必须完成

1. 提交跨平台发布校验修复，且不修改任何冻结实验产物；
2. 创建递增的纠正版tag；缺陷发现前的旧本地`wireless-v2-core`及首次CI失败的`wireless-v2-core-corrected`都不得作为最终通过版本；
3. push `main`和新tag；
4. 等最终HEAD GitHub Actions通过；
5. 从公开页面确认README、技术报告和CI可见。

### 不阻塞本次发布

- RadioML是公开合成数据，没有独立外部集或真实空口证据；
- A2-L只有固定validation结果，预注册联合假设不支持；
- V1历史test缺少当前仓库中的原始checkpoint/JSON，不能独立重算；
- 延迟只在一台RTX 4060 Laptop GPU测量；
- 最小推理demo不是数据采集、同步、切窗、拒识或生产服务。

## 9. 后续范围

Wireless当前P0不再扩张。若未来开启P1，优先级为：独立数据协议与域外集，其次是低成本temporal occlusion解释和受控Channel Robustness Lab。Transformer、强化学习、Web服务、实验数据库、Docker和刷榜式调参不进入当前发布。

本地模型、实验产物和文档已经达到纠正版发布条件。远端CI失败属于校验链的跨平台缺陷：工作流默认浅克隆无法读取`c05d7ef`，且历史JSON、`pyproject.toml`与`requirements-gpu.txt`的SHA-256记录基于CRLF工作区字节，而Linux默认检出LF。修复采用完整历史检出及对冻结哈希相关文本显式统一检出换行，不重写任何冻结实验产物；最终远端CI结果另行报告。

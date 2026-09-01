# Wireless Signal Modulation Classification：V2 Core 技术报告

## 1. 结论摘要

V2 Core在不重新使用V1 test的前提下，完成了固定开发边界、五个预注册run seed、A0/A1成对模型对照、两条容量受控消融、完整原始产物和单次运行manifest复现入口。

> 发布状态（2026-09-02）：开发阶段完成，发布纠错进行中。原最终冻结审计漏掉A3的元数据语义不一致，原“无P0阻塞项”结论已经撤销；纠正版发布审计通过前，本报告不能作为最终发布状态证明。

当前默认模型仍是A1/A2-T `TemporalCNN1D`。主要理由：

- 相比A0，A1在五个seed下的validation Macro F1平均提高`15.23`个百分点，五个paired delta方向全部为正；
- A2-G的平均Macro F1只比A2-T高约`0.25`个百分点，差异小于配对波动且方向不一致，不足以支持稳定结构优势；
- A3去掉Dropout后平均Macro F1降低约`1.08`个百分点、波动增大，不支持移除当前`p=0.3`；
- V2 Core没有新的独立test，因此以上都是固定validation上的开发阶段结论，不是新的最终泛化成绩。

A3限定：W3工厂在实现提交`596900a`中实际构造`TemporalCNN1D(dropout=0.0)`，但通用`V2ExperimentConfig`的默认`dropout=0.3`被序列化进5份A3 JSON及对应run manifest。原始产物不覆盖；`experiments/v2/w3/A3/ERRATA.json`提供机器可读的effective值。该缺陷影响配置记录准确性，不改变实际训练模型或已报告指标。

## 2. 数据与评测边界

- 数据：RadioML 2016.10A，220,000条样本，11种调制方式，20个SNR条件；
- 单条输入：`(2,128)`，批次输入：`(B,2,128)`；
- 数据SHA-256：`b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c`；
- 固定划分：train 154,000 / validation 33,000 / V1 test metadata 33,000；
- 分层方式：modulation×SNR，共220个分层单元，每单元`700/150/150`；
- `split_seed=20260812`，固定manifest SHA-256：`48ad195d5552e3ec4e5a6d1bc4fc0f20099df8dc70f8eb78a80df95e7f5297a7`；
- `run_seed`：`20260901`、`20260902`、`20260903`、`20260904`、`20260905`；
- V2入口只返回train/validation DataLoader；V1 test只允许数量和索引哈希互斥检查；
- V2没有创建test DataLoader、没有读取V1 test信号/标签、没有重新推理，也没有把历史test指标纳入V2汇总。

V1 test已在历史阶段启封。V1的56.38% Accuracy和56.26% Macro F1只能保留为“一次性历史验收记录”，不能用于V2模型选择，也不能重新包装为V2 test结果。

## 3. 冻结训练口径

所有正式运行统一使用：

- train batch size：256；
- validation batch size：512；
- optimizer：Adam，`lr=0.001`、`betas=(0.9,0.999)`、`eps=1e-8`、`weight_decay=0`；
- scheduler：无；
- epoch上限：20；
- early stopping：`patience=4`、`min_delta=0.0001`；
- best checkpoint：最低validation loss；
- AMP：关闭；
- `num_workers=0`；
- 确定性算法：启用，`cudnn.benchmark=false`；
- 每个模型使用相同split manifest、同一组run seed和同seed训练batch顺序；
- 五个seed只做描述性mean±sample std，不进行显著性检验。

历史正式环境为Windows 11、Python 3.12.10、NumPy 2.5.2、PyTorch 2.12.1+cu130、CUDA 13.0、cuDNN 9.2、NVIDIA GeForce RTX 4060 Laptop GPU。

## 4. 实验矩阵与全部原始结果

下表全部为固定validation结果。A2-T与A1是同一配置，W3直接复用W2的A1五次运行，没有重复训练。

| 配置 | run seed | Accuracy | Macro F1 | best epoch | 记录耗时/s |
|---|---:|---:|---:|---:|---:|
| A0 | 20260901 | 41.04% | 38.81% | 20 | 63.34 |
| A0 | 20260902 | 42.93% | 41.42% | 20 | 62.49 |
| A0 | 20260903 | 42.17% | 39.64% | 20 | 67.86 |
| A0 | 20260904 | 45.33% | 43.93% | 20 | 66.89 |
| A0 | 20260905 | 39.82% | 37.41% | 20 | 62.42 |
| A1/A2-T | 20260901 | 55.47% | 55.96% | 13 | 77.23 |
| A1/A2-T | 20260902 | 55.12% | 56.36% | 6 | 47.37 |
| A1/A2-T | 20260903 | 54.18% | 55.50% | 10 | 66.54 |
| A1/A2-T | 20260904 | 54.08% | 53.67% | 9 | 59.94 |
| A1/A2-T | 20260905 | 55.22% | 55.89% | 7 | 49.82 |
| A2-G | 20260901 | 55.87% | 55.84% | 7 | 45.89 |
| A2-G | 20260902 | 54.41% | 55.91% | 3 | 29.23 |
| A2-G | 20260903 | 54.42% | 54.63% | 4 | 33.50 |
| A2-G | 20260904 | 54.94% | 56.32% | 9 | 54.42 |
| A2-G | 20260905 | 55.35% | 55.95% | 11 | 63.35 |
| A3 | 20260901 | 55.17% | 56.42% | 6 | 40.83 |
| A3 | 20260902 | 49.98% | 49.01% | 1 | 20.81 |
| A3 | 20260903 | 54.88% | 55.13% | 6 | 41.09 |
| A3 | 20260904 | 53.83% | 54.70% | 9 | 53.56 |
| A3 | 20260905 | 55.67% | 56.72% | 7 | 45.72 |

20条结果均保留，没有挑选最高seed。基础设施失败、NaN、重跑和seed替换均为0。

## 5. W2：A0与A1成对对照

| 配置 | Accuracy mean±sample std | Macro F1 mean±sample std | 参数量 |
|---|---:|---:|---:|
| A0 SimpleCNN | 42.26% ± 2.08 pp | 40.24% ± 2.52 pp | 11,499 |
| A1 TemporalCNN | 54.81% ± 0.64 pp | 55.47% ± 1.05 pp | 224,587 |

同seed `A1−A0`：

- Accuracy paired delta：`+12.55 ± 2.57`个百分点；
- Macro F1 paired delta：`+15.23 ± 3.35`个百分点；
- `n_pairs=5`，五个seed方向全部为正。

安全结论：在当前RadioML固定validation、冻结训练口径和五个seed下，保留粗粒度时序位置的A1稳定高于简单全局平均基线A0。不能由此声称统计显著、真实空口泛化或对所有无线数据都有效。

## 6. W3：容量受控消融

### 6.1 聚合方式

共享backbone输出为`(B,128,32)`：

- A2-T：8个时序分箱，`Flatten(1024) -> Linear(1024,128) -> 11`；
- A2-G：全局池化，`Flatten(128) -> Linear(128,947) -> 11`。

| 配置 | Accuracy mean±sample std | Macro F1 mean±sample std | 全模型参数 | head参数 | MACs |
|---|---:|---:|---:|---:|---:|
| A2-T | 54.81% ± 0.64 pp | 55.47% ± 1.05 pp | 224,587 | 132,619 | 4,441,472 |
| A2-G | 55.00% ± 0.63 pp | 55.73% ± 0.64 pp | 224,559 | 132,591 | 4,440,625 |

`A2-T−A2-G` Macro F1 paired delta为`-0.25 ± 1.38`个百分点，方向不一致。两者参数只差28个（`0.0125%`），但函数结构不同，因此只能称“共享backbone与近似参数预算下的容量受控比较”。当前不promote A2-G。

### 6.2 Dropout

| 配置 | Accuracy mean±sample std | Macro F1 mean±sample std | 参数量 |
|---|---:|---:|---:|
| A2-T，Dropout 0.3 | 54.81% ± 0.64 pp | 55.47% ± 1.05 pp | 224,587 |
| A3，Dropout 0 | 53.91% ± 2.30 pp | 54.40% ± 3.13 pp | 224,587 |

`A2-T−A3` Macro F1 paired delta为`+1.08 ± 3.54`个百分点。去除Dropout后均值下降且波动增大，但seed方向不完全一致。当前不promote A3，也不把结果外推为Dropout普遍有效。

## 7. SNR与错误分析

A2-T在代表性SNR上的五seed均值：

| SNR | Accuracy | Macro F1 |
|---:|---:|---:|
| -20 dB | 9.28% | 3.50% |
| -14 dB | 10.42% | 5.42% |
| -10 dB | 23.07% | 20.01% |
| -6 dB | 49.76% | 46.58% |
| 0 dB | 77.31% | 75.82% |
| 6 dB | 81.30% | 79.60% |
| 12 dB | 81.12% | 79.40% |
| 18 dB | 80.50% | 78.78% |

解释边界：

- `-20`至`-14 dB`接近11分类机会水平，主要是低SNR下信号结构被噪声淹没；
- 从`-10 dB`到`0 dB`性能快速提升，说明模型利用了随信号质量恢复而出现的结构特征；
- `0 dB`以上进入约77%–82%的平台，仍存在类别表征限制，而不是只剩噪声问题；
- 汇总五个seed的整体混淆矩阵后，主要bad case包括QAM16/QAM64、WBFM/AM-DSB，以及低SNR下PSK类别误入AM-SSB；
- 这些是当前RadioML生成条件下的观察，不代表真实空口或所有调制系统。

## 8. 复现与审计

`experiments/v2/run_manifests/catalog.json`登记20份manifest，对应4个配置×5个seed。每份manifest记录：

- 数据路径和SHA-256；
- split manifest、split seed和哈希；
- run seed、模型、完整训练配置和优化器；
- 历史Python/NumPy/PyTorch/CUDA/cuDNN/GPU环境；
- 历史实现Git commit；
- 原始结果JSON与checkpoint路径、哈希、best epoch；
- 协议与依赖规格文件哈希；
- 单次运行dry-run和显式执行入口。

代表性非训练审计命令：

```powershell
.\.venv\Scripts\python.exe scripts\replay_v2_run.py `
  experiments\v2\run_manifests\w3-a2-g-20260901.json `
  --data-file data\raw\RML2016.10a\RML2016.10a_dict.pkl `
  --output-directory artifacts\replay\w3-a2-g-20260901
```

默认只校验数据和仓库文件哈希并打印重启命令，不训练。只有显式添加`--execute`才会重新训练该manifest对应的单次运行；输出目录必须不存在。

发布纠错阶段在执行前提交`docs/REPLAY_REPRODUCTION_PROTOCOL.md`，冻结Accuracy和Macro F1绝对差值均不超过`1.0`个百分点的规则，随后只执行一次A3/20260901 replay。结果的best epoch、Accuracy、Macro F1、best validation loss和checkpoint SHA-256均与历史完全一致，`test_set_used=false`。独立记录位于`artifacts/replay/w3-a3-20260901/REPRODUCTION.md`。

复现不等于保证跨机器逐位一致。manifest记录并检查环境，PyTorch确定性设置也已启用，但驱动、硬件、底层算子或未来软件版本仍可能造成数值差异。因此验收目标是“运行身份和过程可审计、条件可重建”，不是承诺任意机器上的bitwise identical。

## 9. 证据索引

- 协议：`docs/v2_protocol.md`；
- 固定划分：`manifests/v2/split_manifest.json`；
- W2汇总：`experiments/v2/w2/w2_summary.json`，SHA-256 `7c5f2513ae78903abad0781801e4f5577ec009a7203cd7a56ac5503bbaa1c00f`；
- W3汇总：`experiments/v2/w3/w3_summary.json`，SHA-256 `b0edff36b715f68599ceb8076e371a80994ac0af4ae02e4d7af67d357eaeccc8`；
- manifest catalog：`experiments/v2/run_manifests/catalog.json`，已登记A3勘误；
- 原始JSON和best checkpoint：`experiments/v2/w2/`、`experiments/v2/w3/`；
- 自动化测试：当前140项离线`unittest`全部通过；新增语义测试直接检查各arm的`nn.Dropout.p`，并验证67个受保护历史产物仍与纠错前基线字节一致；
- W4本身没有重新训练；发布纠错阶段完成一次A3训练级replay。两阶段均未访问V1 test信号、标签或推理结果。

原冻结审计的撤销原因、Git边界、远端同步状态和后续范围见`docs/FINAL_FREEZE_AUDIT.md`。

## 10. 作品集结论与限制

V2 Core证明了固定数据边界、PyTorch训练、early stopping、checkpoint、多seed成对实验、sample std、容量受控消融、SNR分析、失败记录和manifest复现等能力。它适合作为通信AI/智能信号处理/初级ML工程岗位的完整教学型作品集。

它不能证明：

- 真实空口鲁棒性；
- 跨设备、跨信道或跨数据源泛化；
- 生产服务能力；
- 统计显著的算法创新；
- SOTA或研究型新算法贡献。

若未来继续P1，应先引入真正独立来源或预先冻结的独立合成域外数据，再考虑复杂模型。不能继续反复查看V1 test来指导开发。

# A2-L 容量受控 LSTM 聚合实验报告

## 结论

A2-L在五个预注册seed上的固定validation Macro F1为`58.28% ± 2.61 pp`，相对A2-T和A2-G的总体paired delta分别为`+2.80 ± 2.40 pp`和`+2.55 ± 2.44 pp`。但预注册的**联合假设判为不支持**：高SNR改善条件满足，低SNR相对A2-G的平均Macro F1下降`1.47 pp`，超出事先冻结的`[-1.0,+0.5] pp`范围。

这不是“LSTM无效”，也不能包装成“LSTM已证明更优”。安全结论是：当前固定validation上出现了较高但波动也更大的总体均值，以及以低SNR退化为代价的高SNR收益；由于没有独立test、五个seed只作描述性统计且计算成本更高，A2-L保留为实验性候选，不替换简洁的A1/A2-T默认实现。

## 实验身份与边界

- 假设在模型代码之前提交：`experiments/v2/w5/HYPOTHESIS.md`，提交`bb7c6ec`；
- 模型实现提交：`5372c20`；正式训练入口提交：`62820b5`；
- run seeds：`20260901`至`20260905`，无失败、无重跑、无seed替换；
- 数据：V2固定train 154,000 / validation 33,000；
- V1 test：未创建DataLoader、未读取信号或标签、未推理；
- checkpoint仍按最低validation loss选择；
- 五个seed仅报告mean±sample std和paired delta，不做显著性检验。

## 结构与容量

```text
shared trainable backbone -> (B, 128, 32)
-> transpose (B, 32, 128)
-> LSTM(input_size=128, hidden_size=127)
-> final time step (B, 127)
-> Dropout(0.3)
-> Linear(127, 11)
```

| 项目 | A2-L | A2-T | 差异/说明 |
|---|---:|---:|---:|
| 全模型参数 | 223,932 | 224,587 | -655（-0.292%） |
| 聚合/head参数 | 131,964 | 132,619 | -655 |
| 估算MACs/样本 | 8,455,669 | 4,441,472 | A2-L约1.90倍 |
| 估算FLOPs/样本 | 16,911,338 | 8,882,944 | 按2 FLOPs/MAC |
| 固定512样本批次延迟 | 4.907 ms | 1.487 ms | 同一RTX 4060环境的单次测量 |

LSTM的MACs包含输入与循环矩阵乘加，不包含门函数、cell逐元素更新、BatchNorm、激活、池化、Dropout和bias加法。因此它是明确口径下的估算，不是完整硬件指令计数。参数量接近也不代表表达能力相同，本实验仍只能称“共享初始backbone与近似参数预算下的容量受控比较”。

## 五个原始结果

| run seed | Accuracy | Macro F1 | best epoch | 记录耗时/s |
|---:|---:|---:|---:|---:|
| 20260901 | 59.63% | 61.88% | 8 | 90.43 |
| 20260902 | 56.53% | 57.12% | 6 | 77.28 |
| 20260903 | 54.57% | 55.63% | 2 | 44.85 |
| 20260904 | 55.11% | 56.65% | 4 | 60.32 |
| 20260905 | 57.96% | 60.09% | 9 | 98.32 |
| mean ± sample std | 56.76% ± 2.08 pp | 58.28% ± 2.61 pp | 5.8 ± 2.77 | 总计371.19 s |

## 预注册判定

| 比较方向 | 总体Macro F1 delta | 高SNR（≥10 dB）delta | 高SNR正向seed | 低SNR（≤-10 dB）delta | 低SNR规则 |
|---|---:|---:|---:|---:|---|
| A2-L − A2-T | +2.80 ± 2.40 pp | +3.13 ± 3.29 pp | 4/5 | -0.57 ± 2.54 pp | 通过 |
| A2-L − A2-G | +2.55 ± 2.44 pp | +2.79 ± 3.35 pp | 4/5 | -1.47 ± 3.26 pp | **失败** |

预注册的“支持”要求两组高SNR规则和两组低SNR规则全部通过；“部分支持”只允许高SNR仅一组通过，或高SNR均值位于`[+0.5,+1.0) pp`且4/5同向。本次两组高SNR均超过`+1.0 pp`且4/5同向，但一组低SNR规则失败，不属于预注册的部分支持情形，所以最终判为**不支持**。

## 汇总器缺陷与保留方式

首次汇总错误地把上述结果标成`partially_supported`。原文件`experiments/v2/w5/w5_summary.json`没有被覆盖，其SHA-256固定为`9beca76b8e5d453da0ee232b39114a641f7aa2ce2343c895d0f14636b4d5c61a`；机器可读勘误在`experiments/v2/w5/SUMMARY_ERRATA.json`，正式引用使用`w5_summary_corrected.json`。

该缺陷只影响离散结论标签，不影响五次训练、checkpoint、原始指标、paired delta或参数量。修复后增加反例测试；只重跑确定性的汇总判定，没有重训模型，也没有访问V1 test。

## 可审计产物

- 原始结果与checkpoint：`experiments/v2/w5/A2-L/`；
- 原汇总、勘误、纠正版汇总：`experiments/v2/w5/`；
- 五份run manifest：`experiments/v2/run_manifests/w5-a2-l-*.json`；
- catalog：25次运行，覆盖A0、A1、A2-G、A3、A2-L各五个seed；
- 完整性测试：加载全部A2-L checkpoint、核对共享初始backbone、重算mean/std、验证纠正前后文件哈希和test边界。

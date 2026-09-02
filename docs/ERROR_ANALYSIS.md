# Wireless V2 固定 Validation 错误分析

## 1. 分析边界

本报告只使用V2固定validation与20个已提交checkpoint重新推理，不训练、不调参、不访问V1 test。四个arm各覆盖五个预注册seed；总体与per-SNR结果先和历史JSON逐项核对，全部一致。

混淆矩阵把五个seed的计数相加后再按真实类别行归一化。同一validation样本因此会被五个模型各预测一次，聚合计数表示“跨seed预测行为”，不是五倍独立数据。class×SNR热力图报告五seed准确率均值，原始JSON同时保留sample std。

产物：

- `experiments/v2/analysis/analysis_summary.json`：四arm×五seed完整派生结果；
- `accuracy_vs_snr.svg`：四条五seed均值曲线，虚线为±1个sample std；
- `confusion_high_snr.svg`：`SNR≥10 dB`四arm混淆矩阵；
- `confusion_low_snr.svg`：`SNR≤-10 dB`四arm混淆矩阵；
- `class_snr_accuracy.svg`：四arm的class×SNR准确率热力图。

所有数字仍是同一validation上的开发结果；best checkpoint由validation loss选择，因此带有模型选择造成的乐观偏差。

## 2. SNR主导的性能变化

| Arm | 低SNR Accuracy（≤−10 dB） | 低SNR Macro F1 | 高SNR Accuracy（≥10 dB） | 高SNR Macro F1 |
|---|---:|---:|---:|---:|
| A0 | 11.21% ± 0.35 pp | 6.36% ± 0.85 pp | 61.82% ± 3.60 pp | 58.80% ± 3.90 pp |
| A1/A2-T | 12.77% ± 0.43 pp | 8.66% ± 0.90 pp | 80.79% ± 1.67 pp | 79.00% ± 2.14 pp |
| A2-G | 13.74% ± 1.54 pp | 9.56% ± 2.48 pp | 80.90% ± 0.92 pp | 79.34% ± 0.90 pp |
| A3 | 12.94% ± 1.29 pp | 8.55% ± 1.61 pp | 79.49% ± 3.98 pp | 77.18% ± 4.97 pp |

A1在`-20 dB`的Accuracy只有`9.28% ± 0.45 pp`，接近11分类机会水平`9.09%`；到`-10 dB`提高到`23.07% ± 1.43 pp`，`0 dB`达到`77.31% ± 1.67 pp`，高SNR平台约`80%`。这直接支持“极低SNR错误主要由噪声主导”；它不支持“某个网络能从接近纯噪声的窗口中稳定恢复调制类别”。

## 3. 低SNR：AM-SSB成为预测汇聚类

四个arm在低SNR都把大量其他类别预测成AM-SSB。以A1为例：

- QPSK→AM-SSB：`85.11%`；
- CPFSK→AM-SSB：`83.69%`；
- 8PSK→AM-SSB：`83.58%`；
- BPSK→AM-SSB：`83.27%`；
- WBFM→AM-SSB：`70.58%`；
- AM-DSB→AM-SSB：`68.58%`。

因此A1低SNR下AM-SSB recall看似很高（`87.69%`），但precision只有`10.89%`。这不是“AM-SSB识别特别成功”，而是模型在信息不足时大量塌缩到该类别。

预先提出的“AM-SSB可能有大量近静默样本”假设没有被本分析直接验证：本项目没有对源音频能量或静默比例做标注分析；而且高SNR时A1的AM-SSB precision/recall为`97.14% / 86.08%`，并未表现为持续难类。安全结论只能是“AM-SSB在低SNR成为预测汇聚类”，不能把原因确定为静默音频。

## 4. 高SNR残余错误：不是噪声消失就全部解决

### 4.1 QAM16 / QAM64

A1高SNR仍有明显双向混淆：

- QAM16→QAM64：`43.41%`；
- QAM64→QAM16：`48.69%`；
- QAM16高SNR平均准确率：`53.09%`；
- QAM64高SNR平均准确率：`49.89%`。

这说明该错误不是单纯低SNR噪声导致，高SNR下仍有数据表征或模型判别能力不足。QAM16星座点可以看成QAM64星座结构的一部分、单个窗口只有128个采样点，是合理机制解释，但当前实验没有直接估计符号数、同步误差或星座覆盖率，因此只能称“合理解释”，不能称已证明因果。

早期“QAM16 recall约4.1%”来自V1 test，不能与这里的V2 validation混用；本报告只引用上面的V2固定validation结果。

### 4.2 WBFM / AM-DSB

A1高SNR仍观察到：

- AM-DSB→WBFM：`50.77%`；
- WBFM→AM-DSB：`29.23%`；
- AM-DSB高SNR平均准确率：`49.23%`；
- WBFM高SNR平均准确率：`69.65%`。

这支持“二者在当前RadioML窗口中存在持续混淆”，但不能仅凭混淆矩阵证明原因一定是源音频静音。音频驱动的模拟调制在低活动段可能呈现相似波形，是待进一步做信号能量/瞬时频率分析的机制假设。

## 5. 聚合方式与Dropout的错误结构

A2-T与A2-G的高SNR总Accuracy几乎相同（`80.79%`与`80.90%`），但错误方向不同：

- A2-T的QAM16→QAM64 / QAM64→QAM16为`43.41% / 48.69%`；
- A2-G为`62.72% / 26.72%`；
- A2-T的WBFM→AM-DSB / AM-DSB→WBFM为`29.23% / 50.77%`；
- A2-G为`51.47% / 22.08%`。

观察到的是“归纳偏置改变了错误方向”，不是“某种聚合解决了这些类别”。结合总体Macro F1 paired delta只有`-0.25 ± 1.38 pp`且方向不一致，仍不足以promote A2-G。

A3高SNRMacro F1为`77.18% ± 4.97 pp`，低于A1的`79.00% ± 2.14 pp`且波动更大；典型残余错误包括QAM16→QAM64 `57.84%`、AM-DSB→WBFM `54.75%`。这与原W3结论一致：当前证据不支持去掉Dropout，但不能外推为“Dropout对所有调制识别都有效”。

## 6. 可用于面试的技术判断

1. 先按SNR拆分错误，才能区分噪声主导错误与高SNR仍存在的表征问题；只看总体Accuracy会把两类问题混在一起。
2. recall高不等于类别学得好：低SNR AM-SSB recall高、precision却接近随机，暴露了预测塌缩。
3. 容量接近的模型可以有相近总分但不同混淆方向；比较模型不能只看一个平均分。
4. 混淆矩阵能证明“发生了什么”，不能单独证明物理机制；星座嵌套和音频静默仍是需要额外信号级证据的假设。
5. 本报告使用固定validation并受到同集checkpoint选择的乐观偏差，不是新的独立test或真实空口证据。

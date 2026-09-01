# W3 学习交接包：容量受控消融与技术判断

## A. 一句话目标

面试语言：W3用五个预注册seed和两条独立的一维消融，区分“聚合方式”和“Dropout”对固定validation结果的影响，让模型选择建立在可审计的成对证据上，而不是单次最高Accuracy。

## B. 最重要的5个知识点

### 1. Controlled ablation（受控消融）

- 是什么：尽量只改变要研究的因素，其余数据、seed、训练预算、优化器和评测口径保持一致。
- 为什么需要：否则结果变化可能来自多个因素，无法回答“究竟是什么造成差异”。
- 不这样做会怎样：即使新模型分数更高，也只能说整体配置不同，不能给出结构判断。
- 与上一阶段关系：W2建立多seed成对比较；W3把同一套公平比较方法用于两个更具体的问题。
- 面试追问：A2为什么仍不能叫严格单变量实验？因为两种head的输入维度、隐藏宽度和函数结构不同，参数量接近也不代表表达能力相同。

### 2. Shared backbone（共享主干）

- 是什么：A2-T与A2-G共用最终聚合前的三层Conv1d、三层BatchNorm、ReLU和两次MaxPool，固定输出`(B,128,32)`。
- 为什么需要：把主要特征提取路径固定住，减少“卷积特征提取变化”对聚合比较的干扰。
- 不这样做会怎样：如果卷积层也变了，无法区分差异来自backbone还是aggregation head。
- 与上一阶段关系：A2-T直接复用W2的A1；W3按相同seed重建并保存A1的初始backbone，A2-G加载同一份state。
- 面试追问：同seed是否代表两个完整模型逐参数同初始化？不是。这里只保证共享backbone相同；不同head按照冻结规则各自初始化。

### 3. 参数量、MACs与FLOPs

- 是什么：参数量衡量可训练权重规模；MACs估算乘加次数；本项目约定`1 MAC≈2 FLOPs`。
- 为什么需要：若A2-G靠大幅增加容量换来分数变化，就不能把差异主要讨论为聚合设计差异。
- 不这样做会怎样：对照可能变成“大模型对小模型”，公平性不足。
- 与上一阶段关系：W2只报告A0/A1的总参数量；W3进一步报告head参数、全模型参数和计算量。
- 面试追问：为什么参数量接近但延迟不一定相同？实际延迟还受算子类型、内存访问、并行效率、框架开销和硬件影响；MACs只是估算。

### 4. Dropout消融

- 是什么：训练时以概率`p`随机屏蔽激活；推理时关闭随机屏蔽。A3仅将A1的`p=0.3`改为`p=0`。
- 为什么需要：检验当前训练协议中Dropout是否带来有用的正则化，而不是凭经验保留。
- 不这样做会怎样：只能说模型“用了Dropout”，不能说明这个选择是否有证据支持。
- 与上一阶段关系：A1的五个结果直接作为`p=0.3`侧，A3补齐相同五个seed的`p=0`侧。
- 面试追问：Dropout为什么不改变参数量？它是对激活的随机掩码操作，不包含可训练权重。

### 5. 从结果形成判断，而不是只报最高分

- 是什么：同时看mean、sample std、同seed paired delta、方向一致性、参数/计算成本和实验限制。
- 为什么需要：平均值可能被单个异常seed拉动；只看最好seed会产生选择偏差。
- 不这样做会怎样：容易把小于随机波动的差异包装成“模型提升”。
- 与上一阶段关系：W2学会多seed统计；W3用它决定是否promote两个变体。
- 面试追问：A2-G平均Macro F1略高，为什么没有替换A2-T？平均只高约`0.25`个百分点，而配对差值sample std约`1.38`个百分点且方向不一致，证据不足以支持稳定优势。

## C. 源码学习路径

1. `src/signal_modulation/model.py`
   - 类：`TemporalCNN1D`、`GlobalPoolingTemporalCNN1D`
   - 先读：两者`features`，确认共享backbone的层与输出shape。
   - 后读：各自`classifier`和`forward`。
   - 重点：A2-T保留8个时序分箱；A2-G压成1个全局向量；A3只通过构造参数改变Dropout。

2. `src/signal_modulation/w3_ablation.py`
   - 函数：`reconstruct_a2t_initial_backbone`、`create_w3_model`、`verify_dropout_initialization`、`architecture_profile`
   - 先读：初始化哈希与backbone提取。
   - 后读：参数/MACs和延迟测量。
   - 重点：代码如何证明共享初始backbone、A1/A3初始state一致，以及确定性量为什么只报原值。

3. `scripts/run_v2_ablation_experiments.py`
   - 函数：`load_reused_a2t_records`、`create_initialization_artifacts`、`run_one_experiment`、`main`
   - 先读：`W3_MATRIX`和A2-T复用校验。
   - 后读：10次运行循环、失败记录和汇总。
   - 重点：为什么只训练A2-G/A3；怎样加载固定manifest；怎样禁止覆盖目录；为何实现commit必须早于结果。

4. `src/signal_modulation/v2_statistics.py`
   - 函数：`summarize_w3_results`、`_paired_summary`、`descriptive_summary`
   - 先读：两组paired delta的方向定义。
   - 后读：per-SNR汇总。
   - 重点：`A2-T-A2-G`和`A2-T-A3`是两条独立比较，不能混为2×2。

5. `tests/test_w3_artifacts.py`
   - 类：`W3ArtifactTests`
   - 先读：checkpoint与结果JSON对应检查。
   - 后读：汇总重算、初始backbone和容量锁检查。
   - 重点：如何从文件和哈希证明实验没有缺seed、替换结果或损坏产物。

## D. 必须能口述的完整链路

### Tensor shape链

```text
输入 I/Q: (B, 2, 128)
-> Conv1d 2→64: (B, 64, 128)
-> MaxPool /2: (B, 64, 64)
-> Conv1d 64→128: (B, 128, 64)
-> MaxPool /2: (B, 128, 32)
-> Conv1d 128→128: (B, 128, 32)

A2-T/A1:
-> AvgPool1d(4): (B, 128, 8)
-> Flatten: (B, 1024)
-> Linear 1024→128 -> Dropout(0.3) -> Linear 128→11
-> logits: (B, 11)

A2-G:
-> AdaptiveAvgPool1d(1): (B, 128, 1)
-> Flatten: (B, 128)
-> Linear 128→947 -> Dropout(0.3) -> Linear 947→11
-> logits: (B, 11)

A3:
与A2-T shape完全相同，只把Dropout(0.3)改为Dropout(0)。
```

### 正式实验链

```text
W0预注册协议和5个seed
-> W1固定split manifest
-> W2 A1五次原始JSON/checkpoint（映射为A2-T）
-> 校验W2 summary hash
-> 每个seed保存同一初始backbone / 校验A1与A3初始state
-> 只训练A2-G和A3各5次
-> 每次按最低validation loss保存best checkpoint
-> 固定validation计算Accuracy、Macro F1、per-SNR与混淆矩阵
-> 保存原始JSON和checkpoint hash
-> 分别计算A2-T−A2-G、A2-T−A3 paired delta
-> 输出mean±sample std、参数、MACs/FLOPs和固定延迟
-> 根据方向、波动、成本和边界决定是否promote
```

## E. 简化代码手写目标

### 目标1：写出全局池化head的forward

```python
def forward(self, x):
    x = self.backbone(x)       # (B, 128, 32)
    x = self.global_pool(x)    # (B, 128, 1)
    x = x.flatten(1)           # (B, 128)
    return self.classifier(x)  # (B, 11)
```

必须能解释：全局池化丢掉粗粒度时间位置，而A2-T保留8个分箱。

### 目标2：写出同seed paired delta与sample std

```python
seeds = sorted(set(a) & set(b))
deltas = [a[s] - b[s] for s in seeds]
mean_delta = sum(deltas) / len(deltas)
sample_std = statistics.stdev(deltas)  # 分母是 n-1
```

必须能解释：先同seed相减，再汇总差值；不能把两个模型各自的均值相减后假装得到了每个pair的信息。

## F. ⑤拥有权验证：3道源码题与标准答案

当前状态：**未完成**。先学习下面的题目与答案；用户确认学习完成后，才更新`DEV_STATE`并允许申请W4。

### 题1：代码如何保证A2-T和A2-G共享同一初始backbone？

标准答案：`reconstruct_a2t_initial_backbone`按每个预注册`run_seed`重新设置随机状态，构造未训练的`TemporalCNN1D`，提取最终聚合层之前的`features[:-1]`并计算state hash；正式A2-G由`create_w3_model`加载这份state，运行JSON再次记录加载后的hash。A2-T不重训，而是复用W2 A1结果。它保证的是共享backbone的初始state一致，不保证两个不同head逐参数一致。

代码依据：`src/signal_modulation/w3_ablation.py`、`scripts/run_v2_ablation_experiments.py`、`experiments/v2/w3/initial_backbones/initialization_manifest.json`。

### 题2：为什么A2参数只差28个，仍不能称严格单变量实验？

标准答案：两边backbone相同、参数预算接近，但A2-T把`(B,128,32)`变成8个时序分箱并用`1024→128→11`分类；A2-G把时间轴全局平均后用`128→947→11`分类。它们的输入信息、隐藏宽度和函数结构不同，参数量相近只排除了明显容量差异，不能让表达能力完全相同。因此只能称“共享backbone与近似参数预算下的容量受控比较”。

代码依据：`src/signal_modulation/model.py`、`tests/test_complexity.py`、`experiments/v2/w3/matrix_lock.json`。

### 题3：根据真实结果，你会promote A2-G或A3吗？为什么？

标准答案：暂时都不promote。A2-G平均Macro F1为`55.73%`，A2-T为`55.47%`，但`A2-T−A2-G`只有`-0.25 ± 1.38`个百分点且seed方向不一致，微小均值差不足以证明稳定优势。A3去掉Dropout后Macro F1为`54.40% ± 3.13`，低于A2-T的`55.47% ± 1.05`；配对差`A2-T−A3=+1.08 ± 3.54`个百分点，波动也很大。安全判断是保留A1/A2-T作为默认，记录两个中性/负向实验，不宣称普遍规律。

实验依据：`experiments/v2/w3/w3_summary.json`和10份W3原始`validation_result.json`。

## G. 面试资产

- 新技术判断：参数量几乎相同不等于严格单变量；聚合设计需要结合paired delta与方向稳定性解释。
- 新的中性实验：A2-G平均Macro F1仅比A2-T高约`0.25`个百分点，但差异小于波动且方向不一致，因此不promote。
- 新的负向实验：去掉Dropout没有得到稳定收益，均值下降且跨seed波动增大，因此保留`p=0.3`。
- 新trade-off：A2-G参数和MACs略低，但固定硬件延迟未显示相应优势，说明理论计算量与真实延迟不是同一概念。
- 新数据：15条可审计记录（5条复用A2-T、10条新训练）、两组完整`n_pairs=5`、10个新checkpoint、5个初始backbone。
- 新限制条件：结论只适用于RadioML 2016.10A固定validation和这5个seed；没有新test结论、统计显著性或真实空口证据。

基础设施失败为0，重跑为0。这里的“中性/负向实验”是有价值的实验结论，不应伪装成算法提升。

## H. 遗忘风险

1. 容易把“参数量差0.0125%”误说成“严格单变量”；必须同时说明head函数结构仍不同。
2. 容易看到A2-G均值高0.25个百分点就说“更好”；必须一起说paired std约1.38个百分点、方向不一致、只做描述性统计。
3. 容易把MACs/FLOPs、参数量和延迟混为一谈；前两者是确定性结构量，延迟是特定硬件和测量协议下的运行量。

## 本阶段验收口径

- 能讲：解释两条独立消融为什么公平、为什么结论有限。
- 能写：写出全局池化forward与paired delta/sample std。
- 能改：能定位aggregation head或Dropout参数，同时知道任何影响数值的修改都需要升级协议并重跑受影响实验。
- 能复现：能从matrix lock、split manifest、seed、原始JSON和checkpoint hash还原一次运行条件。
- 能分析：能依据均值、波动、方向、参数和延迟作出“不promote”的真实判断，而不是只挑最高分。

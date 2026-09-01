# W4 学习交接包：Run Manifest、复现验收与面试叙事

## A. 一句话目标

面试语言：W4把W1–W3的代码、配置、原始结果和checkpoint串成可审计的run manifest，使任意一次历史实验都能仅凭manifest完成身份校验并重建单次启动命令，同时用README和技术报告明确实验能力与结果边界。

## B. 最重要的5个知识点

### 1. Run manifest

- 是什么：一次实验的“身份证和启动说明”，记录数据、split、seed、模型、训练配置、环境、代码commit、输入产物和输出产物哈希。
- 为什么需要：只有结果JSON或checkpoint无法完整回答“用什么代码、什么数据、什么随机性和什么环境得到”。
- 不这样做会怎样：历史实验可能只能看成绩，不能审计或重启；路径、配置或checkpoint一旦混淆就难以发现。
- 与上一阶段关系：W1提供split身份，W2/W3提供正式运行和产物，W4把它们绑定成单次运行记录。
- 面试追问：manifest和普通config有什么区别？config主要描述“准备怎么跑”；manifest还记录实际数据/代码/环境/产物身份和哈希，是可审计的历史事实。

### 2. 哈希完整性与可复现边界

- 是什么：SHA-256用于判断文件字节是否与登记时完全一致；Git commit标识代码版本。
- 为什么需要：文件名相同不代表内容相同，manifest必须发现数据、split、依赖、结果或checkpoint被替换。
- 不这样做会怎样：命令可能能运行，却已经不是同一个实验。
- 与上一阶段关系：W1固定数据和索引哈希，W2/W3记录checkpoint哈希，W4统一验证。
- 面试追问：哈希一致是否保证结果逐位一致？不保证。它证明输入文件身份一致；硬件、驱动和底层算子仍可能影响数值。

### 3. Dry-run与显式执行

- 是什么：默认只校验manifest和文件并打印命令；只有显式`--execute`才进行训练。
- 为什么需要：复现审计不应该意外消耗GPU、覆盖产物或产生一组未登记结果。
- 不这样做会怎样：一次“看看能不能复现”的操作可能直接开始长训练，甚至污染正式实验目录。
- 与上一阶段关系：W2/W3禁止覆盖原始JSON；W4的重启入口继续执行这一纪律。
- 面试追问：dry-run实际验证了什么？验证数据SHA、split、协议、依赖、历史结果和checkpoint哈希，并重建单配置单seed命令；不验证重新训练后的数值。

### 4. README中的实验能力与生产能力

- 是什么：实验能力是能训练、比较、分析和复现；生产能力还需要服务接口、监控、吞吐、故障恢复、线上数据和部署验收。
- 为什么需要：这个项目是通信AI实验作品集，不是在线无线识别生产系统。
- 不这样做会怎样：把checkpoint和离线评测写成生产能力，面试追问部署、漂移或SLA时会暴露夸大。
- 与上一阶段关系：W2/W3提供实验判断；W4负责把能力和限制写成对外可审查口径。
- 面试追问：为什么V2结果不能写成test accuracy？V2模型选择和消融全部使用固定validation，没有新的独立test。

### 5. 从W1到W4的完整叙事

- 是什么：W1固定数据身份，W2验证多seed模型差异，W3进行容量受控消融，W4保存证据并形成技术决策。
- 为什么需要：面试官更关心你如何控制实验、排除混杂因素和解释结果，而不是只听一个最高准确率。
- 不这样做会怎样：项目容易退化成“下载公开数据、跑CNN、报56%”。
- 与上一阶段关系：W4不产生新模型分数，而是把前三阶段变成可验证的完整工程故事。
- 面试追问：最终做了什么模型决定？保留A1/A2-T；A2-G差异小于波动不promote，A3去Dropout没有可靠收益也不promote。

## C. 源码学习路径

1. `src/signal_modulation/run_manifest.py`
   - 函数：`validate_run_manifest`、`verify_run_manifest`、`build_replay_commands`
   - 先读：schema和冻结字段校验。
   - 后读：仓库文件路径约束、SHA-256校验和命令构造。
   - 重点：哪些字段被修改会立即拒绝；为什么校验阶段不读取信号或运行模型。

2. `scripts/create_v2_run_manifests.py`
   - 函数：`_result_paths`、`build_manifest`、`main`
   - 先读：怎样只接受20条W2/W3 completed结果。
   - 后读：如何复制训练事实、记录provenance并生成catalog。
   - 重点：实现commit为什么必须先提交；为什么manifest不能事后覆盖。

3. `scripts/replay_v2_run.py`
   - 函数：`main`、`_verify_environment`、`execute_replay`
   - 先读：默认dry-run分支。
   - 后读：显式执行如何按W2/W3配置只启动一次运行。
   - 重点：`--execute`硬闸门、环境匹配、新输出目录和train/validation-only边界。

4. `tests/test_run_manifest_artifacts.py`
   - 类：`RunManifestArtifactTests`
   - 先读：catalog的20个配置×seed覆盖检查。
   - 后读：逐manifest文件哈希与命令重建测试。
   - 重点：测试如何证明“任取一份manifest都能形成合法复现计划”。

5. `README.md`与`docs/V2_CORE_REPORT.md`
   - 先读：V2结果表和核心技术取舍清单。
   - 后读：V1历史test、限制和复现章节。
   - 重点：validation/test、实验能力/生产能力、描述性统计/显著性之间的措辞边界。

## D. 必须能口述的完整链路

```text
W1 split manifest
  数据SHA + split_seed + train/validation索引哈希
-> W2/W3单次运行
  configuration + run_seed + config + environment
-> 原始运行产物
  validation_result.json + best_checkpoint.pt + SHA-256
-> W4 run manifest
  绑定数据、split、代码commit、依赖、环境、结果、checkpoint
-> catalog
  4配置 × 5 seed = 20份manifest
-> replay dry-run
  schema检查 -> 路径边界 -> 文件存在 -> SHA核对 -> 历史字段一致
-> 输出单次运行重启命令
-> 只有显式 --execute
  restricted dataset load -> fixed train/validation loaders
  -> 指定模型与seed -> fit -> validation -> 新目录产物
```

必须强调：dry-run计算原始数据文件SHA但不反序列化信号；显式执行也不会创建V1 test DataLoader。

## E. 简化代码手写目标

### 目标1：写出manifest文件校验核心

```python
def verify_file(root, relative_path, expected_sha):
    path = (root / relative_path).resolve()
    path.relative_to(root.resolve())  # 越界会抛错
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected_sha:
        raise ValueError("artifact changed")
```

需要解释：先限制路径在仓库内，再检查存在和hash；只比较文件名是不够的。

### 目标2：写出显式执行闸门

```python
verification = verify_manifest(manifest)
if not args.execute:
    print(verification)
    print(replay_command)
    return
execute_one_run(manifest, new_output_directory)
```

需要解释：默认路径必须无副作用；长时间训练只能由用户显式触发。

## F. ⑤拥有权验证：3道源码题与标准答案

当前状态：**未完成**。学习下面问题和答案后，由用户确认是否完成W4学习。

### 题1：为什么manifest里既要保存训练config，又要保存原始结果与checkpoint哈希？

标准答案：config说明训练应该怎样执行，但不能证明历史运行实际产生了哪份结果和模型。原始结果与checkpoint哈希把“运行条件”和“实际产物”绑定起来；若文件被替换，即使config没变，`verify_run_manifest`也会拒绝。代码依据是`src/signal_modulation/run_manifest.py`中的`validate_run_manifest`与`verify_run_manifest`。

### 题2：从manifest执行dry-run时，为什么不能说已经复现了56%左右的结果？

标准答案：dry-run只验证文件身份、数据SHA、环境记录和启动命令，没有执行forward、训练或validation评测，因此只能说“复现条件可重建、产物可审计”。只有显式训练并按相同协议重新计算指标，才能比较数值复现；即便如此，跨硬件也不承诺bitwise一致。代码依据是`scripts/replay_v2_run.py`的`main`分支。

### 题3：如何用W1–W4讲出完整的面试故事，而不是只报Accuracy？

标准答案：W1把数据划分随机性与训练随机性解耦，并保存固定split；W2用五个预注册seed成对比较A0/A1，发现A1的Macro F1 paired delta平均`+15.23`个百分点且方向一致；W3在共享backbone和近似参数预算下比较聚合方式，并单独做Dropout消融，结果不足以支持A2-G或A3替换A1；W4用20份manifest、checkpoint哈希、报告和135项测试固定证据。限制是所有V2结论来自固定validation，不是新test或真实空口证据。

## G. 面试资产

- 新技术判断：可复现不只是设置seed，还需要数据、split、代码、依赖、环境和产物身份。
- 新trade-off：更强的严格校验提高可信度，但顶层依赖锁和环境记录仍不等于完整容器或跨硬件逐位复现。
- 新工程证据：20份单次run manifest、1份catalog、默认无副作用的dry-run、显式单次执行入口。
- 新限制条件：W4不产生新指标；dry-run证明条件可重建，不证明模型分数重新计算一致。
- 新面试叙事：能够解释为什么两个候选变体没有promote，而不是为了简历强行宣布提升。

W4没有新失败训练；manifest篡改测试会主动拒绝seed、test边界、依赖hash或历史文件不一致，这是预期的安全失败。

## H. 遗忘风险

1. 容易把config和manifest混为一谈：config是计划参数，manifest还绑定实际代码、环境与产物。
2. 容易把dry-run说成“指标复现”：它只完成身份和启动条件验收，没有重新计算Accuracy/F1。
3. 容易在README把V2 validation写成test，或把离线实验能力写成生产部署能力。

## 本阶段验收口径

- 能讲：完整说明manifest为什么比单独checkpoint/config更可靠。
- 能写：写出安全路径+SHA校验和默认dry-run/显式执行闸门。
- 能改：能新增manifest字段并同步schema、生成器和测试，而不是只改JSON示例。
- 能复现：能从catalog选择任意run，先dry-run核验，再解释显式执行的成本和边界。
- 能分析：能把W1–W4结果讲成实验决策链，并准确区分V2 validation、V1历史test和未来独立外部数据。

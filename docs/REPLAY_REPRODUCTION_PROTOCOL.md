# A3 单次训练级复现协议

> 冻结时间：2026-09-02，正式 replay 执行之前  
> 状态：阈值已预注册，尚未执行

## 1. 固定运行身份

- run id：`w3-a3-20260901`
- manifest：`experiments/v2/run_manifests/w3-a3-20260901.json`
- 选择理由：它是冻结 seed 列表中的第一项，不依据历史分数选择
- 输出目录：`artifacts/replay/w3-a3-20260901/`
- 数据边界：只创建固定 train/validation DataLoader；禁止创建或评估 V1 test
- effective模型配置：A3使用`TemporalCNN1D(dropout=0.0)`；历史manifest中的`0.3`按`experiments/v2/w3/A3/ERRATA.json`解释

## 2. 历史参照值

参照文件保持原字节不变：
`experiments/v2/w3/A3/run_seed_20260901/validation_result.json`。

| 项目 | 历史值 |
|---|---:|
| best epoch | 6 |
| validation Accuracy | 55.172727% |
| validation Macro F1 | 56.420482% |
| best validation loss | 1.1784395901 |

## 3. 执行前冻结的验收规则

以下规则在查看 replay 结果前冻结：

- validation Accuracy绝对差值不超过`1.0`个百分点；
- validation Macro F1绝对差值不超过`1.0`个百分点；
- best epoch、best validation loss、环境、耗时和checkpoint哈希必须原样记录并逐项比较；
- best epoch不单独作为硬失败阈值，但只要发生变化就必须分析，不能隐藏；
- 超出阈值时仍保留结果并判为复现未通过，不允许调参、换seed或反复运行来凑阈值；
- 指标进入验收的前提是manifest、数据、split和依赖文件的启动前哈希检查通过；
- 本次验收是同机训练级复现，不证明跨硬件逐位一致。

## 4. 产物规则

- replay只写新目录，不覆盖W3历史JSON或checkpoint；
- `REPRODUCTION.md`记录历史值、replay值、逐项差值和最终判定；
- 本协议提交必须早于replay产物和结论提交。

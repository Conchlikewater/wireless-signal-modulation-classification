# w3-a3-20260901 训练级复现记录

## 运行身份

- manifest：`experiments/v2/run_manifests/w3-a3-20260901.json`
- 历史实现commit：`596900aa81c0b8502a05563c1986a9231b61462d`
- replay实现commit：`370e6f70a300c8408c910acc9cf2b53123b27b66`
- run seed：`20260901`
- split seed：`20260812`
- scope：固定validation，未创建或评估V1 test
- A3 effective模型：`TemporalCNN1D(dropout=0.0)`；历史manifest的通用`config.dropout=0.3`按A3 ERRATA解释

## 预注册阈值

`docs/REPLAY_REPRODUCTION_PROTOCOL.md`在执行前提交。Accuracy与Macro F1的允许绝对差值均为`1.0`个百分点；超出时不调参、不换seed、不重跑凑结果。

## 逐项结果

| 项目 | 历史值 | replay值 | 差值 | 判定 |
|---|---:|---:|---:|---|
| best epoch | 6 | 6 | 0 | 记录一致 |
| validation Accuracy | 55.172727% | 55.172727% | 0.000000 pp | 通过 |
| validation Macro F1 | 56.420482% | 56.420482% | 0.000000 pp | 通过 |
| best validation loss | 1.1784395901 | 1.1784395901 | 0 | 记录一致 |
| elapsed time | 40.8266 s | 51.6323 s | +10.8057 s | 仅记录，不作精度验收 |

- 历史checkpoint SHA-256：`6730fa44397b179f337660db8ff18c3d8b79057ffdb58375f3511f760777ffaf`
- replay checkpoint SHA-256：`6730fa44397b179f337660db8ff18c3d8b79057ffdb58375f3511f760777ffaf`
- 原始replay结果：`artifacts/replay/w3-a3-20260901/A3/run_seed_20260901/validation_result.json`
- `test_set_used=false`

## 环境

- Windows 11
- Python 3.12.10
- NumPy 2.5.2
- PyTorch 2.12.1+cu130
- CUDA 13.0
- cuDNN 9.2
- NVIDIA GeForce RTX 4060 Laptop GPU
- deterministic algorithms：启用
- `cudnn.benchmark=false`

## 结论和限制

本次同机训练级复现通过预注册阈值，并得到相同checkpoint哈希。它支持“当前仓库、当前数据身份和当前环境可以重启这一历史运行”，但不证明跨硬件、跨驱动或未来依赖版本能够逐位一致，也不提供新的test或域外泛化证据。

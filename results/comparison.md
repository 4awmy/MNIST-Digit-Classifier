# Measured model comparison

| Model | Validation accuracy | Test accuracy | Macro F1 | Selected fit (s) | Size (MiB) |
|---|---:|---:|---:|---:|---:|
| Random Forest | 96.65% | 96.95% | 0.9693 | 4.33 | 18.94 |
| MLP | 97.47% | 97.62% | 0.9761 | 39.74 | 2.72 |
| CNN | 98.74% | 99.02% | 0.9901 | 54.91 | 1.43 |

Context for interpreting the table:

- gpu_devices differs: rf=[]; mlp=["/physical_device:GPU:0"]; cnn=["/physical_device:GPU:0"]
- Random Forest uses CPU even if GPU devices are available. Qualify timing comparisons.
- 10,000-image timing is batched inference, not single-drawing app response time.
- RF tree/node counts and neural-network parameter counts are different complexity measures.

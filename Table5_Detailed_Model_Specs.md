# Table 5 — Detailed Model Specifications and Hyper-parameters

| Attribute | 3B | 9B | 27B | 3B (1T Tokens) |
|---|---:|---:|---:|---:|
| Vocab Params | 331M | 496M | 662M | 331M |
| Active Params | 612M | 1.66B | 4.14B | 612M |
| Total Params | 2.97B | 9.18B | 27.0B | 2.97B |
| Layers | 12 | 18 | 30 | 12 |
| Leading Dense Layers | — | 1 | — | 1 |
| Routed Experts | 64 | 64 | 72 | 64 |
| Active Experts | — | 6 | — | 6 |
| Shared Experts | — | 2 | — | 2 |
| Dimension | 1280 | 1920 | 2560 | 1280 |
| FFN Dimension | 896 | 1280 | 1536 | 896 |
| Load Balancing Method | Loss-Free (Wang et al., 2024) | Loss-Free (Wang et al., 2024) | — | Loss-Free (Wang et al., 2024) |
| Attention Heads | 16 | 24 | 32 | 16 |
| Attention Dimension | 128 | 128 | 128 | 128 |
| Attention Variant | MLA (Liu et al., 2024a) | MLA (Liu et al., 2024a) | MLA (Liu et al., 2024a) | MLA (Liu et al., 2024a) |
| KV Rank | — | 512 | — | 512 |
| Position Embedding | RoPE (Su et al., 2024) | RoPE (Su et al., 2024) | RoPE (Su et al., 2024) | RoPE (Su et al., 2024) |
| RoPE Dimension | 64 | 64 | 64 | 64 |
| RoPE θ | 10000 | 10000 | 10000 | 10000 |
| Layer Norm Type | RMSNorm (Zhang and Sennrich, 2019) | RMSNorm (Zhang and Sennrich, 2019) | RMSNorm (Zhang and Sennrich, 2019) | RMSNorm (Zhang and Sennrich, 2019) |
| Layer Norm ε | 1e-20 | 1e-20 | 1e-20 | 1e-20 |
| mHC/HC Expansion Rate n | 4 | 4 | 4 | 4 |
| mHC/HC Gating Factor Init α | 0.01 | 0.01 | 0.01 | 0.01 |
| mHC Sinkhorn-Knopp tmax | 20 | 20 | 20 | 20 |
| Sequence Length | 4096 | 4096 | 4096 | 4096 |
| Vocab Size | 129280 | 129280 | 129280 | 129280 |
| Batch Size | 320 | 512 | 1280 | 2560 |
| Training Steps | 30000 | 50000 | 50000 | 100000 |
| Training Tokens | 39.3B | 105B | 262B | 1.05T |
| Warmup Steps | 2000 | 2000 | 2000 | 2000 |
| Optimizer | AdamW (Loshchilov and Hutter, 2017) | AdamW (Loshchilov and Hutter, 2017) | AdamW (Loshchilov and Hutter, 2017) | AdamW (Loshchilov and Hutter, 2017) |
| AdamW Betas | (0.9, 0.95) | (0.9, 0.95) | (0.9, 0.95) | (0.9, 0.95) |
| AdamW ε | 1e-20 | 1e-20 | 1e-20 | 1e-20 |
| Base Learning Rate | 8.6e-4 | 5.9e-4 | 4.0e-4 | 9.0e-4 |
| Lr Scheduler | Step | Step | Step | Step |
| Lr Decay Step Ratio | [0.8×, 0.9×] | [0.8×, 0.9×] | [0.8×, 0.9×] | [0.8×, 0.9×] |
| Lr Decay Rate | [0.316, 0.1] | [0.316, 0.1] | [0.316, 0.1] | [0.316, 0.1] |
| Weight Decay | 0.1 | 0.1 | 0.1 | 0.1 |

**Notes**

- Table follows the original paper's Table 5 (Detailed Model Specifications and Hyper-parameters).
- Cells marked "—" indicate entries not shown or not specified in the provided excerpt.

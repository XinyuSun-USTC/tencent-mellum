# Delivery reduced cumulative screen

A32 / B6 / C8; native fast control, not historical Step0. Not formal assessment-scale ablation.

| Configuration | A tok/s/GPU | B TPOTp50 ms | C TPOTp99 ms | C TTFTp99 ms |
|---|---:|---:|---:|---:|
| Native fast control (not historical Step0) | 584.769 | 39.124 | 49.513 | 1133.564 |
| + decode graph | 908.391 | 8.998 | 17.140 | 1057.081 |
| + fused QK and native routing | 1101.481 | 4.925 | 11.921 | 1024.438 |
| + prefix and metadata reuse | 1092.948 | 4.901 | 10.380 | 779.620 |

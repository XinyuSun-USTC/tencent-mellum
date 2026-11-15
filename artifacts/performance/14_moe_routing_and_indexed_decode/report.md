# MoE候选自动实验

状态：`complete`。结果仅限本地开发负载，默认配置未切换。

| job | 集合 | 位置匹配 | 完整序列 | wall均值秒 |
| --- | --- | --- | --- | --- |
| screen_B0 | K4 | 96.88% | 75.00% | 3.142 |
| screen_B0 | S32 | 93.36% | 87.50% | 5.468 |
| screen_B0 | L32 | 89.06% | 71.88% | 17.053 |
| screen_CPD | K4 | 100.00% | 100.00% | 3.938 |
| screen_CPD | S32 | 97.85% | 96.88% | 13.462 |
| screen_CPD | L32 | 99.61% | 96.88% | 25.159 |
| screen_shared_routing | K4 | 100.00% | 100.00% | 3.844 |
| screen_shared_routing | S32 | 97.85% | 96.88% | 12.158 |
| screen_shared_routing | L32 | 99.61% | 96.88% | 23.941 |
| screen_indexed_decode | K4 | 96.88% | 75.00% | 3.499 |
| screen_indexed_decode | S32 | 94.34% | 84.38% | 9.353 |
| screen_indexed_decode | L32 | 93.95% | 81.25% | 21.055 |
| D200_CPD | D200 | 96.81% | 87.00% | 310.909 |
| D200_shared_routing | D200 | 96.81% | 87.00% | 279.371 |
| timing_shared_routing | S32_A0 | 97.85% | 96.88% | 13.315 |
| timing_shared_routing | L32_A0 | 99.61% | 96.88% | 25.079 |
| timing_shared_routing | S32_B1 | 97.85% | 96.88% | 12.258 |
| timing_shared_routing | L32_B1 | 99.61% | 96.88% | 23.970 |
| timing_shared_routing | S32_B2 | 97.85% | 96.88% | 12.276 |
| timing_shared_routing | L32_B2 | 99.61% | 96.88% | 23.976 |
| timing_shared_routing | S32_A3 | 97.85% | 96.88% | 13.406 |
| timing_shared_routing | L32_A3 | 99.61% | 96.88% | 25.097 |
| timing_shared_routing | S32_A4 | 97.85% | 96.88% | 13.417 |
| timing_shared_routing | L32_A4 | 99.61% | 96.88% | 25.099 |
| timing_shared_routing | S32_B5 | 97.85% | 96.88% | 12.293 |
| timing_shared_routing | L32_B5 | 99.61% | 96.88% | 23.978 |
| timing_shared_routing | S32_B6 | 97.85% | 96.88% | 12.274 |
| timing_shared_routing | L32_B6 | 99.61% | 96.88% | 23.981 |
| timing_shared_routing | S32_A7 | 97.85% | 96.88% | 13.392 |
| timing_shared_routing | L32_A7 | 99.61% | 96.88% | 25.104 |
| profile_shared_routing | S32_candidate | 97.85% | 96.88% | 12.307 |
| profile_shared_routing | L32_candidate | 99.61% | 96.88% | 24.066 |

## 自动决策

```json
{
  "skipped": [],
  "selected_d200": [
    "shared_routing"
  ],
  "passing_d200": [
    "shared_routing"
  ],
  "historical_screen_match": {
    "S32": {
      "exact_sequences": 32,
      "records": 32
    },
    "L32": {
      "exact_sequences": 32,
      "records": 32
    }
  },
  "historical_D200_match": {
    "D200": {
      "exact_sequences": 200,
      "records": 200
    }
  },
  "best": "shared_routing",
  "profile_passed": true,
  "service_passed": true
}
```

## ABBA

```json
{
  "shared_routing": {
    "cohorts": {
      "S32": {
        "cpd_seconds": [
          13.314901278121397,
          13.406270467909053,
          13.41687499312684,
          13.392143368022516
        ],
        "candidate_seconds": [
          12.258487408980727,
          12.27642812486738,
          12.292742680991068,
          12.274492105003446
        ],
        "cpd_median_seconds": 13.399206917965785,
        "candidate_median_seconds": 12.275460114935413,
        "speedup": 1.0915441696285684,
        "stable": true
      },
      "L32": {
        "cpd_seconds": [
          25.079265245003626,
          25.096769852098078,
          25.09942311490886,
          25.104139670962468
        ],
        "candidate_seconds": [
          23.96950156800449,
          23.975783445872366,
          23.97841151896864,
          23.98070671199821
        ],
        "cpd_median_seconds": 25.09809648350347,
        "candidate_median_seconds": 23.977097482420504,
        "speedup": 1.046752906681255,
        "stable": true
      }
    },
    "stable": true,
    "batch_verified": true,
    "geometric_mean_speedup": 1.068913949918178,
    "performance_eligible": true
  }
}
```

CPD只是本地正确性对照，不能用其较慢的耗时替代题目官方Baseline放大成绩。

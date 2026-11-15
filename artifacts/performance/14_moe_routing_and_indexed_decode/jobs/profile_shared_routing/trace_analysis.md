# Profile摘要

仅为诊断trace，不是正式A/B/C性能分数。单位为毫秒。

GPU kernel数：185796；模块归因覆盖：99.9%。

| 项目 | ms |
| --- | --- |
| summed_gpu_kernel_us | 2188.706 |
| kernel_active_union_us | 2188.706 |
| kernel_span_us | 5675.726 |
| kernel_gaps_within_span_us | 3487.020 |

kernel累计时长可重叠；空隙不自动等于CPU瓶颈。

## 按最内层模块归因

| 模块 | 累计kernel ms |
| --- | --- |
| moe_experts | 1662.895 |
| attention_swa | 217.577 |
| qkv | 129.155 |
| attention_full | 75.896 |
| output_projection | 59.283 |
| router | 27.825 |
| norm_post | 7.481 |
| norm_input | 7.372 |
| unattributed | 1.221 |

## 分阶段模块kernel累计ms

| 阶段 | 模块 | 累计ms |
| --- | --- | --- |
| prefill | moe_experts | 1389.367 |
| prefill | attention_swa | 135.910 |
| prefill | qkv | 59.636 |
| prefill | attention_full | 43.973 |
| prefill | output_projection | 33.308 |
| prefill | router | 9.601 |
| prefill | norm_post | 5.358 |
| prefill | norm_input | 5.231 |
| prefill | unattributed | 0.469 |
| unattributed | unattributed | 0.184 |
| decode | moe_experts | 273.528 |
| decode | attention_swa | 81.667 |
| decode | qkv | 69.519 |
| decode | attention_full | 31.923 |
| decode | output_projection | 25.975 |
| decode | router | 18.224 |
| decode | norm_input | 2.141 |
| decode | norm_post | 2.123 |
| decode | unattributed | 0.568 |

## 热点kernel

| 名称 | 次数 | 累计ms |
| --- | --- | --- |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 33348 | 425.707 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 31556 | 402.793 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 17434 | 211.589 |
| void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) | 5461 | 120.858 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 6132 | 82.530 |
| void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::bina | 9635 | 76.991 |
| Cijk_Alik_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x16x128_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB1_GSUAMBSK_GLS0_ | 1792 | 65.044 |
| attn_fwd | 2688 | 61.019 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x48x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 3878 | 56.811 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 5516 | 55.444 |
| Cijk_Alik_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB1_GSUAMBSK_GLS0_IS | 1792 | 43.367 |
| void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<2> >(at::TensorIteratorBase&, c10::ArrayRef<long | 4480 | 38.940 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x48x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 1837 | 34.911 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI32x32x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 2114 | 34.409 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x48x128_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 2005 | 29.636 |

详细算子、phase和重叠说明见同名JSON；inclusive模块总量包含父子重复，不相加作为总耗时。

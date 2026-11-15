# Profile摘要

仅为诊断trace，不是正式A/B/C性能分数。单位为毫秒。

GPU kernel数：204024；模块归因覆盖：99.9%。

| 项目 | ms |
| --- | --- |
| summed_gpu_kernel_us | 2318.419 |
| kernel_active_union_us | 2318.418 |
| kernel_span_us | 5975.376 |
| kernel_gaps_within_span_us | 3656.958 |

kernel累计时长可重叠；空隙不自动等于CPU瓶颈。

## 按最内层模块归因

| 模块 | 累计kernel ms |
| --- | --- |
| moe_experts | 1792.766 |
| attention_swa | 217.500 |
| qkv | 129.372 |
| attention_full | 75.713 |
| output_projection | 59.270 |
| router | 27.759 |
| norm_post | 7.446 |
| norm_input | 7.375 |
| unattributed | 1.218 |

## 分阶段模块kernel累计ms

| 阶段 | 模块 | 累计ms |
| --- | --- | --- |
| prefill | moe_experts | 1446.523 |
| prefill | attention_swa | 136.246 |
| prefill | qkv | 59.767 |
| prefill | attention_full | 43.956 |
| prefill | output_projection | 33.304 |
| prefill | router | 9.565 |
| prefill | norm_post | 5.355 |
| prefill | norm_input | 5.258 |
| prefill | unattributed | 0.469 |
| unattributed | unattributed | 0.182 |
| decode | moe_experts | 346.243 |
| decode | attention_swa | 81.255 |
| decode | qkv | 69.604 |
| decode | attention_full | 31.757 |
| decode | output_projection | 25.966 |
| decode | router | 18.194 |
| decode | norm_input | 2.117 |
| decode | norm_post | 2.091 |
| decode | unattributed | 0.566 |

## 热点kernel

| 名称 | 次数 | 累计ms |
| --- | --- | --- |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 33348 | 426.228 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 31556 | 403.153 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 17434 | 212.061 |
| void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) | 5461 | 121.096 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 6132 | 82.732 |
| void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::bina | 9635 | 77.258 |
| Cijk_Alik_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x16x128_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB1_GSUAMBSK_GLS0_ | 1792 | 64.122 |
| attn_fwd | 2688 | 61.020 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x48x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 3878 | 56.844 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 5516 | 55.514 |
| Cijk_Alik_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB1_GSUAMBSK_GLS0_IS | 1792 | 43.801 |
| void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<2> >(at::TensorIteratorBase&, c10::ArrayRef<long | 4480 | 38.729 |
| void at::native::sbtopk::gatherTopK<float, unsigned int, 1, false>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned | 1792 | 36.248 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x48x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 1837 | 34.619 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI32x32x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 2114 | 34.404 |

详细算子、phase和重叠说明见同名JSON；inclusive模块总量包含父子重复，不相加作为总耗时。

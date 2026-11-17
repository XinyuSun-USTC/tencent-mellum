# Profile摘要

仅为诊断trace，不是正式A/B/C性能分数。单位为毫秒。

GPU kernel数：614900；模块归因覆盖：99.5%。

| 项目 | ms |
| --- | --- |
| summed_gpu_kernel_us | 12941.785 |
| kernel_active_union_us | 12941.779 |
| kernel_span_us | 22446.053 |
| kernel_gaps_within_span_us | 9504.274 |

kernel累计时长可重叠；空隙不自动等于CPU瓶颈。

## 按最内层模块归因

| 模块 | 累计kernel ms |
| --- | --- |
| sdpa | 4827.915 |
| moe_experts | 4174.472 |
| attention_swa | 1126.888 |
| qkv | 731.961 |
| attention_full | 562.910 |
| gather_reference_kv | 394.947 |
| norm_post | 282.972 |
| output_projection | 281.568 |
| norm_input | 276.312 |
| requestwise_decode | 103.939 |
| router | 64.988 |
| unattributed | 58.908 |
| grouped_swa | 54.004 |

## 分阶段模块kernel累计ms

| 阶段 | 模块 | 累计ms |
| --- | --- | --- |
| prefill | sdpa | 3883.884 |
| prefill | moe_experts | 3408.915 |
| prefill | attention_swa | 756.439 |
| prefill | qkv | 576.971 |
| prefill | attention_full | 439.496 |
| prefill | output_projection | 222.923 |
| prefill | norm_post | 125.770 |
| prefill | norm_input | 125.654 |
| prefill | router | 24.385 |
| prefill | unattributed | 17.562 |
| unattributed | unattributed | 11.259 |
| decode | sdpa | 944.031 |
| decode | moe_experts | 765.557 |
| decode | gather_reference_kv | 394.947 |
| decode | attention_swa | 370.449 |
| decode | norm_post | 157.201 |
| decode | qkv | 154.990 |
| decode | norm_input | 150.658 |
| decode | attention_full | 123.414 |
| decode | requestwise_decode | 103.939 |
| decode | output_projection | 58.646 |
| decode | grouped_swa | 54.004 |
| decode | router | 40.603 |
| decode | unattributed | 30.086 |

## 热点kernel

| 名称 | 次数 | 累计ms |
| --- | --- | --- |
| attn_fwd | 5124 | 4728.799 |
| void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<2> >(at::TensorIteratorBase&, c10::ArrayRef<long | 12090 | 459.528 |
| void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::bina | 43077 | 457.243 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x128x64_MI32x32x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_ | 6009 | 439.647 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_ | 7409 | 406.930 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x224x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_ | 1372 | 399.854 |
| void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()( | 13426 | 396.756 |
| void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) | 10087 | 391.049 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 28645 | 380.700 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 25061 | 321.995 |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::nat | 19549 | 155.140 |
| void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloa | 34081 | 150.032 |
| void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor< | 19549 | 147.764 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 10818 | 137.787 |
| void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<c | 19376 | 125.234 |

详细算子、phase和重叠说明见同名JSON；inclusive模块总量包含父子重复，不相加作为总耗时。

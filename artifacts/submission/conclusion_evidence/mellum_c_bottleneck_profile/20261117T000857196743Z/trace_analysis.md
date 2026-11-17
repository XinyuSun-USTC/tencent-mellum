# Profile摘要

仅为诊断trace，不是正式A/B/C性能分数。单位为毫秒。

GPU kernel数：172608；模块归因覆盖：99.9%。

| 项目 | ms |
| --- | --- |
| summed_gpu_kernel_us | 4773.821 |
| kernel_active_union_us | 4773.816 |
| kernel_span_us | 6937.179 |
| kernel_gaps_within_span_us | 2163.362 |

kernel累计时长可重叠；空隙不自动等于CPU瓶颈。

## 按最内层模块归因

| 模块 | 累计kernel ms |
| --- | --- |
| sdpa | 1707.514 |
| moe_experts | 1366.640 |
| gather_reference_kv | 498.439 |
| grouped_swa | 464.593 |
| qkv | 291.013 |
| attention_swa | 113.445 |
| output_projection | 109.952 |
| attention_full | 95.253 |
| router | 78.496 |
| norm_input | 23.262 |
| norm_post | 19.259 |
| unattributed | 5.953 |

## 分阶段模块kernel累计ms

| 阶段 | 模块 | 累计ms |
| --- | --- | --- |
| prefill | moe_experts | 306.353 |
| prefill | sdpa | 157.621 |
| prefill | attention_full | 74.796 |
| prefill | attention_swa | 50.088 |
| prefill | qkv | 17.680 |
| prefill | output_projection | 6.604 |
| prefill | router | 4.685 |
| prefill | norm_post | 1.482 |
| prefill | norm_input | 1.384 |
| prefill | unattributed | 0.296 |
| unattributed | unattributed | 1.078 |
| decode | sdpa | 1549.893 |
| decode | moe_experts | 1060.287 |
| decode | gather_reference_kv | 498.439 |
| decode | grouped_swa | 464.593 |
| decode | qkv | 273.333 |
| decode | output_projection | 103.348 |
| decode | router | 73.811 |
| decode | attention_swa | 63.357 |
| decode | norm_input | 21.879 |
| decode | attention_full | 20.457 |
| decode | norm_post | 17.777 |
| decode | unattributed | 4.579 |

## 热点kernel

| 名称 | 次数 | 累计ms |
| --- | --- | --- |
| attn_fwd | 3584 | 1680.168 |
| void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<2> >(at::TensorIteratorBase&, c10::ArrayRef<long | 7616 | 558.908 |
| void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) | 4481 | 534.093 |
| Cijk_Alik_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x128_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB1_GSUAMBSK_GLS0_I | 3584 | 435.252 |
| void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()( | 5719 | 348.337 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 18380 | 239.656 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 22848 | 226.076 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 10764 | 133.540 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 7392 | 90.811 |
| void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::Op | 9520 | 55.196 |
| void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<long, long, long, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda() | 5936 | 46.760 |
| void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::bina | 8305 | 45.410 |
| void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<long, long, long, at::native::remainder_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() | 5488 | 40.412 |
| void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const | 3585 | 26.839 |
| void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::o | 5376 | 22.557 |

详细算子、phase和重叠说明见同名JSON；inclusive模块总量包含父子重复，不相加作为总耗时。

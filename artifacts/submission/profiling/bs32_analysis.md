# Profile摘要

仅为诊断trace，不是正式A/B/C性能分数。单位为毫秒。

GPU kernel数：20911；模块归因覆盖：99.7%。

| 项目 | ms |
| --- | --- |
| summed_gpu_kernel_us | 1573.962 |
| kernel_active_union_us | 1573.961 |
| kernel_span_us | 2027.532 |
| kernel_gaps_within_span_us | 453.571 |

kernel累计时长可重叠；空隙不自动等于CPU瓶颈。

## 按最内层模块归因

| 模块 | 累计kernel ms |
| --- | --- |
| moe_experts | 714.275 |
| qkv | 331.458 |
| attention_swa | 258.752 |
| output_projection | 94.234 |
| attention_full | 84.897 |
| norm_input | 40.254 |
| norm_post | 39.979 |
| router | 5.444 |
| unattributed | 4.668 |

## 分阶段模块kernel累计ms

| 阶段 | 模块 | 累计ms |
| --- | --- | --- |
| prefill | moe_experts | 668.953 |
| prefill | qkv | 319.952 |
| prefill | attention_swa | 222.134 |
| prefill | output_projection | 90.022 |
| prefill | attention_full | 73.610 |
| prefill | norm_post | 32.073 |
| prefill | norm_input | 32.016 |
| prefill | router | 2.926 |
| prefill | unattributed | 1.786 |
| unattributed | unattributed | 0.615 |
| decode | moe_experts | 45.322 |
| decode | attention_swa | 36.617 |
| decode | qkv | 11.506 |
| decode | attention_full | 11.287 |
| decode | norm_input | 8.239 |
| decode | norm_post | 7.905 |
| decode | output_projection | 4.212 |
| decode | router | 2.518 |
| decode | unattributed | 2.267 |

## 热点kernel

| 名称 | 次数 | 累计ms |
| --- | --- | --- |
| void ck::kernel_moe_gemm_2lds<ck::GridwiseMoeGemm<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, ck::Tuple<ck::tensor_layout::gemm::RowMajor, ck::tensor_l | 280 | 427.666 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x224x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_ | 168 | 302.329 |
| void ck::kernel_moe_gemm_2lds<ck::GridwiseMoeGemm<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, ck::Tuple<ck::tensor_layout::gemm::RowMajor, ck::tensor_l | 280 | 232.891 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x192x64_MI32x32x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_ | 56 | 90.022 |
| void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::bina | 2250 | 83.621 |
| std::enable_if<!(kattr_no_packed_fp32_ops_v<ck_tile::gfx9_t>), void>::type ck_tile::kentry<ck_tile::gfx9_t, 2, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile: | 42 | 76.647 |
| void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor< | 1130 | 35.020 |
| void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 3, 64, 64>(at::native::(anonymous namespace)::OpaqueTyp | 560 | 34.878 |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::nat | 1130 | 32.833 |
| void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned | 280 | 32.742 |
| aiter::fmha_fwd_hd128_bf16_causal_rtna_group | 14 | 24.699 |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, | 1130 | 20.484 |
| void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloa | 1970 | 19.701 |
| void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::Op | 280 | 19.134 |
| void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<c | 1120 | 17.301 |

详细算子、phase和重叠说明见同名JSON；inclusive模块总量包含父子重复，不相加作为总耗时。

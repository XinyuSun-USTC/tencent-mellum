# Profile摘要

仅为诊断trace，不是正式A/B/C性能分数。单位为毫秒。

GPU kernel数：18627；模块归因覆盖：98.3%。

| 项目 | ms |
| --- | --- |
| summed_gpu_kernel_us | 170.570 |
| kernel_active_union_us | 170.569 |
| kernel_span_us | 641.084 |
| kernel_gaps_within_span_us | 470.515 |

kernel累计时长可重叠；空隙不自动等于CPU瓶颈。

## 按最内层模块归因

| 模块 | 累计kernel ms |
| --- | --- |
| moe_experts | 58.372 |
| attention_swa | 41.228 |
| qkv | 20.007 |
| attention_full | 12.814 |
| norm_post | 12.746 |
| norm_input | 12.437 |
| output_projection | 7.300 |
| unattributed | 2.937 |
| router | 2.729 |

## 分阶段模块kernel累计ms

| 阶段 | 模块 | 累计ms |
| --- | --- | --- |
| prefill | moe_experts | 34.771 |
| prefill | attention_swa | 11.393 |
| prefill | qkv | 10.571 |
| prefill | output_projection | 3.851 |
| prefill | attention_full | 3.794 |
| prefill | norm_input | 2.217 |
| prefill | norm_post | 2.098 |
| prefill | router | 0.382 |
| prefill | unattributed | 0.312 |
| unattributed | unattributed | 0.564 |
| decode | attention_swa | 29.835 |
| decode | moe_experts | 23.601 |
| decode | norm_post | 10.648 |
| decode | norm_input | 10.221 |
| decode | qkv | 9.436 |
| decode | attention_full | 9.019 |
| decode | output_projection | 3.449 |
| decode | router | 2.348 |
| decode | unattributed | 2.060 |

## 热点kernel

| 名称 | 次数 | 累计ms |
| --- | --- | --- |
| void ck::kernel_moe_gemm_2lds<ck::GridwiseMoeGemm<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, ck::Tuple<ck::tensor_layout::gemm::RowMajor, ck::tensor_l | 28 | 20.501 |
| void ck::kernel_moe_gemm_2lds<ck::GridwiseMoeGemm<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, ck::Tuple<ck::tensor_layout::gemm::RowMajor, ck::tensor_l | 28 | 10.958 |
| void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::bina | 1569 | 9.330 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x224x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_ | 28 | 8.149 |
| Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_I | 672 | 7.182 |
| void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloa | 1773 | 6.822 |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::nat | 1017 | 6.623 |
| void ck::kernel_moe_gemm<ck::GridwiseMoeGemm<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, ck::Tuple<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout | 224 | 6.602 |
| void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int,  | 1465 | 6.222 |
| void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<c | 1008 | 5.886 |
| paged_attention_decode_sliding_window | 168 | 5.695 |
| void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor< | 1017 | 5.135 |
| void at::native::sbtopk::gatherTopK<float, unsigned int, 1, false>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned | 224 | 4.673 |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, | 1017 | 4.279 |
| void at::native::vectorized_elementwise_kernel<4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambd | 1017 | 4.254 |

详细算子、phase和重叠说明见同名JSON；inclusive模块总量包含父子重复，不相加作为总耗时。

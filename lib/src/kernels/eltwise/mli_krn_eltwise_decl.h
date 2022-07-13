/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_ELTWISE_ADD_DECL_REF_H_
#define _MLI_KRN_ELTWISE_ADD_DECL_REF_H_

#include "mli_config.h"
#include "mli_mem_info.h"
#include "mli_types.h"
#include "mli_prv_quant.h"

namespace mli {
typedef enum {
    ELTWISE_ADD = 0,
    ELTWISE_SUB,
    ELTWISE_MUL,
    ELTWISE_MAX,
    ELTWISE_MIN
} mli_eltwise_type;

namespace krn {
////////////////////////////////////////////////////////////////////////////////
// Functions (in *_ref/*_dsp/*vdsp) that can be called from outside their own
// file must be declared here. This includes all overloads. For example, if we
// have: o_T f(i_T a) and int8_t f(int8_t a), then both must be declared.
// Not doing so, can cause the compiler to use the wrong overload.
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// REF
////////////////////////////////////////////////////////////////////////////////
namespace ref {

template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert = false , bool no_scalar = false, bool no_out_update = false, bool shape_1d = false>
void eltwise_prepare_and_run(
        const mli_tensor *__restrict in1,
        const mli_tensor *__restrict in2,
        mli_tensor *__restrict out);

template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert = false>
void eltwise_op_basic(
        const generic_tensor_private_t<MLI_PTR(i_T)> * __restrict in1,
        const generic_tensor_private_t<MLI_PTR(i_T)> * __restrict in2,
        generic_tensor_private_t<MLI_OUT_PTR(o_T)> * __restrict out,
        const i_T op1_s,
        const i_T op2_s,
        const bool scalar_op1,
        const bool scalar_op2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift,
        const int scale16_1,
        const int scale16_2,
        const int in_offset1,
        const int in_offset2,
        const int out_offset);

template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert = false>
void eltwise_op_basic(
        const mli_tensor * __restrict in1,
        const mli_tensor * __restrict in2,
        mli_tensor * __restrict out,
        const int *shape,
        const i_T op1_s,
        const i_T op2_s,
        const bool scalar_op1,
        const bool scalar_op2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift,
        const int scale16_1,
        const int scale16_2,
        const int in_offset1,
        const int in_offset2,
        const int out_offset);

template <typename in_T, typename out_T, mli_eltwise_type func_type, bool convert>
out_T eltwise_perform_operation(
        const in_T op1,
        const in_T op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift);

template <>
MLI_FORCE_INLINE int8_t eltwise_perform_operation <int8_t, int8_t, ELTWISE_MUL, true>(
        const int8_t op1,
        const int8_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift);

template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert>
void eltwise_innerloop(
        const MLI_PTR(i_T) __restrict op1_ptr,
        const MLI_PTR(i_T) __restrict op2_ptr,
        MLI_PTR(o_T) __restrict out_ptr,
        int idx1,
        int idx2,
        int idx_out,
        const int count,
        const i_T op1_s,
        const i_T op2_s,
        const bool scalar_op1,
        const bool scalar_op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale1,
        const int16_t scale2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift);
} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {
template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert = false>
static MLI_FORCE_INLINE void eltwise_prepare_and_run(
        const mli_tensor * in1,
        const mli_tensor * in2,
        mli_tensor * out);

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {
template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert = false>
void eltwise_prepare_and_run(
        const mli_tensor * in1,
        const mli_tensor * in2,
        mli_tensor * out);

template <typename in_T, typename out_T, mli_eltwise_type func_type, bool convert>
out_T eltwise_perform_operation(
        const in_T op1,
        const in_T op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift);

#if defined(__Xvec_width)
    template <>
    MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_ADD, false>(
            const vNx2short_t op1,
            const vNx2short_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4int_t eltwise_perform_operation<vNx4int_t, vNx4int_t, ELTWISE_ADD, false>(
            const vNx4int_t op1,
            const vNx4int_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_ADD, false>(
            const vNx4char_t op1,
            const vNx4char_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_ADD, true>(
            const vNx4char_t op1,
            const vNx4char_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNint_t eltwise_perform_operation<vNint_t, vNint_t, ELTWISE_ADD, true>(
            const vNint_t op1,
            const vNint_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_MUL, false>(
            const vNx2short_t op1,
            const vNx2short_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MUL, false>(
            const vNx4char_t op1,
            const vNx4char_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MUL, true>(
            const vNx4char_t op1,
            const vNx4char_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4int_t eltwise_perform_operation<vNx4char_t, vNx4int_t, ELTWISE_MUL, false>(
            const vNx4char_t op1,
            const vNx4char_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx2int_t eltwise_perform_operation<vNx2short_t, vNx2int_t, ELTWISE_MUL, false>(
            const vNx2short_t op1,
            const vNx2short_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_SUB, false>(
            const vNx2short_t op1,
            const vNx2short_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_SUB, false>(
            const vNx4char_t op1,
            const vNx4char_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_SUB, true>(
            const vNx4char_t op1,
            const vNx4char_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNint_t eltwise_perform_operation<vNint_t, vNint_t, ELTWISE_SUB, true>(
            const vNint_t op1,
            const vNint_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_MAX, false>(
            const vNx2short_t op1,
            const vNx2short_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MAX, false>(
            const vNx4char_t op1,
            const vNx4char_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MAX, true>(
            const vNx4char_t op1,
            const vNx4char_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNint_t eltwise_perform_operation<vNint_t, vNint_t, ELTWISE_MAX, true>(
            const vNint_t op1,
            const vNint_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_MIN, false>(
            const vNx2short_t op1,
            const vNx2short_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MIN, false>(
            const vNx4char_t op1,
            const vNx4char_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MIN, true>(
            const vNx4char_t op1,
            const vNx4char_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);

    template <>
    MLI_FORCE_INLINE vNint_t eltwise_perform_operation<vNint_t, vNint_t, ELTWISE_MAX, true>(
            const vNint_t op1,
            const vNint_t op2,
            const int16_t in_offset1,
            const int16_t in_offset2,
            const int16_t out_offset,
            const int16_t scale_factor1,
            const int16_t scale_factor2,
            const int pre_op_shift1,
            const int pre_op_shift2,
            const int post_op_shift);
#endif

template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert>
MLI_FORCE_INLINE void eltwise_innerloop(
        const MLI_PTR(i_T) __restrict op1_ptr,
        const MLI_PTR(i_T) __restrict op2_ptr,
        MLI_PTR(o_T) __restrict out_ptr,
        int idx1,
        int idx2,
        int idx_out,
        const int count,
        const i_T op1_s,
        const i_T op2_s,
        const bool scalar_op1,
        const bool scalar_op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale1,
        const int16_t scale2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift);
} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_ELTWISE_ADD_DECL_REF_H_

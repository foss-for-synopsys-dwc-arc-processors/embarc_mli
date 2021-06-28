/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_L2_NORMALIZE_DECL_H_
#define _MLI_KRN_L2_NORMALIZE_DECL_H_

#include "mli_config.h"
#include "mli_types.h"
#include "mli_prv_tensor.h"

namespace mli {
namespace krn {
////////////////////////////////////////////////////////////////////////////////
// Functions (in *_ref/*_dsp/*vdsp) that can be called from outside their own
// file must be declared here. This includes all overloads. For example, if we
// have: io_T f(io_T a) and int8_t f(int8_t a), then both must be declared.
// Not doing so, can cause the compiler to use the wrong overload.
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// REF
////////////////////////////////////////////////////////////////////////////////
namespace ref {

template<typename io_T, bool convert, bool one_dim_with_mem_stride = false>
static MLI_FORCE_INLINE int16_t compute_normalized_sum_square_one_dim(
        const MLI_PTR(io_T) vec_in,
        int16_t in_zp,
        int *norm_shift,
        const int one_dim_shape,
        const int one_dim_mem_stride = 1);

template<typename io_T, bool convert>
static MLI_FORCE_INLINE int16_t compute_normalized_sum_square(
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        const MLI_PTR(io_T) vec_in,
        int16_t in_zp,
        int *norm_shift);

template<typename io_T, bool convert, bool one_dim_with_mem_stride = false>
static MLI_FORCE_INLINE void normalize_tensor_one_dim(
        const MLI_PTR(io_T) vec_in,
        MLI_PTR(io_T) vec_out,
        int16_t scale,
        int16_t in_zp,
        int shift, 
        const int one_dim_shape,
        const int one_dim_in_mem_stride = 1,
        const int one_dim_out_mem_stride = 1);

template<typename io_T, bool convert>
static MLI_FORCE_INLINE void normalize_tensor(
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out_prv,
        const MLI_PTR(io_T) vec_in,
        MLI_PTR(io_T) vec_out,
        int16_t scale,
        int16_t in_zp,
        int shift);

template <typename io_T, bool convert = false>
static MLI_FORCE_INLINE mli_status mli_krn_l2_normalize_run(const mli_tensor *in, 
        const mli_tensor *epsilon, 
        const mli_l2_normalize_cfg *cfg, 
        mli_tensor *out,
        const mli_lut *lut);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

template<typename io_T, bool convert, bool one_dim_with_mem_stride = false>
static MLI_FORCE_INLINE int16_t compute_normalized_sum_square_one_dim(
        const MLI_PTR(io_T) vec_in,
        int16_t in_zp,
        int *norm_shift,
        const int one_dim_shape,
        const int one_dim_mem_stride = 1);

template<>
MLI_FORCE_INLINE int16_t compute_normalized_sum_square_one_dim<int16_t, false, false>(
        const MLI_PTR(int16_t) vec_in,
        int16_t in_zp,
        int *norm_shift,
        const int one_dim_shape,
        const int one_dim_mem_stride);

template<>
MLI_FORCE_INLINE int16_t compute_normalized_sum_square_one_dim<int16_t, false, true>(
        const MLI_PTR(int16_t) vec_in,
        int16_t in_zp,
        int *norm_shift,
        const int one_dim_shape,
        const int one_dim_mem_stride);

template<typename io_T, bool convert>
static MLI_FORCE_INLINE int16_t compute_normalized_sum_square(
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        const MLI_PTR(io_T) vec_in,
        int16_t in_zp,
        int *norm_shift);

template<>
MLI_FORCE_INLINE int16_t compute_normalized_sum_square<int16_t, false>(
        struct generic_tensor_private_t<MLI_PTR(int16_t)> *in_prv,
        const MLI_PTR(int16_t) vec_in,
        int16_t in_zp,
        int *norm_shift);

template<typename io_T, bool convert, bool one_dim_with_mem_stride = false>
static MLI_FORCE_INLINE void normalize_tensor_one_dim(
        const MLI_PTR(io_T) vec_in,
        MLI_PTR(io_T) vec_out,
        int16_t scale,
        int16_t in_zp,
        int shift, 
        const int one_dim_shape,
        const int one_dim_in_mem_stride = 1,
        const int one_dim_out_mem_stride = 1);

template<typename io_T, bool convert>
static MLI_FORCE_INLINE void normalize_tensor(
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out_prv,
        const MLI_PTR(io_T) vec_in,
        MLI_PTR(io_T) vec_out,
        int16_t scale,
        int16_t in_zp,
        int shift);

} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_L2_NORMALIZE_DECL_H_

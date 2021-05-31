/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PRELU_DECL_H_
#define _MLI_KRN_PRELU_DECL_H_

#include "mli_config.h"
#include "mli_types.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"

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

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift);

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift,
        const int remaining_part);

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params);

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params,
        const int remaining_part);

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu_no_broadcast(
        const MLI_PTR(io_T) __restrict vec_in,
        MLI_OUT_PTR(io_T) __restrict vec_out,
        const scale_T scale_v,
        const int shift,
        const generic_tensor_private_t<MLI_PTR(io_T)> in_prv,
        const generic_tensor_private_t<MLI_OUT_PTR(io_T)> out_prv,
        const int remaining_part = 0);

static MLI_FORCE_INLINE s8asym_quant_params prelu_define_requant_params(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out,
        const int8_t alpha_sa8,
        const s8asym_quant_params *identity_params);

template <typename io_T>
static MLI_FORCE_INLINE mli_status prelu_fx_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg, 
        mli_tensor *out);
        
static MLI_FORCE_INLINE mli_status prelu_sa8_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg, 
        mli_tensor *out);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift);

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift,
        const int remaining_part);

#if !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
static MLI_FORCE_INLINE s8asym_quant_params_v prelu_define_requant_params(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out,
        const v2q15_t alpha_sa8,
        const s8asym_quant_params *identity_params);

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params);

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params,
        const int remaining_part);
#endif

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift);

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift,
        const int remaining_part);

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        const vNx4char_t scale,
        MLI_OUT_PTR(int8_t) vec_out,
        const int shift);

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        const vNx4char_t scale,
        MLI_OUT_PTR(int8_t) vec_out,
        const int shift,
        const int remaining_part);

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int16_t) vec_in,
        const vNx2short_t scale,
        MLI_OUT_PTR(int16_t) vec_out,
        const int shift);

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int16_t) vec_in,
        const vNx2short_t scale,
        MLI_OUT_PTR(int16_t) vec_out,
        const int shift,
        const int remaining_part);

static MLI_FORCE_INLINE s8asym_quant_params_v prelu_define_requant_params(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out,
        vNx4char_t alpha_sa8,
        const s8asym_quant_params *identity_params);

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params);

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params,
        const int remaining_part);
#endif

} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RELU_DECL_H_

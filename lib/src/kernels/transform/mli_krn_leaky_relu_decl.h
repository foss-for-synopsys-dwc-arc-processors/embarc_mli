/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_LEAKY_RELU_DECL_H_
#define _MLI_KRN_LEAKY_RELU_DECL_H_

#include "mli_config.h"
#include "mli_mem_info.h"
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

template <typename io_T>
static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift);

template <typename io_T>
static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift,
        const int remaining_part);

static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params);

static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params,
        const int remaining_part);

template<typename io_T>
static MLI_FORCE_INLINE void compute_leaky_relu_fx_inner_loop(
        const MLI_PTR(io_T) __restrict vec_in,
        MLI_OUT_PTR(io_T) __restrict vec_out,
        const io_T scale,
        const int shift,
        const int count,
        const int remaining_part);

static MLI_FORCE_INLINE void compute_leaky_relu_sa8_inner_loop(
        const MLI_PTR(int8_t) __restrict vec_in,
        MLI_OUT_PTR(int8_t) __restrict vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params,
        const int count,
        const int remaining_part);

template <typename io_T>
static MLI_FORCE_INLINE mli_status leaky_relu_fx_run(const mli_tensor *in,
        const mli_tensor *slope_coeff,
        mli_tensor *out);

static MLI_FORCE_INLINE mli_status leaky_relu_sa8_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {

template <typename io_T>
static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift);

template <typename io_T>
static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift,
        const int remaining_part);

static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params);

static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params,
        const int remaining_part);

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

template<typename io_T>
MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift);

template<typename io_T>
MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift,
        const int remaining_part);

template<typename io_T>
static MLI_FORCE_INLINE void compute_leaky_relu_fx_inner_loop(
        const MLI_PTR(io_T) __restrict vec_in,
        MLI_OUT_PTR(io_T) __restrict vec_out,
        const io_T scale,
        const int shift,
        const int count,
        const int remaining_part);

static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params);

static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params,
        const int remaining_part);

static MLI_FORCE_INLINE void compute_leaky_relu_sa8_inner_loop(
        const MLI_PTR(int8_t) __restrict vec_in,
        MLI_OUT_PTR(int8_t) __restrict vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params,
        const int count,
        const int remaining_part);

} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_LEAKY_RELU_DECL_H_

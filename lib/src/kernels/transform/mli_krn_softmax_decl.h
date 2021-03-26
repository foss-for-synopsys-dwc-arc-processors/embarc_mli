/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_SOFTMAX_DECL_H_
#define _MLI_KRN_SOFTMAX_DECL_H_

#include "mli_config.h"
#include "mli_types.h"
#include "mli_prv_tensor.h"
#include "mli_math.h"
#include "mli_prv_quant_decl.h"

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

template <typename io_T, bool is_asym>
static MLI_FORCE_INLINE mli_status mli_krn_softmax_run(const mli_tensor *in, const mli_softmax_cfg *cfg,
        mli_tensor *out, const mli_lut *lut);

template <typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_fx_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        int in_frac, int frac_bits, const mli_lut *lut);

template<typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        s8asym_quant_params in_params, s8asym_quant_params out_params, const mli_lut* lut);

template<>
MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(int16_t) vec_in, MLI_PTR(int16_t) vec_out, 
        generic_tensor_private_t<MLI_PTR(int16_t)> in_prv, generic_tensor_private_t<MLI_PTR(int16_t)> out_prv,
        s8asym_quant_params in_params, s8asym_quant_params out_params, const mli_lut *lut);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {

template <typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_fx_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        int in_frac, int frac_bits, const mli_lut *lut);

template<typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        s8asym_quant_params in_params, s8asym_quant_params out_params, const mli_lut *lut);

template<>
MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(int16_t) vec_in, MLI_PTR(int16_t) vec_out, 
        generic_tensor_private_t<MLI_PTR(int16_t)> in_prv, generic_tensor_private_t<MLI_PTR(int16_t)> out_prv,
        s8asym_quant_params in_params, s8asym_quant_params out_params, const mli_lut *lut);

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

template <typename io_T, bool is_asym>
static MLI_FORCE_INLINE mli_status mli_krn_softmax_run(const mli_tensor *in, const mli_softmax_cfg *cfg,
        mli_tensor *out, const mli_lut *lut);

template <typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_fx_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        int in_frac, int* in_step, int* out_step, int frac_bits, const mli_lut *lut);

template<typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        int* in_step, int* out_step, s8asym_quant_params in_params, s8asym_quant_params out_params, const mli_lut *lut);
template<>
MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(int16_t) vec_in, MLI_PTR(int16_t) vec_out, 
        generic_tensor_private_t<MLI_PTR(int16_t)> in_prv, generic_tensor_private_t<MLI_PTR(int16_t)> out_prv,
        int* in_step, int* out_step, s8asym_quant_params in_params, s8asym_quant_params out_params, const mli_lut *lut);

} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_SOFTMAX_DECL_H_

/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_LUT_DECL_H_
#define _MLI_PRV_LUT_DECL_H_

#include "mli_config.h"
#include "mli_types.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"

//TODO: Remove extra
const int kTmpBufSize = 32;
const int kTransfFuncIntBits = 0;
const int kMaxFracBitsFx16 = (sizeof(int16_t) * 8) - 1;
const int kMaxFracBitsFx8 = (sizeof(int8_t) * 8) - 1;   //


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
template <typename io_T, bool convert = false>
static MLI_FORCE_INLINE void activation_lut(
        const struct generic_tensor_private_t<MLI_PTR(io_T)> *in,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params  = nullptr,
        struct s8asym_quant_params *out_params = nullptr);

template <typename io_T, bool convert = false>
static MLI_FORCE_INLINE void activation_lut(
        const mli_tensor *in,
        const mli_tensor *out,
        const mli_lut *lut,
        int in_frac_bits,
        struct s8asym_quant_params *in_params = nullptr,
        struct s8asym_quant_params *out_params = nullptr);

template <typename io_T, bool convert>
static MLI_FORCE_INLINE void compute_activation_lut(
        const struct generic_tensor_private_t<MLI_PTR(io_T)> *in,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params);

template <typename in_T, typename out_T, bool convert_input, bool convert_output>
static MLI_FORCE_INLINE out_T activation_lut_one_elem_interpolate(
        const in_T in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params = nullptr,
        struct s8asym_quant_params *out_params = nullptr);

template <typename in_T, typename out_T, bool convert_input, bool convert_output>
static MLI_FORCE_INLINE out_T activation_lut_one_elem_no_interpolate(
        const in_T in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params = nullptr,
        struct s8asym_quant_params *out_params = nullptr);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {

template <typename io_T, bool convert>
static MLI_FORCE_INLINE void compute_activation_lut(
        const struct generic_tensor_private_t<MLI_PTR(io_T)> *in,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params);

#if !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
template <typename out_T, bool convert_input, bool convert_output>
static MLI_FORCE_INLINE v2q15_t activation_lut_two_elem_interpolate(
        const v2q15_t in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params = nullptr,
        struct s8asym_quant_params *out_params = nullptr);

template <typename out_T, bool convert_input, bool convert_output>
static MLI_FORCE_INLINE v2q15_t activation_lut_two_elem_no_interpolate(
        const v2q15_t in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params = nullptr,
        struct s8asym_quant_params *out_params = nullptr);
#endif

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

template <typename io_T, bool convert>
static MLI_FORCE_INLINE void compute_activation_lut(
        const struct generic_tensor_private_t<MLI_PTR(io_T)> *in,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params);

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
template <bool convert>
static MLI_FORCE_INLINE vNx4short_t activation_lut_vec_elem_interpolate(
        vNx4short_t in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params);

template <bool convert>
static MLI_FORCE_INLINE vNx4short_t activation_lut_vec_elem_no_interpolate(
        vNx4short_t in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params);
#endif

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_PRV_LUT_DECL_H_

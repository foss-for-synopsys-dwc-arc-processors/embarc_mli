/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_QUANT_DECL_H_
#define _MLI_PRV_QUANT_DECL_H_

#include "mli_config.h"

//namespace mli {
//namespace krn {

typedef enum {
    FX_MATH = 0,
    S8ASYM_MATH
} mli_math_type;


/**
 * @brief Quantization specific parameter to perform correct calculations in s8asym quantization scheme.
 */
struct s8asym_quant_specific_params {
    int16_t in_offset;
    int16_t out_offset;
    int16_t weights_offset;

    const int32_t *weight_scales;
    int32_t in_to_out_scales_ratio;

    int32_t out_mul;
    int out_shift;
};

/**
 * @brief Quantization specific parameter to perform correct calculations in MLI_FX quantization scheme.
 */
struct fx_quant_specific_params {
    int bias_shift;
    int out_shift;
};

typedef union _conv_math_params {
    struct fx_quant_specific_params fx;

    struct s8asym_quant_specific_params i8asym;
} conv_math_params;

//} // namespace krn
//} // namespace mli

namespace mli {
namespace krn {

////////////////////////////////////////////////////////////////////////////////
// REF
////////////////////////////////////////////////////////////////////////////////
namespace ref {

template <typename quant_T>
MLI_FORCE_INLINE void define_quant_params(const mli_tensor* in, const mli_tensor* weights,
        const mli_tensor* bias, const mli_tensor* out, quant_T* params);
template <>
MLI_FORCE_INLINE void define_quant_params(const mli_tensor* in, const mli_tensor* weights,
        const mli_tensor* bias, const mli_tensor* out, fx_quant_specific_params* params);
template <>
MLI_FORCE_INLINE void define_quant_params(const mli_tensor* in, const mli_tensor* weights,
        const mli_tensor* bias, const mli_tensor* out, s8asym_quant_specific_params* params);

template <typename quant_T>
MLI_FORCE_INLINE void adjust_quant_params(quant_T* params, int krn_idx = 0);
template <>
MLI_FORCE_INLINE void adjust_quant_params(s8asym_quant_specific_params* params, int krn_idx);
template <>
MLI_FORCE_INLINE void adjust_quant_params(fx_quant_specific_params* in, int krn_idx);

static MLI_FORCE_INLINE int32_t mli_prv_calc_out_mul(const mli_tensor *in0, const mli_tensor *in1,
        const mli_tensor* out, int* shift);

template <typename w_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T weights_additive(const w_T* __restrict weights,
        acc_T init_accum, const quant_T* quant_params,
        const int width, const int height = 1, int col_step = 1, int row_step = 1);
template <>
MLI_FORCE_INLINE mli_acc32_t weights_additive(const MLI_PTR(int8_t) __restrict weights,
        mli_acc32_t init_accum, const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step);

template <typename in_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T in_additive(const MLI_PTR(in_T) __restrict,
        acc_T init_accum, const quant_T* quant_params,
        const int, const int, int, int);
template <>
MLI_FORCE_INLINE mli_acc32_t in_additive(const MLI_PTR(int8_t) __restrict in,
        mli_acc32_t init_accum, const s8asym_quant_specific_params* quant_params,
        const int width, const int height, int col_step, int row_step);

template <typename in_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T in_additive(const MLI_PTR(in_T) __restrict,
        acc_T init_accum, const quant_T* quant_params,
        const int, const int, const int, int, int, int);
template <>
MLI_FORCE_INLINE mli_acc32_t in_additive(const MLI_PTR(int8_t) __restrict in,
        mli_acc32_t init_accum, const s8asym_quant_specific_params* quant_params,
        const int width, const int height, const int ch, int col_step, int row_step, int ch_step);

template <typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T zp_additive(const quant_T*,
        acc_T init_accum, const int);
template <>
MLI_FORCE_INLINE mli_acc32_t zp_additive(const s8asym_quant_specific_params* quant_params,
        mli_acc32_t init_accum, const int mac_serias_len);

template <typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T bias_additive(const b_T bias, acc_T init_accum,
        const quant_T* quant_params);
template <>
MLI_FORCE_INLINE mli_acc32_t bias_additive(const int8_t bias, mli_acc32_t init_accum,
        const fx_quant_specific_params* quant_params);
template <>
MLI_FORCE_INLINE mli_acc40_t bias_additive(const int16_t bias, mli_acc40_t init_accum,
        const fx_quant_specific_params* quant_params);
template <>
MLI_FORCE_INLINE mli_acc32_t bias_additive(const int32_t bias, mli_acc32_t init_accum,
        const s8asym_quant_specific_params* quant_params);

template <typename o_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE o_T result_cast(const acc_T acc, const quant_T* quant_params);
template <>
MLI_FORCE_INLINE int16_t result_cast(const mli_acc40_t acc, const fx_quant_specific_params* math_params);
template <>
MLI_FORCE_INLINE int16_t result_cast(const mli_acc32_t acc, const fx_quant_specific_params* math_params);
template <>
MLI_FORCE_INLINE int8_t result_cast(const mli_acc32_t acc, const fx_quant_specific_params* math_params);
template <>
MLI_FORCE_INLINE int8_t result_cast(const mli_acc32_t acc, const s8asym_quant_specific_params* quant_params);

template <typename io_T, typename acc_T, typename b_T, mli_math_type math_type>
MLI_FORCE_INLINE io_T result_cast(const acc_T acc, const b_T bias, const int32_t out_mul, const conv_math_params* math_params);
template <>
MLI_FORCE_INLINE int8_t result_cast<int8_t, mli_acc32_t, int32_t, S8ASYM_MATH>(
        const mli_acc32_t acc, const int32_t bias, const int32_t out_mul, const conv_math_params* math_params);

template <typename o_T, typename acc_T, typename quant_T>
static MLI_FORCE_INLINE void result_cast_relu_store(
        MLI_PTR(o_T) __restrict o_ptr,
        acc_T acc,
        const quant_T* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit);

template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_sa8_fx16(
        const in_T in, const int16_t zero_point, const int scale);

template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_fx16_sa8(
        const in_T in, const int16_t zero_point, const int scale);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {
template <typename quant_T>
MLI_FORCE_INLINE void adjust_quant_params(quant_T* params, int krn_idx);
template <>
MLI_FORCE_INLINE void adjust_quant_params(s8asym_quant_specific_params* params, int krn_idx);
template <>
MLI_FORCE_INLINE void adjust_quant_params(fx_quant_specific_params* in, int krn_idx);

template <typename acc_T>
MLI_FORCE_INLINE acc_T weights_additive_v(
        const MLI_PTR(int8_t) __restrict *weights, acc_T *init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step);

template <typename acc_T>
MLI_FORCE_INLINE acc_T weights_additive_v(
        const MLI_PTR(int8_t) __restrict *weights, acc_T *init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step);

template <typename acc_T>
MLI_FORCE_INLINE acc_T weights_additive_v(
        const MLI_PTR(int8_t) __restrict weights, acc_T *init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step);

MLI_FORCE_INLINE mli_acc32_t weights_additive_d(
        const MLI_PTR(int8_t) __restrict weights, mli_acc32_t *init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step);

template <typename o_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void result_cast_relu_store(
        MLI_PTR(o_T) __restrict o_ptr,
        acc_T acc,
        const quant_T* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit);

template <>
MLI_FORCE_INLINE void result_cast_relu_store(
        MLI_PTR(int8_t) __restrict o_ptr,
        int32_t conv_out,
        const s8asym_quant_specific_params* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit);

#if defined(__FXAPI__)
MLI_FORCE_INLINE void result_cast_relu_store_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        __v2i32_t *conv_out_v,
        const s8asym_quant_specific_params quant_params[],
        const int16_t val_min_limit,
        const int16_t val_max_limit);

MLI_FORCE_INLINE void result_cast_relu_store_inp_width_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        __v2i32_t *conv_out_v,
        const s8asym_quant_specific_params *quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int next_out_indx);
#endif

template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_sa8_fx16(
        const in_T in, const int16_t zero_point, const int scale);

template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_fx16_sa8(
        const in_T in, const int16_t zero_point, const int scale);
} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {
template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_sa8_fx16(
        const in_T in, const int16_t zero_point, const int scale);
template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_fx16_sa8(
        const in_T in, const int16_t zero_point, const int scale);

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_PRV_QUANT_DECL_H_
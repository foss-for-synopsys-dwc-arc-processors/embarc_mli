/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_QUANT_DECL_H_
#define _MLI_PRV_QUANT_DECL_H_

#include "mli_config.h"
#include "mli_types.h"

namespace mli {
namespace krn {

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

    const int16_t *weight_scales;
    const int8_t *weight_shifts;
    int weight_dim;
    int16_t in_to_out_scales_ratio;
    int32_t in_to_out_shift;

    int32_t out_mul;
    int out_shift;
};

struct s8asym_quant_params {
    int16_t offset;
    int16_t shift;
    int16_t scale;
};
#if defined(__Xvec_width)
struct s8asym_quant_params_v {
    vNx4short_t offset;
    vNx4short_t shift;
    vNx4short_t scale;
};
struct s8asym_quant_specific_out_params_v {
    int16_t out_offset;
    vNx4short_t out_mul;
    vNx4short_t out_shift;
};

#elif defined(__FXAPI__) 
struct s8asym_quant_params_v {
    v2i16_t offset;
    v2i16_t shift;
    v2i16_t scale;
};
#endif
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


////////////////////////////////////////////////////////////////////////////////
// REF
////////////////////////////////////////////////////////////////////////////////
namespace ref {

MLI_FORCE_INLINE void define_requant_params(const mli_tensor *in, const mli_tensor *out,
    s8asym_quant_params *params, const uint32_t index = 0);
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

MLI_FORCE_INLINE int16_t quant_params_get_weigths_zeropoint(s8asym_quant_specific_params* params);

MLI_FORCE_INLINE int16_t quant_params_get_weigths_zeropoint(fx_quant_specific_params* params);

MLI_FORCE_INLINE int16_t quant_params_set_in_zeropoint(s8asym_quant_specific_params* params, int16_t new_zp);
MLI_FORCE_INLINE int16_t quant_params_set_in_zeropoint(fx_quant_specific_params* params, int16_t new_zp);

template <typename w_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T weights_additive(const MLI_PTR(w_T) __restrict weights,
        acc_T init_accum, const quant_T* quant_params,
        const int width, const int height = 1, int col_step = 1, int row_step = 1);
template <>
MLI_FORCE_INLINE mli_acc32_t weights_additive(const MLI_PTR(int8_t) __restrict weights,
        mli_acc32_t init_accum, const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step);

template <typename w_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T weights_additive(const MLI_PTR(w_T) __restrict weights, acc_T init_accum,
        const quant_T* quant_params,
        const int width, const int height, const int ch, int col_step, int row_step, int ch_step);

template <>
MLI_FORCE_INLINE mli_acc32_t weights_additive(
        const MLI_PTR(int8_t) __restrict weights, mli_acc32_t init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, const int ch, int col_step, int row_step, int ch_step);

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
MLI_FORCE_INLINE mli_acc32_t bias_additive(const MLI_PTR(int8_t) bias, mli_acc32_t init_accum,
        const fx_quant_specific_params* quant_params);
template <>
MLI_FORCE_INLINE mli_acc40_t bias_additive(const MLI_PTR(int16_t) bias, mli_acc40_t init_accum,
        const fx_quant_specific_params* quant_params);
template <>
MLI_FORCE_INLINE mli_acc32_t bias_additive(const MLI_PTR(int32_t) bias, mli_acc32_t init_accum,
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
        MLI_CONV_OUT_PTR(o_T) __restrict o_ptr,
        acc_T acc,
        const quant_T* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit);

#if defined(__Xvec_width)
template <>
MLI_FORCE_INLINE void result_cast_relu_store(
        MLI_PTR(int8_t) __restrict o_ptr,
        vNx4accshort_t acc,
        const s8asym_quant_specific_out_params_v* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit);
#endif

template <typename o_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE o_T ir_result_cast_relu_store(const acc_T acc, const quant_T* quant_params,
                                               const o_T val_min_limit, const o_T val_max_limit);
template <>
MLI_FORCE_INLINE int16_t ir_result_cast_relu_store(const mli_acc40_t acc, const fx_quant_specific_params* math_params,
                                                   const int16_t val_min_limit, const int16_t val_max_limit);
template <>
MLI_FORCE_INLINE int16_t ir_result_cast_relu_store(const mli_acc32_t acc, const fx_quant_specific_params* math_params,
                                                   const int16_t val_min_limit, const int16_t val_max_limit);
template <>
MLI_FORCE_INLINE int8_t ir_result_cast_relu_store(const mli_acc32_t acc, const s8asym_quant_specific_params* quant_params,
                                                  const int8_t val_min_limit, const int8_t val_max_limit);

template <typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T ir_rnn_result_requantize(const acc_T acc, const quant_T* params);
template <typename acc_T>
MLI_FORCE_INLINE acc_T ir_rnn_result_requantize(const acc_T acc, const fx_quant_specific_params* params);
template <>
MLI_FORCE_INLINE mli_acc32_t ir_rnn_result_requantize(
        const mli_acc32_t acc, const s8asym_quant_specific_params* params);

template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_sa8_fx16(
        const in_T in, const int16_t zero_point, const int scale);

template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_sa8_fx16(
        const in_T in, const int16_t zero_point, const int16_t scale, const int shift);

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
        const MLI_PTR(int8_t) __restrict weights, acc_T *init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step);

MLI_FORCE_INLINE mli_acc32_t weights_additive_d(
        const MLI_PTR(int8_t) __restrict weights, mli_acc32_t *init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step);

template <typename o_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void result_cast_relu_store(
        o_T __restrict o_ptr,
        acc_T acc,
        const quant_T* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit);

template <>
MLI_FORCE_INLINE void result_cast_relu_store(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
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
MLI_FORCE_INLINE out_T mli_prv_convert_sa8_fx16(
        const in_T in, const int16_t zero_point, const int16_t scale, const int shift);

template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_fx16_sa8(
        const in_T in, const int16_t zero_point, const int scale);
} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

#if defined(__Xvec_width)
MLI_FORCE_INLINE s8asym_quant_specific_out_params_v adjust_quant_params_v(s8asym_quant_specific_params* params, int krn_idx);
#endif

MLI_FORCE_INLINE fx_quant_specific_params adjust_quant_params_v(fx_quant_specific_params* in, int krn_idx);

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod_inputzp_1D_v(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int vals,
        const int in_step,
        const int krn_step,
        const s8asym_quant_specific_params* quant_params);

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod_inputzp_1D_v(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int vals,
        const int in_step,
        const int krn_step,
        const fx_quant_specific_params* quant_params);

//==========================================================================
// Calculation of weights additive (w_add) in
// dot_prod_asym = dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <typename w_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T weights_additive(
        const MLI_PTR(w_T) __restrict weights, acc_T init_accum,
        const quant_T* quant_params,
        const int width, const int height, const int ch, int col_step, int row_step, int ch_step);

#if defined(__Xvec_width)
template <>
MLI_FORCE_INLINE vNx4accshort_t weights_additive(
        const MLI_PTR(int8_t) __restrict weights, vNx4accshort_t init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, const int ch, int col_step, int row_step, int ch_step);
#endif

//==========================================================================
// Calculation of bias additive (bias_add) in
// dot_prod_asym= dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T bias_additive(const MLI_PTR(b_T) bias, acc_T init_accum,
        const quant_T* quant_params, bool add_preshift_rnd = true);

#if defined(__Xvec_width)
template <>
MLI_FORCE_INLINE vNx4accshort_t bias_additive(const MLI_PTR(int32_t) bias, vNx4accshort_t init_accum,
        const s8asym_quant_specific_out_params_v* quant_params, bool add_preshift_rnd);
#endif

//==========================================================================
// Storing result
//==========================================================================
template <typename o_T, typename acc_T, typename quant_T>
static MLI_FORCE_INLINE void result_cast_relu_store_v(
        MLI_CONV_OUT_PTR(o_T) __restrict o_ptr,
        acc_T acc,
        const quant_T* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        int num,
        bool add_preshift_rnd = false);

#if defined(__Xvec_width)
template <>
MLI_FORCE_INLINE void result_cast_relu_store_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        vNx4accshort_t acc,
        const s8asym_quant_specific_out_params_v* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        int num,
        bool add_preshift_rnd);
#endif

template <typename o_T, typename acc_T, typename quant_T>
static MLI_FORCE_INLINE void ir_result_cast_relu_store_v(
        MLI_CONV_OUT_PTR(o_T) __restrict o_ptr,
        acc_T acc,
        const quant_T* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        int num);

#if defined(__Xvec_width)
template <>
MLI_FORCE_INLINE void ir_result_cast_relu_store_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        vNx4accshort_t acc,
        const s8asym_quant_specific_out_params_v* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        int num);
#endif

template <typename acc_T, typename out_T, typename quant_T>
MLI_FORCE_INLINE out_T ir_rnn_result_requantize(
        const acc_T acc, const quant_T* params);

template <typename acc_T, typename out_T>
MLI_FORCE_INLINE out_T ir_rnn_result_requantize(
        const acc_T acc, const fx_quant_specific_params* params);

#if defined(__Xvec_width)
template <>
MLI_FORCE_INLINE vNx4int_t ir_rnn_result_requantize(
        const vNx4accshort_t acc,
        const s8asym_quant_specific_params* params);
#endif

//==========================================================================
// Convert functions
//==========================================================================
template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_sa8_fx16(
        const in_T in, const int16_t zero_point, const int scale);

template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_sa8_fx16(
        const in_T in, const int16_t zero_point, const int16_t scale, const int shift);

template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_fx16_sa8(
        const in_T in, const int16_t zero_point, const int scale);

#if defined(__Xvec_width)
template<>
MLI_FORCE_INLINE vNx4short_t mli_prv_convert_sa8_fx16(
        const vNx4short_t in_val,
        const int16_t zero_point,
        const int scale);
template<>
MLI_FORCE_INLINE vNx4char_t mli_prv_convert_fx16_sa8(
        const vNx4short_t in_val,
        const int16_t zero_point,
        const int scale);
template<>
MLI_FORCE_INLINE vNx4short_t mli_prv_convert_fx16_sa8(
        const vNx4short_t in_val,
        const int16_t zero_point,
        const int scale);
#endif

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_PRV_QUANT_DECL_H_

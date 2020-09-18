/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_DOTPROD_DECL_REF_H_
#define _MLI_KRN_DOTPROD_DECL_REF_H_

#include "mli_config.h"
#include "mli_prv_quant.h"
#include "mli_types.h"
#include "mli_prv_layout.h"

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
        
template <typename io_T, typename w_T, typename acc_T>
static acc_T __attribute__ ((always_inline)) dotprod1D(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int vals,
        const int in_step = 1,
        const int krn_step = 1);

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step);

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step);

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE void dotprod3D (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T * accu);

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE acc_T dotprod3D (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T accu);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step);

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE void dotprod2D_hwc_v (
        const MLI_PTR(in_T) __restrict in, 
        const MLI_PTR(w_T) __restrict krn,
        acc_T * accu,        
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step);

//The function uses pointers to pointers for in and krn. 
//The caller of the function should compensate for the increment
//done inside this function.
template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE void dotprod2D_hwc_v (
        const MLI_PTR(in_T) __restrict *in, 
        const MLI_PTR(w_T) __restrict *krn,
        acc_T * accu,        
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step);
//The function uses pointers to pointers for in and krn. 
//The caller of the function should compensate for the increment
//done inside this function.
template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D_inp_width_v(
        const MLI_PTR(io_T) __restrict *inp,
        const MLI_PTR(w_T)  __restrict *krn,
        acc_T *accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step,
        int in_width_step);

//The function uses pointers to pointers for in and krn. 
//The caller of the function should compensate for the increment
//done inside this function.
template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D(
        const MLI_PTR(io_T) __restrict *in,
        const MLI_PTR(w_T)  __restrict *krn,
        acc_T accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step);

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE void dotprod2D_hwc_v_point (
        const MLI_PTR(in_T) __restrict in, 
        const MLI_PTR(w_T) __restrict krn,
        acc_T * accu);

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D_inp_width_v(
        const MLI_PTR(io_T) __restrict inp,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T *accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step,
        int in_width_step);

template < typename in_T, typename w_T, typename acc_T > 
static MLI_FORCE_INLINE void dotprod3D_v_simple (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int in_ch,
        int in_row_step,
        int kern_row_step,
        int in_ch_step,
        int kern_ch_step,
        acc_T * accu);

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {
template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod1D_v(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int vals,
        const int in_step = 1,
        const int krn_step = 1);

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D_vv(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step);

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE acc_T dotprod3D_v (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T accu);

} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_DOTPROD_DECL_REF_H_

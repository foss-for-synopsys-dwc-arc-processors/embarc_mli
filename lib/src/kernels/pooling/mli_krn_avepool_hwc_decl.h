/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_AVEPOOL_HWC_DECL_H_
#define _MLI_KRN_AVEPOOL_HWC_DECL_H_

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
template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc_nopad(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot);
template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot);
template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc_krnpad(
        int row_beg,
        int row_end,
        int clmn_beg,
        int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {
template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc_nopad(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot);
template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot);
template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc_krnpad(
        int row_beg,
        int row_end,
        int clmn_beg,
        int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot);

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_AVEPOOL_HWC_DECL_H_

/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_REDUCE_SUM2D_DECL_H_
#define _MLI_KRN_REDUCE_SUM2D_DECL_H_

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

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void reduce_sum2D_v(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T * accu,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size);

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void reduce_sum2D(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T * accu,
        const int width,
        const int height,
        const int channels,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size);

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void reduce_sum(
        const io_T* __restrict in,
        const int16_t mul,
        acc_T * accu,
        const int vals,
        const int step = 1);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void reduce_sum2D_v(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T * accu,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size);

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void reduce_sum2D_d(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T * accu,
        const int width,
        const int height,
        int col_mem_stride,
        int row_mem_stride,
        const bool fixed_size);

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void reduce_sum2D_v(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T * accu,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size);

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void reduce_sum2D(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T * accu,
        const int width,
        const int height,
        const int channels,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size);

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_REDUCE_SUM2D_DECL_H_
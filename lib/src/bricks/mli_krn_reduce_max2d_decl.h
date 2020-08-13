/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RECUCE_MAX2D_DECL_H_
#define _MLI_KRN_RECUCE_MAX2D_DECL_H_

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
static inline void __attribute__((always_inline)) reduce_max2D_hwc_v(
		const MLI_PTR(io_T) in,
		MLI_PTR(io_T) out,
		const int width,
        const int height,
		const int channels,
		const int col_mem_stride,
		const int row_mem_stride,
		const bool fixed_size);

template <typename io_T>
static inline void __attribute__((always_inline)) reduce_max2D_hwc(
		const MLI_PTR(io_T) in,
		MLI_PTR(io_T) out,
		const int width,
        const int height,
		const int channels,
		const int col_mem_stride,
		const int row_mem_stride,
		const bool fixed_size);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {

template <typename io_T>
static inline void __attribute__((always_inline)) reduce_max2D_hwc_v(
		const MLI_PTR(io_T) in,
		MLI_PTR(io_T) out,
		const int width,
        const int height,
		const int channels,
		const int col_mem_stride,
		const int row_mem_stride,
		const bool fixed_size);

template <typename io_T>
static inline void __attribute__((always_inline)) reduce_max2D_hwc(
		const MLI_PTR(io_T) in,
		MLI_PTR(io_T) out,
		const int width,
        const int height,
		const int channels,
		const int col_mem_stride,
		const int row_mem_stride,
		const bool fixed_size);

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

template <typename io_T>
static inline void __attribute__((always_inline)) reduce_max2D_hwc_v(
		const MLI_PTR(io_T) in,
		MLI_PTR(io_T) out,
		const int width,
        const int height,
		const int channels,
		const int col_mem_stride,
		const int row_mem_stride,
		const bool fixed_size);

template <typename io_T>
static inline void __attribute__((always_inline)) reduce_max2D_hwc(
		const MLI_PTR(io_T) in,
		MLI_PTR(io_T) out,
		const int width,
        const int height,
		const int channels,
		const int col_mem_stride,
		const int row_mem_stride,
		const bool fixed_size);

} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RECUCE_MAX2D_DECL_H_

/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_ARGMAX_DECL_H_
#define _MLI_KRN_ARGMAX_DECL_H_

#include "mli_config.h"
#include "mli_mem_info.h"
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
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {
template <typename in_T, typename out_T>
static MLI_FORCE_INLINE void inner_loop(
        const generic_tensor_private_t<MLI_PTR(in_T)> *src_prv,
        const int dim0_idx,
        const int dim1_idx,
        const int dim2_idx,
        int dim3_idx,
        const int dim3_end,
        const int32_t topk,
        MLI_OUT_PTR(out_T) dst_tensor_arr);
} // namespace vdsp

////////////////////////////////////////////////////////////////////////////////
// REF
////////////////////////////////////////////////////////////////////////////////
namespace ref {
template <typename in_T, typename out_T>
static MLI_FORCE_INLINE void heapify(
        const MLI_PTR(in_T) src_tensor_arr,
        int size,
        int root_idx,
        MLI_OUT_PTR(out_T) dst_tensor_arr);

template <typename in_T, typename out_T>
static MLI_FORCE_INLINE void inner_loop(
        const generic_tensor_private_t<MLI_PTR(in_T)> *src_prv,
        const int dim0_idx,
        const int dim1_idx,
        const int dim2_idx,
        int dim3_idx,
        const int dim3_end,
        const int32_t topk,
        MLI_OUT_PTR(out_T) dst_tensor_arr);

template <typename in_T>
MLI_FORCE_INLINE void argmax_prepare_and_run(
        const mli_tensor *in,
        const mli_argmax_cfg *cfg,
        mli_tensor *out);

} // namespace ref

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_ARGMAX_DECL_H_

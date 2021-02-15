/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PERMUTE_DECL_H_
#define _MLI_KRN_PERMUTE_DECL_H_

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

template <typename io_T, bool asym>
static MLI_FORCE_INLINE mli_status mli_krn_permute_run(const mli_tensor *in, 
        const mli_permute_cfg *cfg, mli_tensor *out);

template <typename io_T>
static void mli_krn_permute_inner(const mli_tensor *in, uint32_t *out_shape, 
        int *out_increments, int *perm_dim,
        const MLI_PTR(io_T) input, MLI_PTR(io_T) output);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {

// template <typename io_T, bool asym>
// static MLI_FORCE_INLINE mli_status mli_krn_permute_run(const mli_tensor *in, 
//      const mli_permute_cfg *cfg, mli_tensor *out);

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

template <typename io_T>
static void mli_krn_permute_inner(const mli_tensor *in, uint32_t *out_shape, 
        int *out_increments, int *perm_dim,
        const MLI_PTR(io_T) input, MLI_PTR(io_T) output);

} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PERMUTE_DECL_H_

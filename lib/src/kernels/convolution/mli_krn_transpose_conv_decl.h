/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_TRANSPOSE_CONV_DECL_H_
#define _MLI_KRN_TRANSPOSE_CONV_DECL_H_

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
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void transpose_conv2d_prepare_and_run(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor *bias,
        const mli_conv2d_cfg *cfg,
        mli_tensor *out);
} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {
} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {
} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_TRANSPOSE_CONV_DECL_H_

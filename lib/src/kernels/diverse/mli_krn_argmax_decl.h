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
#include "mli_types.h"

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

template <typename in_T>
MLI_FORCE_INLINE void argmax_prepare_and_run(
        const mli_tensor *in,
        const mli_argmax_cfg *cfg,
        mli_tensor *out);

} // namespace ref

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_ARGMAX_DECL_H_

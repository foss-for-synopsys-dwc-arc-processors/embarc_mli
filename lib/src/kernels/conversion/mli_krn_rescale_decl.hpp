/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RESCALE_DECL_HPP_
#define _MLI_KRN_RESCALE_DECL_HPP_

#include "mli_config.h"
#include "mli_mem_info.h"
#include "mli_types.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"

namespace snps_arc::metaware::mli {
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

template <typename i_T, typename o_T>
mli_status MLI_FORCE_INLINE mli_krn_rescale(const mli_tensor *in,
                                            const mli_tensor *bias_in,
                                            const mli_tensor *scale,
                                            const mli_tensor *shift,
                                            const mli_tensor *bias_out,
                                            const int32_t rescale_axis,
                                            mli_tensor *out);

} // namespace ref
} // namespace krn
} // namespace snps_arc::metaware::mli

#endif // _MLI_KRN_RESCALE_DECL_HPP_

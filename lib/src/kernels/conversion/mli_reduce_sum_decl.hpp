/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_REDUCE_SUM_DECL_HPP_
#define _MLI_REDUCE_SUM_DECL_HPP_

#include "mli_config.h"
#include "mli_types.h"

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

template <typename io_T>
mli_status MLI_FORCE_INLINE mli_reduce_sum(const mli_tensor *in,
                                           const int8_t reduce_axis,
                                           mli_tensor *out);

} // namespace ref
} // namespace krn
} // namespace snps_arc::metaware::mli

#endif // _MLI_REDUCE_SUM_DECL_HPP_

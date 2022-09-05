/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_REDUCE_SUM_HPP_
#define _MLI_REDUCE_SUM_HPP_

#include "mli_reduce_sum_decl.hpp"

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
// Selecting between different variants (depending on hardware features) is
// done with 'using'. A completely different implementation can be used/'using'.
// However, also only a part of the reference together with optimized functions
// (for example *_dsp) can be used/'using'.

namespace snps_arc::metaware::mli {
namespace krn {

using snps_arc::metaware::mli::krn::ref::mli_reduce_sum;

} // namespace krn
} // namespace snps_arc::metaware::mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
// The reference (*_ref.h) implementation can run on all platforms and is always
// included. Other variants are included based on capabilities. Implementations
// below can depend on each other through declarations in *_decl.h.
#include "impl/mli_reduce_sum_ref.hpp"

#endif // _MLI_REDUCE_SUM_HPP_

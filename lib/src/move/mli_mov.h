/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_MOV_H_
#define _MLI_MOV_H_

#include "mli_mov_decl.h"

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
// Selecting between different variants (depending on hardware features) is
// done with 'using'. A completely different implementation can be used/'using'.
// However, also only a part of the reference together with optimized functions
// (from example *_dsp) can be used/'using'.

namespace mli {
namespace mov {
#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
using mli::mov::vdsp::mli_mov_memcpy;
using mli::mov::vdsp::mov_inner_loop;
#else
using mli::mov::ref::mli_mov_memcpy;
using mli::mov::ref::mov_inner_loop;
#endif
} // namespace krn
} // namespace mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
// The reference (*_ref.h) implementation can run on all platforms and is always
// included. Other variants are included based on capabilities. Implementations
// below can depend on each other through declarations in *_decl.h.

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#include "impl/mli_mov_vdsp.h"
#else
#include "impl/mli_mov_ref.h"
#endif

#endif // _MLI_MOV_H_

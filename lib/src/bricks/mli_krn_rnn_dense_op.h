/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RNN_DENSE_OP_H_
#define _MLI_KRN_RNN_DENSE_OP_H_

#include "mli_krn_rnn_dense_op_decl.h"

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
// Selecting between different variants (depending on hardware features) is
// done with 'using'. A completely different implementation can be used/'using'.
// However, also only a part of the reference together with optimized functions
// (from example *_dsp) can be used/'using'.

namespace mli {
namespace krn {
#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
using mli::krn::vdsp::rnn_dense_op;
using mli::krn::vdsp::rnn_dense_op_stacked;

#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
using mli::krn::ref::rnn_dense_op;
using mli::krn::ref::rnn_dense_op_stacked;

#else
using mli::krn::ref::rnn_dense_op;
using mli::krn::ref::rnn_dense_op_stacked;

#endif
} // namespace krn
} // namespace mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
// The reference (*_ref.h) implementation can run on all platforms and is always
// included. Other variants are included based on capabilities. Implementations
// below can depend on each other through declarations in *_decl.h.
#include "impl/mli_krn_rnn_dense_op_ref.h"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#include "impl/mli_krn_rnn_dense_op_vdsp.h"
#endif

#if !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
#include "impl/mli_krn_rnn_dense_op_dsp.h"
#endif

#endif // _MLI_KRN_RNN_DENSE_OP_H_

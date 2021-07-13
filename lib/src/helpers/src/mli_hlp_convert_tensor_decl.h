/*
* Copyright 2020 - 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_HLP_CONVERT_TENSOR_DECL_H_
#define _MLI_HLP_CONVERT_TENSOR_DECL_H_

#include "mli_config.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace hlp {

typedef enum {
    QUANTIZE = 0,
    DEQUANTIZE
} convert_mode;

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

template <typename in_T, typename out_T, typename acc_T>
mli_status convert_quantized_data(
        const mli_tensor *src,
        mli_tensor *dst);

template <typename t_T>
mli_status convert_float_data(
        const mli_tensor *src,
        mli_tensor *dst,
        convert_mode direction);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

template <typename in_T, typename out_T, typename acc_T>
mli_status convert_quantized_data(
        const mli_tensor *src,
        mli_tensor *dst);

} // namespace vdsp

} // namespace hlp
} // namespace mli

#endif // _MLI_HLP_CONVERT_TENSOR_DECL_H_

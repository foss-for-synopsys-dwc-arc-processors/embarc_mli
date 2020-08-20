/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_LUT_DECL_H_
#define _MLI_PRV_LUT_DECL_H_

#include "mli_config.h"
#include "mli_types.h"
#include "mli_prv_tensor.h"


//TODO: Remove extra
const int kTmpBufSize = 32;
const int kLutOutFracBits = 15;
const int kTransfFuncIntBits = 0;
const int kMaxFracBitsFx16 = (sizeof(int16_t) * 8) - 1;
const int kMaxFracBitsFx8 = (sizeof(int8_t) * 8) - 1;   //

const int kAsymZeroPointFx16 = -128;
const int kAsymScaleFx16 = 256;
const int kAsymScalePowerFx16 = 8;
const int kAsymScaleFracBits = 16;

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
template <typename io_T, bool convert = false>
static void activation_lut(
    const MLI_PTR(io_T) in,
    MLI_OUT_PTR(io_T) out,
    const mli_lut *lut,
    int8_t scale_frac_bits,
    int length,
    int scale = 0,
    int16_t zero_point = 0);
} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {
template <typename io_T, bool convert = false>
static void activation_lut(
    const MLI_PTR(io_T) in,
    MLI_OUT_PTR(io_T) out,
    const mli_lut *lut,
    int8_t scale_frac_bits,
    int length,
    int scale = 0,
    int16_t zero_point = 0);
} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {
template <typename io_T, bool convert = false>
static void activation_lut(
    const MLI_PTR(io_T) in,
    MLI_OUT_PTR(io_T) out,
    const mli_lut *lut,
    int8_t scale_frac_bits,
    int length,
    int scale = 0,
    int16_t zero_point = 0);
} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_PRV_LUT_DECL_H_

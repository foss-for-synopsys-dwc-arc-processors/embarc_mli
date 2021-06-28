/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_LSTM_CELL_DEC_H_
#define _MLI_KRN_LSTM_CELL_DEC_H_

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
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void lstm_cell_prepare_and_run(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out, 
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut, 
        const mli_rnn_cell_cfg * cfg, 
        mli_tensor * cell,
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
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void lstm_cell_prepare_and_run(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out, 
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut, 
        const mli_rnn_cell_cfg * cfg, 
        mli_tensor * cell,
        mli_tensor *out);
} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_LSTM_CELL_DEC_H_

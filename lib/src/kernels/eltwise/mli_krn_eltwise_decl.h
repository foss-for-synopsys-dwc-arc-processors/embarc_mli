/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_ELTWISE_ADD_DECL_REF_H_
#define _MLI_KRN_ELTWISE_ADD_DECL_REF_H_

#include "mli_config.h"
#include "mli_types.h"
#include "mli_prv_quant.h"

namespace mli {
typedef enum {
    ELTWISE_ADD = 0,
    ELTWISE_SUB,
    ELTWISE_MUL,
    ELTWISE_MAX,
    ELTWISE_MIN
} mli_eltwise_type;

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

template <typename io_T, mli_eltwise_type func_type, bool convert = false>
static MLI_FORCE_INLINE void eltwise_prepare_and_run(
        const mli_tensor *in1,
        const mli_tensor *in2,
        mli_tensor *out);

template <typename io_T, mli_eltwise_type func_type, bool convert = false>
static MLI_FORCE_INLINE void eltwise_op_basic(
        const generic_tensor_private_t<MLI_PTR(io_T)> *in1,
        const generic_tensor_private_t<MLI_PTR(io_T)> *in2,
        generic_tensor_private_t<MLI_OUT_PTR(io_T)> *out,
        const int op1_size,
        const int op2_size,
        const int mul_out_shift,
        const struct s8asym_quant_params *in_quant_params = nullptr,
        const struct s8asym_quant_params *out_quant_params = nullptr);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {

template <typename io_T, mli_eltwise_type func_type, bool convert = false>
static MLI_FORCE_INLINE void eltwise_op_basic(
        const generic_tensor_private_t<MLI_PTR(io_T)> *in1,
        const generic_tensor_private_t<MLI_PTR(io_T)> *in2,
        generic_tensor_private_t<MLI_OUT_PTR(io_T)> *out,
        const int op1_size,
        const int op2_size,
        const int mul_out_shift,
        const struct s8asym_quant_params *in_quant_params = nullptr,
        const struct s8asym_quant_params *out_quant_params = nullptr);

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {

template <typename io_T, mli_eltwise_type func_type, bool convert = false>
static MLI_FORCE_INLINE void eltwise_op_basic(
        const generic_tensor_private_t<MLI_PTR(io_T)> *in1,
        const generic_tensor_private_t<MLI_PTR(io_T)> *in2,
        generic_tensor_private_t<MLI_OUT_PTR(io_T)> *out,
        const int op1_size,
        const int op2_size,
        const int mul_out_shift,
        const struct s8asym_quant_params *in_quant_params = nullptr,
        const struct s8asym_quant_params *out_quant_params = nullptr);

} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_ELTWISE_ADD_DECL_REF_H_

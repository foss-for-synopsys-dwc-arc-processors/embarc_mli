/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_QUANT_H_
#define _MLI_PRV_QUANT_H_

#include "mli_config.h"
#include "mli_math.h"
#include "mli_prv_quant_decl.h"

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
// Reference code is not yet suitable for VDSP
using mli::krn::vdsp::mli_prv_convert_sa8_fx16;
using mli::krn::vdsp::mli_prv_convert_fx16_sa8;

#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
using mli::krn::ref::define_quant_params;
//using mli::krn::dsp::adjust_quant_params;
using mli::krn::ref::adjust_quant_params;
using mli::krn::ref::mli_prv_calc_out_mul;
using mli::krn::ref::weights_additive;
using mli::krn::dsp::weights_additive_d;
using mli::krn::dsp::weights_additive_v;
using mli::krn::ref::zp_additive;
using mli::krn::ref::bias_additive;
using mli::krn::ref::result_cast;
using mli::krn::ref::result_cast_relu_store;
using mli::krn::dsp::result_cast_relu_store_v;
using mli::krn::dsp::result_cast_relu_store_inp_width_v;
using mli::krn::dsp::mli_prv_convert_sa8_fx16;
using mli::krn::dsp::mli_prv_convert_fx16_sa8;

#else
using mli::krn::ref::define_quant_params;
using mli::krn::ref::adjust_quant_params;
using mli::krn::ref::mli_prv_calc_out_mul;
using mli::krn::ref::weights_additive;
using mli::krn::ref::zp_additive;
using mli::krn::ref::bias_additive;
using mli::krn::ref::result_cast;
using mli::krn::ref::result_cast_relu_store;
using mli::krn::ref::mli_prv_convert_sa8_fx16;
using mli::krn::ref::mli_prv_convert_fx16_sa8;

#endif

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
//#if defined(__FXAPI__)
#include "impl/mli_prv_quant_ref.h"
//#endif

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#include "impl/mli_prv_quant_vdsp.h"
#endif

#if !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
#include "impl/mli_prv_quant_dsp.h"
#endif

#endif /* _MLI_PRV_QUANT_H_ */

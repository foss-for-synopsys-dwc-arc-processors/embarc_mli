/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_MATH_H_
#define _MLI_MATH_H_

#include "mli_config.h"
//=========================================================================
//
// Declaration
//
//=========================================================================

template < typename io_T > MLI_FORCE_INLINE io_T mli_math_add_fx(io_T L, io_T R);
template < typename io_T > MLI_FORCE_INLINE io_T mli_math_sub_fx(io_T L, io_T R);
template < typename io_T > MLI_FORCE_INLINE io_T mli_math_max_fx(io_T L, io_T R);
template <typename l_T, typename r_T> MLI_FORCE_INLINE l_T mli_math_max_fx(l_T L, r_T R);
template < typename io_T > MLI_FORCE_INLINE io_T mli_math_min_fx(io_T L, io_T R);
template <typename l_T, typename r_T> MLI_FORCE_INLINE l_T mli_math_min_fx(l_T L, r_T R);
template < typename in_T, typename acc_T > MLI_FORCE_INLINE acc_T mli_math_mul_fx(in_T L, in_T R);
template < typename in_T, typename acc_T > MLI_FORCE_INLINE acc_T mli_math_mul_fx_high(in_T L, in_T R);
template < typename l_T, typename r_T, typename acc_T > MLI_FORCE_INLINE acc_T mli_math_mac_fx(acc_T acc, l_T L, r_T R);
template < typename out_T, typename acc_T > MLI_FORCE_INLINE out_T mli_math_acc_cast_fx(acc_T acc, int shift_right);
template < typename acc_T > MLI_FORCE_INLINE acc_T mli_math_acc_ashift_fx(acc_T acc, int shift_right);
template < typename out_T > MLI_FORCE_INLINE out_T mli_math_cast_ptr_to_scalar_fx(void *src);
template < typename in_T > MLI_FORCE_INLINE void *mli_math_cast_scalar_to_ptr_fx(in_T src);

template <typename in_T, typename out_T> MLI_FORCE_INLINE out_T mli_math_cast_fx(in_T in_val, int shift_right);
template <typename in_T, typename out_T> MLI_FORCE_INLINE out_T mli_math_cast_fx(in_T in_val);
template <typename in_T> MLI_FORCE_INLINE in_T mli_math_asr_rnd_fx(in_T x, int nbits);

// TODO: the reference PAL is not yet fully developed and cannot be used here.
//#if defined(MLI_BUILD_REFERENCE)
//#include "ref/mli_math.h"
#if defined(__Xvec_width)
#include "vdsp/mli_math.h"
#elif defined(__FXAPI__)
#include "dsp/mli_math.h"
#endif

#endif // _MLI_MATH_H_
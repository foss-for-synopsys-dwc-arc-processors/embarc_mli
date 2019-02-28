/* This file is generated, do not edit!
 * edit following template file instead:
 * header_filetemplate.txt
 */
/*
 *  Copyright (c) 2019, Synopsys, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1) Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * 2)  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3) Neither the name of the <ORGANIZATION> nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _MLI_KRN_MAXPOOL_SPEC_API_H_
#define _MLI_KRN_MAXPOOL_SPEC_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "mli_types.h"

//===================================================================
// MaxPooling specialization kernels implementation
//===================================================================

mli_status mli_krn_maxpool_chw_fx16_k2x2_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k2x2_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k2x2_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x3_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x3_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k4x4_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k4x4_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k4x4_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k5x5_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k5x5_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k5x5_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k6x6_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k6x6_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k6x6_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k7x7_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k7x7_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k7x7_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k8x8_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k8x8_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k8x8_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k9x9_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k9x9_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k9x9_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k10x10_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k10x10_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k10x10_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k1x2_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k1x2_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k1x3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k1x3_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k2x1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k2x1_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x1_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k2x2_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k2x2_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k2x2_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x3_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x3_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k4x4_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k4x4_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k4x4_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k5x5_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k5x5_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k5x5_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k6x6_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k6x6_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k6x6_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k7x7_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k7x7_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k7x7_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k8x8_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k8x8_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k8x8_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k9x9_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k9x9_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k9x9_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k10x10_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k10x10_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k10x10_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k1x2_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k1x2_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k1x3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k1x3_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k2x1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k2x1_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x1_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k1xn_str1(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_knx1_str1(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_ch1_str1(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k2x2(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k2x2_ch1(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x3(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_k3x3_ch1(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx16_generic(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);

mli_status mli_krn_maxpool_chw_fx8_k2x2_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k2x2_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k2x2_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x3_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x3_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k4x4_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k4x4_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k4x4_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k5x5_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k5x5_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k5x5_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k6x6_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k6x6_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k6x6_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k7x7_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k7x7_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k7x7_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k8x8_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k8x8_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k8x8_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k9x9_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k9x9_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k9x9_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k10x10_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k10x10_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k10x10_ch3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k1x2_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k1x2_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k1x3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k1x3_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k2x1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k2x1_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x1_ch1_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k2x2_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k2x2_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k2x2_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x3_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x3_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k4x4_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k4x4_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k4x4_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k5x5_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k5x5_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k5x5_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k6x6_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k6x6_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k6x6_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k7x7_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k7x7_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k7x7_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k8x8_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k8x8_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k8x8_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k9x9_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k9x9_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k9x9_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k10x10_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k10x10_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k10x10_ch3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k1x2_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k1x2_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k1x3_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k1x3_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k2x1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k2x1_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x1_ch1_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k1xn_str1(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_knx1_str1(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_ch1_str1(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k2x2(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k2x2_ch1(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x3(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_k3x3_ch1(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);
mli_status mli_krn_maxpool_chw_fx8_generic(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out);

#ifdef __cplusplus
}
#endif
#endif                          //_MLI_KRN_MAXPOOL_SPEC_API_H_
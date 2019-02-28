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

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"  // for mli_prv_fx_init_dsp_ctrl()
#include "mli_prv_load_store.h"
#include "mli_prv_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")

mli_status mli_krn_leaky_relu_fx8(const mli_tensor *in, const mli_tensor *slope_coeff, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_leaky_relu_fx8(in, slope_coeff, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli_prv_fx_init_dsp_ctrl();
    mli_prv_copy_tensor_format(in, out);

    const MLI_PTR(int8_t) in_data = (MLI_PTR(int8_t))in->data;
    MLI_PTR(int8_t) out_data = (MLI_PTR(int8_t))out->data;
    int el_num = (int)mli_prv_count_elem_num(in);
    int shift = mli_prv_calc_shift(in, slope_coeff, out);
    v2q15_t x;

    // scalar value is stored in data field or is pointed by data field
    v2q15_t scale = fx_replic_v2q15(
            (slope_coeff->rank == 0) ? (int8_t)(intptr_t)(slope_coeff->data) : ((int8_t *)slope_coeff->data)[0]);

    if (mli_prv_less_than_1(scale[0], slope_coeff->el_params.fx.frac_bits)) {
        if (el_num & 1) {
            x = mli_prv_load_1_sample(in_data);
            x = fx_max_v2q15(
                    x, fx_sat_v2q15_n(fx_v2q15_cast_nf_asl_rnd_v2a40(fx_v2a40_mpy_nf_v2q15(x, scale), 16 - shift), 8));
            mli_prv_store_1_sample(out_data, x);
            in_data++;
            out_data++;
        }
        for (int i = 0; i < (el_num >> 1); i++) {
            x = mli_prv_load_2_samples(in_data);
            x = fx_max_v2q15(
                    x, fx_sat_v2q15_n(fx_v2q15_cast_nf_asl_rnd_v2a40(fx_v2a40_mpy_nf_v2q15(x, scale), 16 - shift), 8));
            mli_prv_store_2_samples(out_data, x);
            in_data += 2;
            out_data += 2;
        }
    } else {
        if (el_num & 1) {
            x = mli_prv_load_1_sample(in_data);
            x = fx_min_v2q15(
                    x, fx_sat_v2q15_n(fx_v2q15_cast_nf_asl_rnd_v2a40(fx_v2a40_mpy_nf_v2q15(x, scale), 16 - shift), 8));
            mli_prv_store_1_sample(out_data, x);
            in_data++;
            out_data++;
        }
        for (int i = 0; i < (el_num >> 1); i++) {
            x = mli_prv_load_2_samples(in_data);
            x = fx_min_v2q15(
                    x, fx_sat_v2q15_n(fx_v2q15_cast_nf_asl_rnd_v2a40(fx_v2a40_mpy_nf_v2q15(x, scale), 16 - shift), 8));
            mli_prv_store_2_samples(out_data, x);
            in_data += 2;
            out_data += 2;
        }
    }

    return MLI_STATUS_OK;
}

mli_status mli_krn_leaky_relu_fx16(const mli_tensor *in, const mli_tensor *slope_coeff, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_leaky_relu_fx16(in, slope_coeff, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli_prv_fx_init_dsp_ctrl();
    mli_prv_copy_tensor_format(in, out);

    const MLI_PTR(int16_t) in_data = (MLI_PTR(int16_t))in->data;
    MLI_PTR(int16_t) out_data = (MLI_PTR(int16_t))out->data;
    int el_num = (int)mli_prv_count_elem_num(in);
    int shift = mli_prv_calc_shift(in, slope_coeff, out);
    v2q15_t x;

    // scalar value is stored in data field or is pointed by data field
    v2q15_t scale = fx_replic_v2q15(
            (slope_coeff->rank == 0) ? (int16_t)(intptr_t)(slope_coeff->data) : ((int16_t *)slope_coeff->data)[0]);

    if (mli_prv_less_than_1(scale[0], slope_coeff->el_params.fx.frac_bits)) {
        if (el_num & 1) {
            x = mli_prv_load_1_sample(in_data);
            x = fx_max_v2q15(x, fx_v2q15_cast_nf_asl_rnd_v2a40(fx_v2a40_mpy_nf_v2q15(x, scale), 16 - shift));
            mli_prv_store_1_sample(out_data, x);
            in_data++;
            out_data++;
        }
        for (int i = 0; i < (el_num >> 1); i++) {
            x = mli_prv_load_2_samples(in_data);
            x = fx_max_v2q15(x, fx_v2q15_cast_nf_asl_rnd_v2a40(fx_v2a40_mpy_nf_v2q15(x, scale), 16 - shift));
            mli_prv_store_2_samples(out_data, x);
            in_data += 2;
            out_data += 2;
        }
    } else {
        if (el_num & 1) {
            x = mli_prv_load_1_sample(in_data);
            x = fx_min_v2q15(x, fx_v2q15_cast_nf_asl_rnd_v2a40(fx_v2a40_mpy_nf_v2q15(x, scale), 16 - shift));
            mli_prv_store_1_sample(out_data, x);
            in_data++;
            out_data++;
        }
        for (int i = 0; i < (el_num >> 1); i++) {
            x = mli_prv_load_2_samples(in_data);
            x = fx_min_v2q15(x, fx_v2q15_cast_nf_asl_rnd_v2a40(fx_v2a40_mpy_nf_v2q15(x, scale), 16 - shift));
            mli_prv_store_2_samples(out_data, x);
            in_data += 2;
            out_data += 2;
        }
    }

    return MLI_STATUS_OK;
}
#pragma code()

#ifdef __cplusplus
}
#endif

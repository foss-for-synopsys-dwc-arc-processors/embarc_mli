/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_prv_dsp.h"
#include "mli_prv_load_store.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")

mli_status mli_krn_relu_fx8(const mli_tensor* in, const mli_relu_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_relu_fx8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(int8_t) in_data = (MLI_PTR(int8_t))in->data;
    MLI_OUT_PTR(int8_t) out_data = (MLI_OUT_PTR(int8_t))out->data;
    int el_num = (int)mli_prv_count_elem_num(in);
    int vec_num = el_num >> 1;

    v2q15_t zero = {0, 0};

    if (cfg->type == MLI_RELU_GEN) {
        if (el_num & 1) {
            mli_prv_store_1_sample(out_data, fx_max_v2q15(mli_prv_load_1_sample(in_data), zero));
            in_data++;
            out_data++;
        }
        // General Relu case we put in the separate branch as more common case
        // Despite the fact that it may be handled by "else" branch
        // we provide compiler a little more info
        for (int i = 0; i < vec_num; i++) {
            mli_prv_store_2_samples(out_data, fx_max_v2q15(mli_prv_load_2_samples(in_data), zero));
            in_data += 2;
            out_data += 2;
        }
    } else {
        // Case with more tricky limits, which we're defining by separate function
        mli_minmax_t limits = mli_prv_get_relu_min_max(cfg, in);
        v2q15_t min_vec = fx_replic_v2q15(limits.min);
        v2q15_t max_vec = fx_replic_v2q15(limits.max);

        if (el_num & 1) {
            mli_prv_store_1_sample(
                    out_data, fx_min_v2q15(fx_max_v2q15(mli_prv_load_1_sample(in_data), min_vec), max_vec));
            in_data++;
            out_data++;
        }

        for (int i = 0; i < vec_num; i++) {
            mli_prv_store_2_samples(
                    out_data, fx_min_v2q15(fx_max_v2q15(mli_prv_load_2_samples(in_data), min_vec), max_vec));
            in_data += 2;
            out_data += 2;
        }
    }

    mli_prv_copy_tensor_format(in, out);
    return MLI_STATUS_OK;
}

mli_status mli_krn_relu_fx16(const mli_tensor* in, const mli_relu_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_relu_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(int16_t) in_data = (MLI_PTR(int16_t))in->data;
    MLI_OUT_PTR(int16_t) out_data = (MLI_OUT_PTR(int16_t))out->data;
    int el_num = (int)mli_prv_count_elem_num(in);
    int vec_num = el_num >> 1;

    v2q15_t zero = {0, 0};

    if (cfg->type == MLI_RELU_GEN) {
        if (el_num & 1) {
            mli_prv_store_1_sample(out_data, fx_max_v2q15(mli_prv_load_1_sample(in_data), zero));
            in_data++;
            out_data++;
        }
        // General Relu case we put in the separate branch as more common case
        // Despite the fact that it may be handled by "else" branch
        // we provide compiler a little more info
        for (int i = 0; i < vec_num; i++) {
            mli_prv_store_2_samples(out_data, fx_max_v2q15(mli_prv_load_2_samples(in_data), zero));
            in_data += 2;
            out_data += 2;
        }
    } else {
        // Case with more tricky limits, which we're defining by separate function
        mli_minmax_t limits = mli_prv_get_relu_min_max(cfg, in);
        v2q15_t min_vec = fx_replic_v2q15(limits.min);
        v2q15_t max_vec = fx_replic_v2q15(limits.max);

        if (el_num & 1) {
            mli_prv_store_1_sample(
                    out_data, fx_min_v2q15(fx_max_v2q15(mli_prv_load_1_sample(in_data), min_vec), max_vec));
            in_data++;
            out_data++;
        }

        for (int i = 0; i < vec_num; i++) {
            mli_prv_store_2_samples(
                    out_data, fx_min_v2q15(fx_max_v2q15(mli_prv_load_2_samples(in_data), min_vec), max_vec));
            in_data += 2;
            out_data += 2;
        }
    }

    mli_prv_copy_tensor_format(in, out);
    return MLI_STATUS_OK;
}

#pragma code()

#ifdef __cplusplus
}
#endif

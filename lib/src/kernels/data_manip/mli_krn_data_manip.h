/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_DATA_MANIP_H_
#define _MLI_KRN_DATA_MANIP_H_

#include <stdint.h>
#include <string.h>

#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"  // for mli_prv_fx_init_dsp_ctrl()
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {

typedef enum {
    LAYOUT_CHW = 0,
    LAYOUT_HWC
} mli_layout_type;

#pragma Code(".mli_lib")

//======================================================
//
//======================================================
template <typename io_T>
inline void permute_data(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {
    mli_prv_fx_init_dsp_ctrl();

    int total_count = mli_prv_count_elem_num(in);
    int rank = in->rank;

    int perm_dim[] = {0, 1, 2, 3};   // default order of output matrix dimension
    int out_shape[] = {1, 1, 1, 1};  // for work with output matrix with any dimension up to 4
    int in_shape[] = {1, 1, 1, 1};   // for work with input matrix with any dimension up to 4

    // Strides on input tensor across output dimensions
    int strides[] = {total_count, total_count, total_count, total_count};

    // Prepare required data - strides on input, shapes
    for (int k = 0; k < rank; k++) {
        int idx = MLI_MAX_RANK - rank + k;
        const int perm_dim_val = cfg->perm_dim[k];
        perm_dim[idx] = cfg->perm_dim[k];
        in_shape[idx] = in->shape[k];
        out_shape[idx] = in->shape[cfg->perm_dim[k]];
        if (perm_dim_val < (MLI_MAX_RANK - 1)) {
            strides[idx] = mli_prv_count_elem_num_part(in, perm_dim_val + 1);
        } else {
            strides[idx] = 1;
        }
    }

    // Main transpose operation.
    const io_T *dim0_data_ptr = static_cast<io_T *>(in->data.mem.void_p);
    io_T *output = static_cast<io_T *>(out->data.mem.void_p);
    for (int d0_cnt = 0; d0_cnt < out_shape[0]; d0_cnt++, dim0_data_ptr += strides[0]) {
        const io_T *dim1_data_ptr = dim0_data_ptr;
        for (int d1_cnt = 0; d1_cnt < out_shape[1]; d1_cnt++, dim1_data_ptr += strides[1]) {
            const io_T *dim2_data_ptr = dim1_data_ptr;
            for (int d2_cnt = 0; d2_cnt < out_shape[2]; d2_cnt++, dim2_data_ptr += strides[2]) {
                const io_T *dim3_data_ptr = dim2_data_ptr;
                for (int d3_cnt = 0; d3_cnt < out_shape[3]; d3_cnt++, dim3_data_ptr += strides[3]) {
                    (*output++) = *dim3_data_ptr;
                }
            }
        }
    }

    // Fill output tensor descr
    int *shape_ptr = &out_shape[MLI_MAX_RANK - rank];
    for (int k = 0; k < rank; k++) out->shape[k] = shape_ptr[k];
    out->rank = in->rank;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;
    out->el_type = in->el_type;
}

//======================================================
//
//======================================================
template <typename io_T, mli_layout_type layout_type>
inline void padding2D_data(const mli_tensor *in, const mli_padding2d_cfg *cfg, mli_tensor *out) {
    mli_prv_fx_init_dsp_ctrl();
    const uint32_t elem_size = mli_hlp_tensor_element_size(in);

    int channels, in_height, in_width, in_row_elements;
    int out_height, out_width;
    int indent_left, indent_right;
    int indent_top, indent_bot;
    if (layout_type == LAYOUT_HWC) {
        // Consider input as 1 fmap with CH*WIDTH elements in row
        in_height = static_cast<int>(in->shape[FMAP_H_DIM_HWC]);
        in_width = static_cast<int>(in->shape[FMAP_W_DIM_HWC]);
        channels = static_cast<int>(in->shape[FMAP_C_DIM_HWC]);

        out_height = in_height + cfg->padding_top + cfg->padding_bottom;
        out_width = in_width + cfg->padding_left + cfg->padding_right;

        indent_left = cfg->padding_left * channels;
        indent_right = cfg->padding_right * channels;
        indent_top = cfg->padding_top * out_width * channels;
        indent_bot = cfg->padding_bottom * out_width * channels;
        in_row_elements = in_width * channels;
    } else { // LAYOUT_CHW
        in_height = static_cast<int>(in->shape[FMAP_H_DIM_CHW]);
        in_width = static_cast<int>(in->shape[FMAP_W_DIM_CHW]);
        channels = static_cast<int>(in->shape[FMAP_C_DIM_CHW]);

        out_height = in_height + cfg->padding_top + cfg->padding_bottom;
        out_width = in_width + cfg->padding_left + cfg->padding_right;

        indent_left = cfg->padding_left;
        indent_right = cfg->padding_right;
        indent_top = cfg->padding_top * out_width;
        indent_bot = cfg->padding_bottom * out_width;
        in_row_elements = in_width;
    }

    const int padding_fmaps = (layout_type == LAYOUT_HWC) ? 1 : channels;
    const int rows_to_copy = in_height;
    const io_T *in_ptr = static_cast<io_T *>(in->data.mem.void_p);
    io_T *out_ptr = static_cast<io_T *>(out->data.mem.void_p);

    // For simplicity - use memset for all out memory at first,
    // and then copying input row-by-row (once in case of HWC, or for each channel in case of CHW)
    memset(out->data.mem.void_p, 0, elem_size * out_width * out_height * channels);
    for (int fmap_idx = 0; fmap_idx < padding_fmaps; fmap_idx++) {
        out_ptr += indent_top;
        for (int row_idx = 0; row_idx < rows_to_copy; row_idx++) {
            out_ptr += indent_left;
            memcpy((void *)out_ptr, (void *)in_ptr, in_row_elements * elem_size);
            out_ptr += indent_right + in_row_elements;
            in_ptr += in_row_elements;
        }
        out_ptr += indent_bot;
    }

    // fill output tensor parameters
    out->rank = in->rank;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;
    out->el_type = in->el_type;
    if (layout_type == LAYOUT_HWC) {
        out->shape[FMAP_H_DIM_HWC] = out_height;
        out->shape[FMAP_W_DIM_HWC] = out_width;
        out->shape[FMAP_C_DIM_HWC] = channels;
    } else {
        out->shape[FMAP_H_DIM_CHW] = out_height;
        out->shape[FMAP_W_DIM_CHW] = out_width;
        out->shape[FMAP_C_DIM_CHW] = channels;
    }
}

template <typename io_T>
inline void concat_data(const mli_tensor **inputs, const mli_concat_cfg *cfg, mli_tensor *out) {
    mli_prv_fx_init_dsp_ctrl();
    // Consider each tensor as 2-DIM: [sub_tensors_num, subtensor_sz]
    // where:
    // subtensor_sz - number of eliments starting
    //					from concatination axis (shape[axis]*shape[axis+1]*...)
    // sub_tensors_num - number of such sub-tensors in the source tensor (must be equal for all tensors)
    // In this case goal is to concatenate subtensors while interchanging sources
    const int concat_dim = cfg->axis;
    const int tensors_num = cfg->tensors_num;
    uint32_t elem_size = mli_hlp_tensor_element_size(inputs[0]);

    int concat_dim_total = 0;
    io_T *out_ptr = static_cast<io_T *>(out->data.mem.void_p);
    const io_T *inputs_ptr[MLI_CONCAT_MAX_TENSORS];
    int sub_tsr_sz[MLI_CONCAT_MAX_TENSORS];

    for (int idx = 0; idx < tensors_num; idx++) {
        inputs_ptr[idx] = static_cast<io_T *>(inputs[idx]->data.mem.void_p);
        sub_tsr_sz[idx] = mli_prv_count_elem_num_part(inputs[idx], concat_dim);
        concat_dim_total += inputs[idx]->shape[concat_dim];
    }

    // Number of subtensors must be equal for all of them
    const int sub_tsr_num = mli_prv_count_elem_num(inputs[0]) / (sub_tsr_sz[0]);

    // Concatenation of tensors into one
    for (int i = 0; i < sub_tsr_num; i++) {
        for (int t_idx = 0; t_idx < tensors_num; t_idx++) {
            memcpy((void *)out_ptr, (void *)inputs_ptr[t_idx], sub_tsr_sz[t_idx] * elem_size);

            out_ptr += sub_tsr_sz[t_idx];
            inputs_ptr[t_idx] += sub_tsr_sz[t_idx];
        }
    }

    out->rank = inputs[0]->rank;
    out->el_params.fx.frac_bits = inputs[0]->el_params.fx.frac_bits;
    out->el_type = inputs[0]->el_type;
    for (int idx = 0; idx < inputs[0]->rank; idx++) out->shape[idx] = inputs[0]->shape[idx];
    out->shape[concat_dim] = concat_dim_total;
}

#pragma Code()
}  // namespace mli

#endif  //_MLI_KRN_DATA_MANIP_H_
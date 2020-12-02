/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PERMUTE_REF_H_
#define _MLI_KRN_PERMUTE_REF_H_

#include <stdint.h>
#include <string.h>
#include "stdio.h"

#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace ref {


#pragma MLI_CODE_SECTION_START(".mli_lib")

template <typename io_T>
void print_tensor(io_T* data, int* shape, int* memstr) {
    printf("[");
    for (int cnt0 = 0; cnt0 < shape[0]; cnt0++) {
        printf("[");
        for (int cnt1 = 0; cnt1 < shape[1]; cnt1++) {
            printf("[");
            for (int cnt2 = 0; cnt2 < shape[2]; cnt2++) {
                printf("[");
                for (int cnt3 = 0; cnt3 < shape[3]; cnt3++) {
                    printf("%i, ", (int) *data);
                    data += memstr[3];
                }
                printf("],\n");
                data += memstr[2];
            }
            printf("],\n");
            data += memstr[1];
        }
        printf("],\n");
        data += memstr[0];
    }
    printf("]\n\n");
}

//======================================================
//
//======================================================
template <typename io_T, bool asym>
static MLI_FORCE_INLINE mli_status mli_krn_permute_run(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {

    // int total_count = mli_prv_count_elem_num(in);
    int rank = in->rank;

    int perm_dim_inv[MLI_MAX_RANK];
    int perm_dim[] = {0, 1, 2, 3};   // default order of output matrix dimension
    int out_shape[] = {1, 1, 1, 1};  // for work with output matrix with any dimension up to 4
    int in_shape[] = {1, 1, 1, 1};   // for work with input matrix with any dimension up to 4

    // Strides on input tensor across output dimensions
    int in_strides[] = {0, 0, 0, 0};
    int out_strides[] = {0, 0, 0, 0};

    // Prepare required data - strides on input, shapes
    for (int k = 0; k < rank; k++) {
        perm_dim[k] = cfg->perm_dim[k];
        in_shape[k] = in->shape[k];
        out_shape[k] = in->shape[cfg->perm_dim[k]];
        out_strides[k] = out->mem_stride[k];
        in_strides[k] = in->mem_stride[k];
    }
    
    for (int dim_ctr = 0; dim_ctr < MLI_MAX_RANK; dim_ctr++) {
        perm_dim_inv[perm_dim[dim_ctr]] = dim_ctr;
    }

    //if out memstride not initialized, calculate it from out_shape
    if (out_strides[0] < 1) {
        out_strides[rank - 1] = 1;
        for (int dim_ctr = rank -2; dim_ctr >=0; dim_ctr--) {
            out_strides[dim_ctr] = out_strides[dim_ctr + 1] * out_shape[dim_ctr + 1];
        }
    }

    if (in_strides[0] < 1) {
        in_strides[rank - 1] = 1;
        for (int dim_ctr = rank - 2; dim_ctr >= 0; dim_ctr--) {
            in_strides[dim_ctr] = in_strides[dim_ctr + 1] * in_shape[dim_ctr + 1];
        }
    }

    for (int dim_ctr = 0; dim_ctr < rank - 1; dim_ctr++) {
        out_strides[dim_ctr] -= out_strides[dim_ctr + 1] * out_shape[dim_ctr + 1];
    }

    printf("Test begin\n");
    printf("perm_dim: %i %i %i %i\n", perm_dim[0], perm_dim[1], perm_dim[2], perm_dim[3]);
    printf("perm_dim_inv: %i %i %i %i\n", perm_dim_inv[0], perm_dim_inv[1], perm_dim_inv[2], perm_dim_inv[3]);
    printf("out_shape: %i %i %i %i\n", out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    printf("in_shape: %i %i %i %i\n", in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    printf("in_strides: %i %i %i %i\n", in_strides[0], in_strides[1], in_strides[2], in_strides[3]);
    printf("out_strides: %i %i %i %i\n", out_strides[0], out_strides[1], out_strides[2], out_strides[3]);
    printf("input assymetric scales\n");
    if (asym) {
        if (in->el_params.sa.dim < 0){
            printf("in->el_params.sa.zero_point.mem.i16 = %d\n", in->el_params.sa.zero_point.mem.i16);
            printf("in->el_params.sa.scale.mem.i16 = %d\n", in->el_params.sa.scale.mem.i16);
            printf("in->el_params.sa.scale_frac_bits.mem.i8 = %d\n", in->el_params.sa.scale_frac_bits.mem.i8);
        } else {
            out->el_params.sa.dim = cfg->perm_dim[in->el_params.sa.dim];
            printf("cfg->perm_dim[%d] = %d\n", in->el_params.sa.dim, cfg->perm_dim[in->el_params.sa.dim]);
            for (int dim_cnt = 0; dim_cnt < in->shape[in->el_params.sa.dim]; dim_cnt++) {
                printf("in->el_params.sa.zero_point.mem.pi16[%d] = %d\n", dim_cnt, in->el_params.sa.zero_point.mem.pi16[dim_cnt]);
                printf("in->el_params.sa.scale.mem.pi16[%d] = %d\n", dim_cnt, in->el_params.sa.scale.mem.pi16[dim_cnt]);
                printf("in->el_params.sa.scale_frac_bits.mem.pi8[%d] = %d\n", dim_cnt, in->el_params.sa.scale_frac_bits.mem.pi8[dim_cnt]);
            }
        }
    }
    // Main transpose operation.
    const io_T *input = static_cast<io_T *>(in->data.mem.void_p);
    io_T *output = static_cast<io_T *>(out->data.mem.void_p);
    for (int d0_cnt = 0; d0_cnt < out_shape[0]; d0_cnt++) {
        for (int d1_cnt = 0; d1_cnt < out_shape[1]; d1_cnt++) {
            for (int d2_cnt = 0; d2_cnt < out_shape[2]; d2_cnt++) {
                for (int d3_cnt = 0; d3_cnt < out_shape[3]; d3_cnt++) {
                    int pos[] = {d0_cnt, d1_cnt, d2_cnt, d3_cnt};
                    int in_pos[] = {pos[perm_dim_inv[0]], pos[perm_dim_inv[1]], pos[perm_dim_inv[2]], pos[perm_dim_inv[3]]};
                    *output = input[in_pos[0] * in_strides[0] + in_pos[1] * in_strides[1] + in_pos[2] * in_strides[2] + in_pos[3] * in_strides[3]];
                    output += out_strides[3];
                }
                output += out_strides[2];
            }
            output += out_strides[1];
        }
        output += out_strides[0];
    }

    // Fill output tensor descr
    for (int k = 0; k < rank; k++) {
        out->shape[k] = out_shape[k];
        // out->mem_stride[k] = stride_ptr[k];
    }
    out->rank = in->rank;
    out->el_type = in->el_type;
    if (asym) {
        if (in->el_params.sa.dim < 0){
            out->el_params.sa.dim = -1;
            out->el_params.sa.zero_point.mem.i16 = in->el_params.sa.zero_point.mem.i16;
            out->el_params.sa.scale.mem.i16 = in->el_params.sa.scale.mem.i16;
            out->el_params.sa.scale_frac_bits.mem.i8 = in->el_params.sa.scale_frac_bits.mem.i8;
        } else {
            out->el_params.sa.dim = cfg->perm_dim[in->el_params.sa.dim];
            for (int dim_cnt = 0; dim_cnt < in->shape[in->el_params.sa.dim]; dim_cnt++) {
                out->el_params.sa.zero_point.mem.pi16[dim_cnt] = in->el_params.sa.zero_point.mem.pi16[dim_cnt];
                out->el_params.sa.scale.mem.pi16[dim_cnt] = in->el_params.sa.scale.mem.pi16[dim_cnt];
                out->el_params.sa.scale_frac_bits.mem.pi8[dim_cnt] = in->el_params.sa.scale_frac_bits.mem.pi8[dim_cnt];
            }
        }

        if (asym) {
            if (in->el_params.sa.dim < 0){
                printf("out->el_params.sa.zero_point.mem.i16 = %d\n", in->el_params.sa.zero_point.mem.i16);
                printf("out->el_params.sa.scale.mem.i16 = %d\n", in->el_params.sa.scale.mem.i16);
                printf("out->el_params.sa.scale_frac_bits.mem.i8 = %d\n", in->el_params.sa.scale_frac_bits.mem.i8);
            } else {
                out->el_params.sa.dim = cfg->perm_dim[in->el_params.sa.dim];
                printf("out->el_params.sa.dim = %d\n", out->el_params.sa.dim);
                for (int dim_cnt = 0; dim_cnt < in->shape[in->el_params.sa.dim]; dim_cnt++) {
                    printf("out->el_params.sa.zero_point.mem.pi16[%d] = %d\n", dim_cnt, out->el_params.sa.zero_point.mem.pi16[dim_cnt]);
                    printf("out->el_params.sa.scale.mem.pi16[%d] = %d\n", dim_cnt, out->el_params.sa.scale.mem.pi16[dim_cnt]);
                    printf("out->el_params.sa.scale_frac_bits.mem.pi8[%d] = %d\n", dim_cnt, out->el_params.sa.scale_frac_bits.mem.pi8[dim_cnt]);
                }
            }
        }
    }
    else {
        out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;
    }
    printf("Test end\n\n");
    return MLI_STATUS_OK;
}

#pragma MLI_CODE_SECTION_END()
}  // namespace ref
}  // namespace krn
}  // namespace mli

#endif  //_MLI_KRN_PERMUTE_REF_H_
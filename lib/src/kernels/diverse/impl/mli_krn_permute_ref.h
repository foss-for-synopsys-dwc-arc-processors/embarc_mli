/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PERMUTE_REF_H_
#define _MLI_KRN_PERMUTE_REF_H_

#include <stdint.h>

#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace ref {


#pragma MLI_CODE_SECTION_START(".mli_lib")

//======================================================
//
//======================================================
template <typename io_T, bool asym>
static MLI_FORCE_INLINE mli_status mli_krn_permute_run(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {

    int rank = in->rank;

    // Fill output tensor description
    out->rank = in->rank;
    out->el_type = in->el_type;

    int perm_dim[] = {0, 1, 2, 3};   // default order of output matrix dimension 4
    int perm_dim_inv[MLI_MAX_RANK];
    int out_strides[] = {0, 0, 0, 0};

    // Prepare required data - strides on input, shapes
    for (int k = 0; k < rank; k++) {
        perm_dim[k] = cfg->perm_dim[k];
        out->shape[k] = in->shape[perm_dim[k]];
        out_strides[k]= out->mem_stride[k];
    }

    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);

    for (int dim_ctr = 0; dim_ctr < MLI_MAX_RANK; dim_ctr++)
        perm_dim_inv[perm_dim[dim_ctr]] = dim_ctr;

    //if out memstride not initialized, calculate it from out_shape
    if (out_strides[0] < 1) {
        out_strides[rank - 1] = 1;
        for (int dim_ctr = rank -2; dim_ctr >=0; dim_ctr--) {
            out_strides[dim_ctr] = out_strides[dim_ctr + 1] * out->shape[dim_ctr + 1];
        }
    }

    for (int dim_ctr = 0; dim_ctr < rank - 1; dim_ctr++) {
        out_strides[dim_ctr] -= out_strides[dim_ctr + 1] * out->shape[dim_ctr + 1];
    }

    for (int i = rank; i < MLI_MAX_RANK; i++) {
        out->shape[i] = 1;
    }

    // Main transpose operation.
    const io_T *input = static_cast<io_T *>(in->data.mem.void_p);
    io_T *output = static_cast<io_T *>(out->data.mem.void_p);
    for (int d0_cnt = 0; d0_cnt < out->shape[0]; d0_cnt++) {
        for (int d1_cnt = 0; d1_cnt < out->shape[1]; d1_cnt++) {
            for (int d2_cnt = 0; d2_cnt < out->shape[2]; d2_cnt++) {
                for (int d3_cnt = 0; d3_cnt < out->shape[3]; d3_cnt++) {
                    int pos[] = {d0_cnt, d1_cnt, d2_cnt, d3_cnt};
                    int in_pos[] = {pos[perm_dim_inv[0]], pos[perm_dim_inv[1]], pos[perm_dim_inv[2]], \
                            pos[perm_dim_inv[3]]};
                    *output = input[in_pos[0] * in_prv.mem_stride[0] + in_pos[1] * in_prv.mem_stride[1] + \
                            in_pos[2] * in_prv.mem_stride[2] + in_pos[3] * in_prv.mem_stride[3]];
                    output += out_strides[3];
                }
                output += out_strides[2];
            }
            output += out_strides[1];
        }
        output += out_strides[0];
    }

    if (asym) {
        if (in->el_params.sa.dim < 0){
            out->el_params.sa.dim = -1;
            out->el_params.sa.zero_point.mem.i16 = in->el_params.sa.zero_point.mem.i16;
            out->el_params.sa.scale.mem.i16 = in->el_params.sa.scale.mem.i16;
            out->el_params.sa.scale_frac_bits.mem.i8 = in->el_params.sa.scale_frac_bits.mem.i8;
        } else {
            out->el_params.sa.dim = perm_dim[in->el_params.sa.dim];
            if(out->el_params.sa.zero_point.mem.pi16 == nullptr) {
                out->el_params.sa.zero_point.mem.pi16 = in->el_params.sa.zero_point.mem.pi16;
            } else if (out->el_params.sa.zero_point.mem.pi16 != in->el_params.sa.zero_point.mem.pi16) {
                for (int dim_cnt = 0; dim_cnt < in->shape[in->el_params.sa.dim]; dim_cnt++) 
                    out->el_params.sa.zero_point.mem.pi16[dim_cnt] = in->el_params.sa.zero_point.mem.pi16[dim_cnt];
            }

            if (out->el_params.sa.scale.mem.pi16 == nullptr) {
                out->el_params.sa.scale.mem.pi16 = in->el_params.sa.scale.mem.pi16;
            } else if (out->el_params.sa.scale.mem.pi16 != in->el_params.sa.scale.mem.pi16){
                for (int dim_cnt = 0; dim_cnt < in->shape[in->el_params.sa.dim]; dim_cnt++) 
                    out->el_params.sa.scale.mem.pi16[dim_cnt] = in->el_params.sa.scale.mem.pi16[dim_cnt];
            }

            if (out->el_params.sa.scale_frac_bits.mem.pi8 == nullptr) {
                out->el_params.sa.scale_frac_bits.mem.pi8 = in->el_params.sa.scale_frac_bits.mem.pi8;
            } else if (out->el_params.sa.scale_frac_bits.mem.pi8 != in->el_params.sa.scale_frac_bits.mem.pi8){
                for (int dim_cnt = 0; dim_cnt < in->shape[in->el_params.sa.dim]; dim_cnt++) 
                    out->el_params.sa.scale_frac_bits.mem.pi8[dim_cnt] = in->el_params.sa.scale_frac_bits.mem.pi8[dim_cnt];
            }
        }
    } else {
        out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;
    }

    return MLI_STATUS_OK;
}

#pragma MLI_CODE_SECTION_END()
}  // namespace ref
}  // namespace krn
}  // namespace mli

#endif  //_MLI_KRN_PERMUTE_REF_H_
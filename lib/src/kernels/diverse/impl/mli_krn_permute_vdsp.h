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
namespace vdsp {


#pragma MLI_CODE_SECTION_START(".mli_lib")

//======================================================
//
//======================================================
template <typename io_T, bool asym>
static void mli_krn_permute_inner(const mli_tensor *in, uint32_t *out_shape, int *out_strides,
        int *perm_dim, const MLI_PTR(io_T) input, MLI_PTR(io_T) output) {

    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);

    //Dummy load to get the number of lanes, remaining part
    auto vec = mli_prv_load_1vec(input);
    const int num_of_lanes = get_number_lanes(vec);
    const int remaining = out_shape[3] & (num_of_lanes - 1);
    const int inner_out_stride = out_strides[3];
    const int inp_stride_3 = in_prv.mem_stride[perm_dim[3]];
    const int inp_stride_2 = in_prv.mem_stride[perm_dim[2]];
    const int inp_stride_1 = in_prv.mem_stride[perm_dim[1]];
    const int inp_stride_0 = in_prv.mem_stride[perm_dim[0]];

    for (int d0_cnt = 0; d0_cnt < out_shape[0]; d0_cnt++) {
        for (int d1_cnt = 0; d1_cnt < out_shape[1]; d1_cnt++) {
            for (int d2_cnt = 0; d2_cnt < out_shape[2]; d2_cnt++) {
                int idx_inp = d0_cnt * inp_stride_0 + d1_cnt * inp_stride_1 + \
                            d2_cnt * inp_stride_2;
                
                if (remaining) {
                    if(inp_stride_3 == 1) {
                        auto input_vec = mli_prv_load_1vec(&input[idx_inp]);
                        if (inner_out_stride == 1) {
                            mli_prv_store_n_samples(output, input_vec, remaining);
                        } else {
                            mli_prv_stride_store_n_samples(output, input_vec, inner_out_stride, remaining);
                        }
                    } else {
                        auto input_vec = mli_prv_stride_load_1vec(&input[idx_inp], inp_stride_3);
                        if (inner_out_stride == 1) {
                            mli_prv_store_n_samples(output, input_vec, remaining);
                        } else {
                            mli_prv_stride_store_n_samples(output, input_vec, inner_out_stride, remaining);
                        }
                    }
                    idx_inp += remaining * inp_stride_3;
                    output += remaining * inner_out_stride;
                }

                for (int d3_cnt = remaining; d3_cnt < out_shape[3]; d3_cnt += num_of_lanes) {
                    if(inp_stride_3 == 1) {
                        auto input_vec = mli_prv_load_1vec(&input[idx_inp]);
                        if (inner_out_stride == 1) {
                                mli_prv_store_n_samples(output, input_vec);
                        } else {
                                mli_prv_stride_store_n_samples(output, input_vec, inner_out_stride);
                        }
                    } else {
                        auto input_vec = mli_prv_stride_load_1vec(&input[idx_inp], inp_stride_3);
                        if (inner_out_stride == 1) {
                                mli_prv_store_n_samples(output, input_vec);
                        } else {
                                mli_prv_stride_store_n_samples(output, input_vec, inner_out_stride);
                        }
                    }
                    idx_inp += num_of_lanes * inp_stride_3;
                    output += num_of_lanes * inner_out_stride;
                }
                output += out_strides[2];
            }
            output += out_strides[1];
        }
        output += out_strides[0];
    }
}

template <typename io_T, bool asym>
static MLI_FORCE_INLINE mli_status mli_krn_permute_run(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {

    int rank = in->rank;

    // Fill output tensor description
    out->rank = in->rank;
    out->el_type = in->el_type;

    int perm_dim[] = {0, 1, 2, 3};   // default order of output matrix dimension 4
    int out_strides[] = {0, 0, 0, 0};

    // Prepare required data - strides on input, shapes
    for (int k = 0; k < rank; k++) {
        perm_dim[k] = cfg->perm_dim[k];
        out->shape[k] = in->shape[perm_dim[k]];
        out_strides[k]= out->mem_stride[k];
    }

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

    const MLI_PTR(io_T) input = (MLI_PTR(io_T))(in->data.mem.void_p);
    MLI_PTR(io_T) output = (MLI_PTR(io_T))(out->data.mem.void_p);
    mli_krn_permute_inner<io_T, asym>(in, out->shape, out_strides, perm_dim, input, output);

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
}  // namespace vdsp
}  // namespace krn
}  // namespace mli

#endif  //_MLI_KRN_PERMUTE_REF_H_
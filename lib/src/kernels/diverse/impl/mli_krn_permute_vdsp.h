/*
* Copyright 2019-2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PERMUTE_VDSP_H_
#define _MLI_KRN_PERMUTE_VDSP_H_

#include "mli_mem_info.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"
#include "mli_prv_load_store.h"

namespace mli {
namespace krn {
namespace vdsp {


#pragma MLI_CODE_SECTION_START(".mli_lib")
template <typename io_T, bool gather, bool scatter>
static void mli_krn_permute_inner_loop(uint32_t *out_shape, int *out_increments, int* inp_stride,
        const MLI_PTR(io_T) input, MLI_PTR(io_T) output) {
    //Dummy load to get the number of lanes, remaining part
    auto vec = mli_prv_load_1vec(input);
    const int num_of_lanes = get_number_lanes(vec);
    const int remaining = out_shape[3] & (num_of_lanes - 1);
    for (int d0_cnt = 0; d0_cnt < (int)out_shape[0]; d0_cnt++) {
        for (int d1_cnt = 0; d1_cnt < (int)out_shape[1]; d1_cnt++) {
            for (int d2_cnt = 0; d2_cnt < (int)out_shape[2]; d2_cnt++) {
                int idx_inp = d0_cnt * inp_stride[0] + d1_cnt * inp_stride[1] + \
                            d2_cnt * inp_stride[2];
                
                if (remaining) {
                    if(!gather) {
                        auto input_vec = mli_prv_load_1vec(&input[idx_inp]);
                        if (!scatter) {
                            mli_prv_store_n_samples(output, input_vec, remaining);
                        } else {
                            mli_prv_stride_store_n_samples(output, input_vec, out_increments[3], remaining);
                        }
                    } else {
                        auto input_vec = mli_prv_stride_load_1vec(&input[idx_inp], inp_stride[3]);
                        if (!scatter) {
                            mli_prv_store_n_samples(output, input_vec, remaining);
                        } else {
                            mli_prv_stride_store_n_samples(output, input_vec, out_increments[3], remaining);
                        }
                    }
                    idx_inp += remaining * inp_stride[3];
                    output += remaining * out_increments[3];
                }

                for (int d3_cnt = remaining; d3_cnt < (int)out_shape[3]; d3_cnt += num_of_lanes) {
                    if(!gather) {
                        auto input_vec = mli_prv_load_1vec(&input[idx_inp]);
                        if (!scatter) {
                                mli_prv_store_n_samples(output, input_vec);
                        } else {
                                mli_prv_stride_store_n_samples(output, input_vec, out_increments[3]);
                        }
                    } else {
                        auto input_vec = mli_prv_stride_load_1vec(&input[idx_inp], inp_stride[3]);
                        if (!scatter) {
                                mli_prv_store_n_samples(output, input_vec);
                        } else {
                                mli_prv_stride_store_n_samples(output, input_vec, out_increments[3]);
                        }
                    }
                    idx_inp += num_of_lanes * inp_stride[3];
                    output += num_of_lanes * out_increments[3];
                }
                output += out_increments[2];
            }
            output += out_increments[1];
        }
        output += out_increments[0];
    }

}
template <typename io_T>
static void mli_krn_permute_calc(const mli_tensor *in, uint32_t *out_shape_, int *out_increments_,
        int *perm_dim, const MLI_PTR(io_T) input, MLI_PTR(io_T) output) {

    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);

    int inp_stride[MLI_MAX_RANK];
    uint32_t out_shape[MLI_MAX_RANK];
    int out_increments[MLI_MAX_RANK];
    int last_dim = MLI_MAX_RANK - 1;
    for (int i = 0; i < MLI_MAX_RANK; i++) {
        out_shape[i] = out_shape_[i];
        out_increments[i] = out_increments_[i];
        inp_stride[i] = in_prv.mem_stride[perm_dim[i]];
    }

    if (out_shape[last_dim] == 1) {
        // look for largest dimension and make that innerloop
        int max = out_shape[last_dim];
        int max_idx = last_dim;
        for(int i = 0; i < last_dim; i++) {
            if ((int)out_shape[i] > max) {
                max = out_shape[i];
                max_idx = i;
            }
        }
        // swap dimensions
        int tmp;
        tmp = out_shape[max_idx];
        out_shape[max_idx] = out_shape[last_dim];
        out_shape[last_dim] = tmp;
        tmp = inp_stride[max_idx];
        inp_stride[max_idx] = inp_stride[last_dim];
        inp_stride[last_dim] = tmp;
        tmp = out_increments[max_idx];
        out_increments[max_idx] = out_increments[last_dim];
        out_increments[last_dim] = tmp;
    }

    for (int dim_ctr = 0; dim_ctr < MLI_MAX_RANK - 1; dim_ctr++) {
        out_increments[dim_ctr] -= out_increments[dim_ctr + 1] * out_shape[dim_ctr + 1];
    }

    if(inp_stride[last_dim] == 1) {
        if (out_increments[last_dim] == 1) {
            mli_krn_permute_inner_loop<io_T, /*gather = */false, /*scatter =*/false>(out_shape, out_increments, inp_stride, input, output);
        } else {
            mli_krn_permute_inner_loop<io_T, /*gather = */false, /*scatter =*/true>(out_shape, out_increments, inp_stride, input, output);
        }
    } else {
        if (out_increments[last_dim] == 1) {
            mli_krn_permute_inner_loop<io_T, /*gather = */true, /*scatter =*/false>(out_shape, out_increments, inp_stride, input, output);
        } else {
            mli_krn_permute_inner_loop<io_T, /*gather = */true, /*scatter =*/true>(out_shape, out_increments, inp_stride, input, output);
        }
    }
}

#pragma MLI_CODE_SECTION_END()
}  // namespace vdsp
}  // namespace krn
}  // namespace mli

#endif  //_MLI_KRN_PERMUTE_VDSP_H_
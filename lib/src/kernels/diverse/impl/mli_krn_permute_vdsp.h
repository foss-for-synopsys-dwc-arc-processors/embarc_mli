/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PERMUTE_VDSP_H_
#define _MLI_KRN_PERMUTE_VDSP_H_

#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace vdsp {


#pragma MLI_CODE_SECTION_START(".mli_lib")

template <typename io_T>
static void mli_krn_permute_inner(const mli_tensor *in, uint32_t *out_shape, int *out_increments,
        int *perm_dim, const MLI_PTR(io_T) input, MLI_PTR(io_T) output) {

    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);

    //Dummy load to get the number of lanes, remaining part
    auto vec = mli_prv_load_1vec(input);
    const int num_of_lanes = get_number_lanes(vec);
    const int remaining = out_shape[3] & (num_of_lanes - 1);
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
                        if (out_increments[3] == 1) {
                            mli_prv_store_n_samples(output, input_vec, remaining);
                        } else {
                            mli_prv_stride_store_n_samples(output, input_vec, out_increments[3], remaining);
                        }
                    } else {
                        auto input_vec = mli_prv_stride_load_1vec(&input[idx_inp], inp_stride_3);
                        if (out_increments[3] == 1) {
                            mli_prv_store_n_samples(output, input_vec, remaining);
                        } else {
                            mli_prv_stride_store_n_samples(output, input_vec, out_increments[3], remaining);
                        }
                    }
                    idx_inp += remaining * inp_stride_3;
                    output += remaining * out_increments[3];
                }

                for (int d3_cnt = remaining; d3_cnt < out_shape[3]; d3_cnt += num_of_lanes) {
                    if(inp_stride_3 == 1) {
                        auto input_vec = mli_prv_load_1vec(&input[idx_inp]);
                        if (out_increments[3] == 1) {
                                mli_prv_store_n_samples(output, input_vec);
                        } else {
                                mli_prv_stride_store_n_samples(output, input_vec, out_increments[3]);
                        }
                    } else {
                        auto input_vec = mli_prv_stride_load_1vec(&input[idx_inp], inp_stride_3);
                        if (out_increments[3] == 1) {
                                mli_prv_store_n_samples(output, input_vec);
                        } else {
                                mli_prv_stride_store_n_samples(output, input_vec, out_increments[3]);
                        }
                    }
                    idx_inp += num_of_lanes * inp_stride_3;
                    output += num_of_lanes * out_increments[3];
                }
                output += out_increments[2];
            }
            output += out_increments[1];
        }
        output += out_increments[0];
    }
}

#pragma MLI_CODE_SECTION_END()
}  // namespace vdsp
}  // namespace krn
}  // namespace mli

#endif  //_MLI_KRN_PERMUTE_VDSP_H_
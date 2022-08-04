/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */

#ifndef _MLI_REDUCE_MAX_REF_HPP_
#define _MLI_REDUCE_MAX_REF_HPP_

#include "mli_math.h"
#include "mli_types.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"

namespace snps_arc::metaware::mli {
namespace krn {
namespace ref {


template <typename io_T>
mli_status MLI_FORCE_INLINE mli_reduce_max(const mli_tensor *in,
                                           const int8_t reduce_axis,
                                           mli_tensor *out){

    mli_prv_fx_init_dsp_ctrl();

    auto in_prv = mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);

    constexpr int kMaxSupportedRank = 4;

    MLI_ASSERT(kMaxSupportedRank <= in_prv.rank);
    MLI_ASSERT(kMaxSupportedRank <= out_prv.rank);

    // for tensors with rank less than kMaxSupportedRank, the tensor is automatically extended to the  kMaxSupportedRank but with dim = 0
    // any dim = 0 should be dim = 1 to be able to go inside it 
    for(int i = 0; i < kMaxSupportedRank; i++){
        if(0 == in_prv.shape[i]){
            in_prv.shape[i] = 1;
        }
        if(0 == out_prv.shape[i]){
            out_prv.shape[i] = 1;
        }
    }

    // swap the dimension order to make the reduce_axis in the most inner loop (last dim)
    int8_t reorder_dim[kMaxSupportedRank] = {0, 1, 2, 3}; // initial order
    reorder_dim[reduce_axis] = reorder_dim[kMaxSupportedRank-1];
    reorder_dim[kMaxSupportedRank-1] = reduce_axis;

    // loop through all input tensor elements (the inner most loop is on the reduce_axis)
    int pos_in[kMaxSupportedRank] = {0};
    int pos_out[kMaxSupportedRank] = {0};
    io_T in_val;
    io_T out_val;
    pos_out[reorder_dim[3]] = 0; 
    for (pos_in[reorder_dim[0]] = 0; pos_in[reorder_dim[0]] < in_prv.shape[reorder_dim[0]]; pos_in[reorder_dim[0]]++) {
        pos_out[reorder_dim[0]] = pos_in[reorder_dim[0]];
        for (pos_in[reorder_dim[1]] = 0; pos_in[reorder_dim[1]] < in_prv.shape[reorder_dim[1]]; pos_in[reorder_dim[1]]++) {
            pos_out[reorder_dim[1]] = pos_in[reorder_dim[1]];
            for (pos_in[reorder_dim[2]] = 0; pos_in[reorder_dim[2]] < in_prv.shape[reorder_dim[2]]; pos_in[reorder_dim[2]]++) {
                pos_out[reorder_dim[2]] = pos_in[reorder_dim[2]];
                
                // prepare the inner most loop 
                pos_in[reorder_dim[3]] = 0;
                out_val = mli_prv_tensor_read(in_prv, pos_in[0], pos_in[1],
                            pos_in[2], pos_in[3]);
                for ( ; pos_in[reorder_dim[3]] < in_prv.shape[reorder_dim[3]]; pos_in[reorder_dim[3]]++) {
                    
                    in_val = mli_prv_tensor_read(in_prv, pos_in[0], pos_in[1],
                            pos_in[2], pos_in[3]);
                    out_val = mli_math_max_fx(out_val, in_val);
                    
                }
                mli_prv_tensor_write(out_val, out_prv, pos_out[0],
                            pos_out[1], pos_out[2], pos_out[3]);
                
            }
        }
    }
    
    return MLI_STATUS_OK;
}


} // namespace ref
} // namespace krn
} // namespace snps_arc::metaware::mli

#endif // _MLI_REDUCE_MAX_REF_HPP_
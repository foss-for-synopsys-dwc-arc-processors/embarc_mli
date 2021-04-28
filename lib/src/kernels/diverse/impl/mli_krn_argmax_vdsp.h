/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef  _MLI_KRN_ARGMAX_VDSP_H_
#define  _MLI_KRN_ARGMAX_VDSP_H_

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "arc_vector.h"
#include "mli_krn_argmax_decl.h"
#include "mli_prv_load_store.h"

namespace mli {
namespace krn {
namespace vdsp {

template <typename in_T, typename out_T>
static MLI_FORCE_INLINE void inner_loop(const generic_tensor_private_t<MLI_PTR(in_T)> *src_prv,
                                        const int dim0_idx,
                                        const int dim1_idx,
                                        const int dim2_idx,
                                        int dim3_idx,
                                        const int dim3_end,
                                        const int32_t topk,
                                        MLI_OUT_PTR(out_T) dst_tensor_arr) {
    const MLI_PTR(in_T) src_arr = src_prv->ptr;
    int count = dim3_end - dim3_idx;
    auto input = mli_prv_load_nx4_samples(src_arr);
    int num_lanes = get_number_lanes(input);
    int remaining_part = count & (num_lanes - 1);
    if (remaining_part) {
        int remaining_end = dim3_idx + remaining_part;
        for (; dim3_idx < remaining_end; ++dim3_idx) {
            int src_pos = POS(src_prv, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
            if (src_arr[dst_tensor_arr[0]] > src_arr[src_pos])
                continue;
            else {
                dst_tensor_arr[0] = src_pos;
                heapify(src_arr, topk, 0, dst_tensor_arr);
            }
        }
    }

    for (; dim3_idx < dim3_end; ) {
        int src_pos = POS(src_prv, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
        auto src_val = mli_prv_load_nx4_samples(src_arr + src_pos);
        auto predicate = init_predicate(src_val >= src_arr[dst_tensor_arr[0]]);
        if (get_predicate_count(predicate) > 0) {
            uint32_t low = move_predicate_lo_to_scalar(predicate);
            vNuint_t low_v = low;
            while (low_v[0] != 0) {
                int index = mli_math_trailing_zeros(low_v)[0];
                low_v &= ~(1 << index);
                src_pos = POS(src_prv, dim0_idx, dim1_idx, dim2_idx, dim3_idx + index);
                if (src_arr[src_pos] >= src_arr[dst_tensor_arr[0]]) {
                    dst_tensor_arr[0] = (src_pos);
                    heapify(src_arr, topk, 0, dst_tensor_arr);
                }
            }

            uint32_t high = move_predicate_hi_to_scalar(predicate);
            vNuint_t high_v = high;
            while (high_v[0] != 0) {
                int index = mli_math_trailing_zeros(high_v)[0];
                high_v &= ~(1 << index);
                src_pos = POS(src_prv, dim0_idx, dim1_idx, dim2_idx, dim3_idx + index + (num_lanes / 2));
                if (src_arr[src_pos] >= src_arr[dst_tensor_arr[0]]) {
                    dst_tensor_arr[0] = (src_pos);
                    heapify(src_arr, topk, 0, dst_tensor_arr);
                }
            }
        }

        dim3_idx += num_lanes;
    }
}

}
}
}


#endif /* PUBLIC_LIB_SRC_KERNELS_DIVERSE_IMPL_MLI_KRN_ARGMAX_VDSP_H_ */

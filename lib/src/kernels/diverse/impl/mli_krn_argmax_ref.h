/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_ARGMAX_REF_H_
#define _MLI_KRN_ARGMAX_REF_H_

#include "mli_config.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace ref {

#pragma MLI_CODE_SECTION_START(".mli_lib")

template <typename in_T, typename out_T>
static MLI_FORCE_INLINE void heapify(
        const MLI_PTR(in_T) src_tensor_arr, 
        int size, 
        int root_idx, 
        MLI_OUT_PTR(out_T) dst_tensor_arr) {

    /* Initialize smalles as root */
    int smallest = root_idx; 
    while (root_idx < size / 2) {
        int l = 2 * root_idx + 1; /* left = 2*i + 1 */
        int r = 2 * root_idx + 2; /* right = 2*i + 2 */

        /* If left child is smaller than root */
        if (l < size && src_tensor_arr[dst_tensor_arr[l]] < src_tensor_arr[dst_tensor_arr[smallest]])
            smallest = l;

        /* If right child is smaller than smallest so far */
        if (r < size && src_tensor_arr[dst_tensor_arr[r]] < src_tensor_arr[dst_tensor_arr[smallest]])
            smallest = r;

        /* If smallest is not root */
        if (smallest != root_idx) {
            /* Swap smallest with root */
            out_T temp = dst_tensor_arr[root_idx];
            dst_tensor_arr[root_idx] = dst_tensor_arr[smallest];
            dst_tensor_arr[smallest] = temp;
            root_idx = smallest;
        } else break;
    }
}

/* Main function to select top k elements using modified heap sort */
template <typename in_T, typename out_T>
static MLI_FORCE_INLINE void heap_select_k(
        const generic_tensor_private_t<MLI_PTR(in_T)> *src_prv,
        const int32_t topk,
        const int argmax_dim,
        const int slice_idx,
        const int *after_kth_element_pos,
        MLI_OUT_PTR(out_T) dst_tensor_arr) {

    const MLI_PTR(in_T) src_arr = src_prv->ptr;

    /* Build heap (rearrange array) */
    for (int i = topk / 2 - 1; i >= 0; i--)
        heapify(src_arr, topk, i, dst_tensor_arr);

    int dim_start[MLI_MAX_RANK] = { 0 };
    int dim_continue[MLI_MAX_RANK] = { 0 };
    int dim_end[MLI_MAX_RANK] = { 0 };

    /* Starting searching for new values from k+1 position */
    for (int i = 0; i < MLI_MAX_RANK; ++i) {
        dim_start[i] = after_kth_element_pos[i];
    }

    /* Need to go through the entire remaining tensor */
    for (int i = 0; i < MLI_MAX_RANK; ++i) {
        dim_continue[i] = (argmax_dim == i) ? slice_idx : 0;
        dim_end[i] = (argmax_dim == i) ? slice_idx + 1 : src_prv->shape[i];
    }

    int dim0_idx, dim1_idx, dim2_idx, dim3_idx;
    for (dim0_idx = dim_start[0]; dim0_idx < dim_end[0]; ++dim0_idx) {
        for (dim1_idx = dim_start[1]; dim1_idx < dim_end[1]; ++dim1_idx) {
            for (dim2_idx = dim_start[2]; dim2_idx < dim_end[2]; ++dim2_idx) {
                for (dim3_idx = dim_start[3]; dim3_idx < dim_end[3]; ++dim3_idx) {
                    int src_pos = POS(src_prv, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
                    if (src_arr[dst_tensor_arr[0]] > src_arr[src_pos])
                        continue;
                    else {
                        dst_tensor_arr[0] = src_pos;
                        heapify(src_arr, topk, 0, dst_tensor_arr);
                    }
                }
                dim_start[3] = dim_continue[3];
            }
            dim_start[2] = dim_continue[2];
        }
        dim_start[1] = dim_continue[1];
    }
}

/* Main function to do heap sort */
template <typename in_T, typename out_T>
static MLI_FORCE_INLINE void heap_sort(const MLI_PTR(in_T) src_tensor_arr, int size, MLI_OUT_PTR(out_T) output_tensor_arr) {

    /* Build heap (rearrange array) */
    for (int i = size / 2 - 1; i >= 0; i--)
        heapify(src_tensor_arr, size, i, output_tensor_arr);

    for (int i = size - 1; i > 0; i--) {
        out_T temp = output_tensor_arr[0];
        output_tensor_arr[0] = output_tensor_arr[i];
        output_tensor_arr[i] = temp;
        heapify(src_tensor_arr, i, 0, output_tensor_arr);
    }
}


template <typename in_T, typename out_T>
static MLI_FORCE_INLINE void fill_dst_array(
        generic_tensor_private_t<MLI_PTR(in_T)> *src_prv,
        const int32_t topk,
        const int argmax_dim,
        const int slice_idx,
        int *after_kth_element_pos,
        MLI_OUT_PTR(out_T) dst_tensor_arr) {

    int dim_start[MLI_MAX_RANK] = { 0 };
    int dim_end[MLI_MAX_RANK] = { 0 };

    for (int i = 0; i < MLI_MAX_RANK; ++i) {
        dim_start[i] = (argmax_dim == i) ? slice_idx : 0;
        after_kth_element_pos[i] = dim_end[i] = (argmax_dim == i) ? slice_idx + 1 : src_prv->shape[i];
    }

    int i = 0;
    for (int dim0_idx = dim_start[0]; dim0_idx < dim_end[0]; ++dim0_idx) {
        for (int dim1_idx = dim_start[1]; dim1_idx < dim_end[1]; ++dim1_idx) {
            for (int dim2_idx = dim_start[2]; dim2_idx < dim_end[2]; ++dim2_idx) {
                for (int dim3_idx = dim_start[3]; dim3_idx < dim_end[3]; ++dim3_idx) {
                    if (i == topk) {
                        /* Saving coordinates of element which goes right after kth */
                        after_kth_element_pos[0] = dim0_idx;
                        after_kth_element_pos[1] = dim1_idx;
                        after_kth_element_pos[2] = dim2_idx;
                        after_kth_element_pos[3] = dim3_idx;
                        return;
                    }
                    dst_tensor_arr[i++] = POS(src_prv, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
                }
            }
        }
    }
}

template <typename in_T, typename out_T>
static MLI_FORCE_INLINE void argmax(const mli_tensor *src, const int32_t axis, int32_t topk, mli_tensor *dst) {

    /* Get Generic Private Tensors */
    auto src_prv = mli_prv_get_generic_tensor<MLI_PTR(in_T)>(src);
    auto dst_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(out_T)>(dst);

    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(in_T)>(&src_prv);
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(out_T)>(&dst_prv);

    const MLI_PTR(in_T) src_tensor_arr = src_prv.ptr;
    MLI_OUT_PTR(out_T) dst_tensor_arr = dst_prv.ptr;

    /* Calculating number of slices in case of per axis argmax */
    int argmax_dim = -1;
    int slices_num = 1;
    if (axis >= 0) {
        argmax_dim = axis + (MLI_MAX_RANK - src->rank);
        slices_num = src_prv.shape[argmax_dim];
    }

    for (int slice_idx = 0; slice_idx < slices_num; ++slice_idx) {
        int after_kth_element_pos[MLI_MAX_RANK] = { 0 };
        int dst_tensor_offset = topk * slice_idx;
        /* Filling dst array with right indexis considering memstrides */
        fill_dst_array(&src_prv, topk, argmax_dim, slice_idx, after_kth_element_pos, dst_tensor_arr + dst_tensor_offset);
        /* Taking top k elements from src array */
        heap_select_k(&src_prv, topk, argmax_dim, slice_idx, after_kth_element_pos, dst_tensor_arr + dst_tensor_offset);
        /* Sorting top k elements */
        heap_sort(src_tensor_arr, topk, dst_tensor_arr + dst_tensor_offset);
    }

}

template <typename in_T>
MLI_FORCE_INLINE void argmax_prepare_and_run(const mli_tensor *in, const mli_argmax_cfg *cfg, mli_tensor *out) {

    /* Setting output tensor parameters based on user mli_argmax_cfg */
    if (out->el_type == MLI_EL_FX_8 || out->el_type == MLI_EL_FX_16) {
        out->el_params.fx.frac_bits = 0;
    }
    if (out->el_type == MLI_EL_SA_8 || out->el_type == MLI_EL_SA_32) {
        out->el_params.sa.scale.mem.i16 = 1;
        out->el_params.sa.zero_point.mem.i16 = 0;
        out->el_params.sa.scale_frac_bits.mem.i8 = 0;
    }

    uint32_t dim_size = 1;
    if (cfg->axis >= 0)
        dim_size = in->shape[cfg->axis];
    out->shape[0] = dim_size;
    out->shape[1] = cfg->topk;
    out->rank = 2;

    /* Running main argmax funtion */
    if (out->el_type == MLI_EL_FX_8 || out->el_type == MLI_EL_SA_8) {
        argmax<in_T, int8_t>(in, cfg->axis, cfg->topk, out);
    } else if (out->el_type == MLI_EL_FX_16) {
        argmax<in_T, int16_t>(in, cfg->axis, cfg->topk, out);
    } else if (out->el_type == MLI_EL_SA_32) {
        argmax<in_T, int32_t>(in, cfg->axis, cfg->topk, out);
    }
}

#pragma MLI_CODE_SECTION_END()
} // namespace ref
} // namespace krn
} // namespace mli

#endif  //_MLI_KRN_ARGMAX_REF_H_

/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_MOV_REF_H_
#define _MLI_MOV_REF_H_

namespace mli {
namespace mov {
namespace ref {

template<typename io_T>
static MLI_FORCE_INLINE void mli_mov_memcpy(mli_mov_handle_t* h, const io_T* __restrict src, io_T* __restrict dst, uint32_t size,
        uint32_t out_stride, uint32_t in_stride, bool src_in_vccm, bool dst_in_vccm) {
#if USE_DMA
        // TODO program DMA
    h->state = MLI_MOV_STATE_DMA_CONFIGURED;
#else
    memcpy((void *)dst, (void *)src, size);
#endif
}


template<typename io_T>
static MLI_FORCE_INLINE void mov_inner_loop (mli_mov_handle_t* h, const io_T* __restrict src, io_T* __restrict dst,
        uint32_t inner_dst_size, uint32_t inner_src_size,
        uint32_t inner_src_strde,uint32_t inner_dst_strde,
        uint32_t inner_src_offset, uint32_t inner_dst_offset,
        uint8_t inner_pre_padding, uint8_t inner_post_padding,
        uint32_t inner_subsample, uint32_t inner_src_shape,
        bool zero_inner_loop, bool src_in_vccm, bool dst_in_vccm) {
    if (zero_inner_loop) {
        for (uint32_t inner_idx = 0; inner_idx < inner_dst_size; inner_idx++) {
            uint32_t inner_dst_pos = (inner_idx + inner_dst_offset) * inner_dst_strde;
            dst[inner_dst_pos] = 0;
        }
    } else {
        for (int inner_idx = 0; inner_idx < inner_dst_size; inner_idx++) {
            int inner_src_pos = inner_idx * inner_subsample + inner_src_offset - inner_pre_padding;
            int inner_dst_pos = (inner_idx + inner_dst_offset) * inner_dst_strde;
            if (inner_src_pos < 0 || inner_src_pos >= inner_src_shape ) {
                dst[inner_dst_pos] = 0;
            } else {
                dst[inner_dst_pos] = src[inner_src_pos * inner_src_strde];
            }

        }
    }

}

template<typename io_T>
static MLI_FORCE_INLINE void mov_inner_loop (mli_mov_handle_t* h, const io_T* __restrict src, io_T* __restrict dst,
        uint32_t inner_dst_size,
        uint32_t inner_src_strde, uint32_t inner_dst_strde,
        uint32_t inner_subsample,
        bool src_in_vccm, bool dst_in_vccm) {

        uint32_t inner_src_step = inner_subsample * inner_src_strde;
        for (int inner_idx = 0; inner_idx < inner_dst_size; inner_idx++) {
            int inner_src_pos = inner_idx * inner_src_step;
            int inner_dst_pos = inner_idx * inner_dst_strde;
            dst[inner_dst_pos] = src[inner_src_pos];

        }

}


template<typename io_T>
static MLI_FORCE_INLINE void mli_mov_prepare_run (mli_mov_handle_t* h, const mli_tensor* src, const mli_mov_cfg_t* cfg,
        mli_tensor* dst, uint32_t* dst_write_size, uint32_t* src_mem_stride, uint32_t* src_cpy_size,
        bool no_padding, bool src_in_vccm, bool dst_in_vccm) {
    int i = MLI_MAX_RANK - 1;
    uint32_t ordered_dst_write_size[4] = {1, 1, 1, 1};
    uint32_t ordered_src_shape[4] = {1, 1, 1, 1};
    uint32_t ordered_src_cpy_size[4] = {1, 1 ,1, 1};
    uint32_t ordered_dst_mem_stride[4] = {0};
    uint32_t ordered_src_mem_stride[4] = {0};
    uint8_t ordered_pre_padding[4] = {0};
    uint8_t ordered_post_padding[4] = {0};
    uint8_t ordered_pdim[4] = {0, 1, 2, 3};
    uint32_t ordered_offset[4] = {0};
    uint32_t ordered_dst_offset[4] = {0};
    uint32_t ordered_subsample[4] = {1, 1, 1, 1};
    for (int j = src->rank - 1 ; j >= 0; i--, j--) {
        ordered_dst_write_size[i]  = dst_write_size[j];
        ordered_src_shape[i] = src->shape[j];
        ordered_src_cpy_size[i] = src_cpy_size[j];
        ordered_dst_mem_stride[i] = dst->mem_stride[j];
        ordered_src_mem_stride[i] = src_mem_stride[j];
        ordered_pre_padding[i] = cfg->padding_pre[j];
        ordered_post_padding[i] = cfg->padding_post[j];
        ordered_pdim[i] = cfg->perm_dim[j] + MLI_MAX_RANK - src->rank;
        ordered_offset[i] = cfg->offset[j];
        ordered_dst_offset[i] = cfg->dst_offset[j];
        ordered_subsample[i] = cfg->sub_sample_step[j];
    }
    if (no_padding) {
        for (int pos0 = 0; pos0 < ordered_dst_write_size[0]; pos0++) {
            for (int pos1 = 0; pos1 < ordered_dst_write_size[1]; pos1++) {
                for (int pos2 = 0; pos2 < ordered_dst_write_size[2]; pos2++) {

                    int dst_pos = (pos0 + ordered_dst_offset[0]) * ordered_dst_mem_stride[0]
                                + (pos1 + ordered_dst_offset[1]) * ordered_dst_mem_stride[1]
                                + (pos2 + ordered_dst_offset[2]) * ordered_dst_mem_stride[2]
                                + ordered_dst_offset[3] * ordered_dst_mem_stride[3];

                    int src_pos0 = pos0 * ordered_subsample[ordered_pdim[0]] + ordered_offset[ordered_pdim[0]];
                    int src_pos1 = pos1 * ordered_subsample[ordered_pdim[1]] + ordered_offset[ordered_pdim[1]];
                    int src_pos2 = pos2 * ordered_subsample[ordered_pdim[2]] + ordered_offset[ordered_pdim[2]];

                    int src_pos = src_pos0 * ordered_src_mem_stride[ordered_pdim[0]]
                                + src_pos1 * ordered_src_mem_stride[ordered_pdim[1]]
                                + src_pos2 * ordered_src_mem_stride[ordered_pdim[2]]
                                + ordered_offset[ordered_pdim[3]] * ordered_src_mem_stride[ordered_pdim[3]];
                    io_T* __restrict psrc = (io_T* __restrict) src->data.mem.void_p;
                    io_T* __restrict pdst = (io_T* __restrict) dst->data.mem.void_p;
                    mli::mov::mov_inner_loop<io_T>(h, &psrc[src_pos], &pdst[dst_pos],
                             ordered_dst_write_size[3],
                             ordered_src_mem_stride[ordered_pdim[3]], ordered_dst_mem_stride[3],
                             ordered_subsample[ordered_pdim[3]],
                             src_in_vccm, dst_in_vccm);

                }
            }
        }
    } else {
          for (int pos0 = 0; pos0 < ordered_dst_write_size[0]; pos0++) {
              for (int pos1 = 0; pos1 < ordered_dst_write_size[1]; pos1++) {
                  for (int pos2 = 0; pos2 < ordered_dst_write_size[2]; pos2++) {
                      int dst_pos = (pos0 + ordered_dst_offset[0]) * ordered_dst_mem_stride[0]
                                  + (pos1 + ordered_dst_offset[1]) * ordered_dst_mem_stride[1]
                                  + (pos2 + ordered_dst_offset[2]) * ordered_dst_mem_stride[2];

                      int src_pos0 = pos0 * ordered_subsample[ordered_pdim[0]] + ordered_offset[ordered_pdim[0]] - ordered_pre_padding[ordered_pdim[0]];
                      int src_pos1 = pos1 * ordered_subsample[ordered_pdim[1]] + ordered_offset[ordered_pdim[1]] - ordered_pre_padding[ordered_pdim[1]];
                      int src_pos2 = pos2 * ordered_subsample[ordered_pdim[2]] + ordered_offset[ordered_pdim[2]] - ordered_pre_padding[ordered_pdim[2]];


                      bool in_padding_area = (((src_pos0 < 0) || (src_pos0  >= ordered_src_shape[ordered_pdim[0]])))
                                          || (((src_pos1 < 0) || (src_pos1  >= ordered_src_shape[ordered_pdim[1]])))
                                          || (((src_pos2 < 0) || (src_pos2  >= ordered_src_shape[ordered_pdim[2]])));

                      int src_pos = src_pos0 * ordered_src_mem_stride[ordered_pdim[0]]
                                  + src_pos1 * ordered_src_mem_stride[ordered_pdim[1]]
                                  + src_pos2 * ordered_src_mem_stride[ordered_pdim[2]];

                      io_T* __restrict psrc = (io_T* __restrict) src->data.mem.void_p;
                      io_T* __restrict pdst = (io_T* __restrict) dst->data.mem.void_p;
                      mli::mov::mov_inner_loop<io_T>(h, &psrc[src_pos], &pdst[dst_pos],
                               ordered_dst_write_size[3], ordered_src_cpy_size[3],
                               ordered_src_mem_stride[ordered_pdim[3]], ordered_dst_mem_stride[3],
                               ordered_offset[ordered_pdim[3]], ordered_dst_offset[3],
                               ordered_pre_padding[ordered_pdim[3]], ordered_post_padding[ordered_pdim[3]],
                               ordered_subsample[ordered_pdim[3]], ordered_src_shape[ordered_pdim[3]],
                               in_padding_area, src_in_vccm, dst_in_vccm);

                  }
              }
          }
     }


}

}
}
}

#endif //_MLI_MOV_REF_H_

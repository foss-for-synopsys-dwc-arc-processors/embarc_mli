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
static MLI_FORCE_INLINE void mli_mov_memcpy(mli_mov_handle_t* h, const io_T* src, io_T* dst, uint32_t size, uint32_t out_stride, uint32_t in_stride) {
#if USE_DMA
        // TODO program DMA
    h->state = MLI_MOV_STATE_DMA_CONFIGURED;
#else
    memcpy((void *)dst, (void *)src, size);
#endif
}


template<typename io_T>
static MLI_FORCE_INLINE void mov_inner_loop (mli_mov_handle_t* h, const io_T* src, io_T* dst,
        uint32_t inner_dst_size, uint32_t inner_src_size,
        uint32_t inner_src_strde,uint32_t inner_dst_strde,
        uint32_t inner_src_offset, uint32_t inner_dst_offset,
        uint8_t inner_pre_padding, uint8_t inner_post_padding,
        uint32_t inner_subsample, uint32_t inner_src_shape,
        bool zero_inner_loop) {
    if (zero_inner_loop) {
        for (uint32_t inner_idx = 0; inner_idx < inner_dst_size; inner_idx++) {
            uint32_t inner_dst_pos = (inner_idx + inner_dst_offset) * inner_dst_strde;
            dst[inner_dst_pos] = 0;
        }
    } else {
        for (int inner_idx = 0; inner_idx < inner_dst_size; inner_idx++) {
            int inner_src_pos = inner_idx * inner_subsample + inner_src_offset - inner_pre_padding;
            int inner_dst_pos = (inner_idx + inner_dst_offset) * inner_dst_strde;
            if(inner_src_pos < 0 || inner_src_pos >= inner_src_shape ) {
                dst[inner_dst_pos] = 0;
            } else {
                dst[inner_dst_pos] = src[inner_src_pos * inner_src_strde];
            }

        }
    }

}

}
}
}

#endif //_MLI_MOV_REF_H_

/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_MOV_DECL_H_
#define _MLI_MOV_DECL_H_

#include "mli_types.h"
#include "mli_prv_tensor.h"



#define MAX_DMA_CHAN 16

typedef enum {
    MLI_MOV_DMA_CH_NOT_USED = 0,
    MLI_MOV_DMA_CH_AVAILABLE,
    MLI_MOV_DMA_CH_IN_USE
} mli_mov_dma_ch_status_t;

typedef struct {
    int base_channel;
    int pool_size;
    mli_mov_dma_ch_status_t channel_status[MAX_DMA_CHAN];
} mli_mov_dma_pool_t;
#define MLI_MOV_DMA_POOL_INIT {0, MAX_DMA_CHAN}

typedef struct {
    void (*cb)(int32_t);
    int32_t cookie;
} mli_mov_cb_t;

namespace mli {
namespace mov {
namespace ref {

template<bool src_in_vccm, bool dst_in_vccm, bool no_inner_src_stride, bool no_inner_dst_stride>
static MLI_NO_INLINE void mli_mov_prepare_run (mli_mov_handle_t* h, const mli_tensor* src, const mli_mov_cfg_t* cfg,
        mli_tensor* dst, uint32_t * dst_write_size, int32_t * src_mem_stride, uint32_t * src_cpy_size,
        bool no_padding, int elem_size = 1);

template<typename io_T>
static MLI_FORCE_INLINE void mli_mov_memcpy(mli_mov_handle_t* h, const io_T* __restrict src, io_T* __restrict dst,
        uint32_t size, uint32_t out_stride, uint32_t in_stride, bool src_in_vccm, bool dst_in_vccm,
        bool no_inner_src_stride, bool no_inner_dst_stride, bool small_size);
template<typename io_T>
static MLI_FORCE_INLINE void mov_inner_loop (mli_mov_handle_t* h, const io_T* __restrict src, io_T* __restrict dst,
        uint32_t inner_dst_size, uint32_t inner_src_size,
        uint32_t inner_src_strde, uint32_t inner_dst_strde,
        uint32_t inner_src_offset, uint32_t inner_dst_offset,
        uint16_t inner_pre_padding, uint16_t inner_post_padding,
        uint32_t inner_subsample, uint32_t inner_src_shape,
        bool zero_inner_loop, bool src_in_vccm, bool dst_in_vccm,
        bool no_inner_src_stride, bool no_inner_dst_stride, bool small_size, io_T pad_val);
template<typename io_T>
static MLI_FORCE_INLINE void mov_inner_loop (mli_mov_handle_t* h, const io_T* __restrict src, io_T* __restrict dst,
        uint32_t inner_dst_size,
        uint32_t inner_src_strde, uint32_t inner_dst_strde,
        uint32_t inner_subsample,
        bool src_in_vccm, bool dst_in_vccm,
        bool no_inner_src_stride, bool no_inner_dst_stride, bool small_size);

}
namespace vdsp {
template<typename io_T>
   static MLI_FORCE_INLINE void mli_mov_memcpy(mli_mov_handle_t* h, const io_T* __restrict src, io_T* __restrict dst,
           uint32_t size, uint32_t out_stride, uint32_t in_stride, bool src_in_vccm, bool dst_in_vccm,
           bool no_inner_src_stride, bool no_inner_dst_stride, bool small_size);
template<typename io_T>
static MLI_FORCE_INLINE void fill_inner_dimension(io_T* __restrict p, uint32_t size, uint32_t inner_mem_stride,
        const bool dst_in_vccm, bool no_inner_dst_stride, io_T val);
template<typename io_T>
static MLI_FORCE_INLINE void mov_inner_loop (mli_mov_handle_t* h, const io_T* __restrict src, io_T* __restrict dst,
        uint32_t inner_dst_size, uint32_t inner_src_size,
        uint32_t inner_src_strde, uint32_t inner_dst_strde,
        uint32_t inner_src_offset, uint32_t inner_dst_offset,
        uint16_t inner_pre_padding, uint16_t inner_post_padding,
        uint32_t inner_subsample, uint32_t inner_src_shape,
        bool zero_inner_loop, bool src_in_vccm, bool dst_in_vccm,
        bool no_inner_src_stride, bool no_inner_dst_stride, bool small_size, io_T pad_val);
template<typename io_T>
static MLI_FORCE_INLINE void mov_inner_loop (mli_mov_handle_t* h, const io_T* src, io_T* dst,
        uint32_t inner_dst_size,
        uint32_t inner_src_strde, uint32_t inner_dst_strde,
        uint32_t inner_subsample,
        bool src_in_vccm, bool dst_in_vccm,
        bool no_inner_src_stride, bool no_inner_dst_stride, bool small_size);
}
}
}

#endif //_MLI_MOV_PRIVATE_TYPES_H_

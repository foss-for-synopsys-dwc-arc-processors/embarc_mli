/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_MOV_VDSP_H_
#define _MLI_MOV_VDSP_H_

#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_prv_load_store.h"
#include "mli_mov_decl.h"
namespace mli {
namespace mov {
namespace vdsp {

const int unroll_factor = 4;
static MLI_FORCE_INLINE void mli_mov_from_cache_to_vccm(const int8_t* __restrict src, int8_t* __restrict dst,
        uint32_t size) {
	//check if the pointers are aligned to 32-bit word
    if ((((uint32_t)src & 3) == 0) && (((uint32_t)dst & 3) == 0)) {
        int remaining = size & (_VDSP_NUM_8BIT_LANES - 1);
        size -= remaining;
        int unroll_size = ((size / _VDSP_NUM_8BIT_LANES) & (unroll_factor - 1)) * _VDSP_NUM_8BIT_LANES;
        size -= unroll_size;
        remaining += unroll_size;
        int32_t* __restrict aligned_src_ptr = (int32_t*)((int32_t)src);
        MLI_PTR(int32_t) __restrict aligned_dst_ptr = (MLI_PTR(int32_t))((int32_t)dst);
        //adjust the size to be the number of words
        size = size / sizeof(int32_t);
        int idx = 0;
#pragma clang loop pipeline(pipeline_parallel_loop)
        for ( ;idx < size; idx += _VDSP_NUM_32BIT_LANES * unroll_factor) {
            auto src_v = mli_prv_load_1vec(&aligned_src_ptr[idx]);
            auto src_v1 = mli_prv_load_1vec(&aligned_src_ptr[idx + _VDSP_NUM_32BIT_LANES]);
            auto src_v2 = mli_prv_load_1vec(&aligned_src_ptr[idx + 2 * _VDSP_NUM_32BIT_LANES]);
            auto src_v3 = mli_prv_load_1vec(&aligned_src_ptr[idx + 3 * _VDSP_NUM_32BIT_LANES]);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx], src_v);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx + _VDSP_NUM_32BIT_LANES], src_v1);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx + 2 * _VDSP_NUM_32BIT_LANES], src_v2);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx + 3 * _VDSP_NUM_32BIT_LANES], src_v3);

        }
        if (remaining) {
            memcpy((void*)(&aligned_dst_ptr[idx]), (void*)(&aligned_src_ptr[idx]), remaining);
        }
      //check that pointers have the same offset from the alignment
    } else if ((((uint32_t)src & 3) == (((uint32_t)dst & 3)))){
        int align_size = (uint32_t)src & 3;
        memcpy((void*)dst, (void*)src, align_size);
        size -= align_size;
        int remaining = size & (_VDSP_NUM_8BIT_LANES - 1);
        size -= remaining;
        int unroll_size = ((size / _VDSP_NUM_8BIT_LANES) & (unroll_factor - 1)) * _VDSP_NUM_8BIT_LANES;
        size -= unroll_size;
        remaining += unroll_size;
        int32_t* __restrict aligned_src_ptr = (int32_t*)((int32_t)&src[align_size]);
        MLI_PTR(int32_t) __restrict aligned_dst_ptr = (MLI_PTR(int32_t))((int32_t)&dst[align_size]);
        size = size / sizeof(int32_t);
        int idx = 0;
#pragma clang loop pipeline(pipeline_parallel_loop)
        for ( ;idx < size; idx += _VDSP_NUM_32BIT_LANES * unroll_factor) {
            auto src_v = mli_prv_load_1vec(&aligned_src_ptr[idx]);
            auto src_v1 = mli_prv_load_1vec(&aligned_src_ptr[idx + _VDSP_NUM_32BIT_LANES]);
            auto src_v2 = mli_prv_load_1vec(&aligned_src_ptr[idx + 2 * _VDSP_NUM_32BIT_LANES]);
            auto src_v3 = mli_prv_load_1vec(&aligned_src_ptr[idx + 3 * _VDSP_NUM_32BIT_LANES]);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx], src_v);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx + _VDSP_NUM_32BIT_LANES], src_v1);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx + 2 * _VDSP_NUM_32BIT_LANES], src_v2);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx + 3 * _VDSP_NUM_32BIT_LANES], src_v3);
        }
        if (remaining) {
            memcpy((void*)(&aligned_dst_ptr[idx]), (void*)(&aligned_src_ptr[idx]), remaining);
        }

    } else {
        memcpy((void*)dst, (void*)src, size);
    }
}

static MLI_FORCE_INLINE void mli_mov_from_vccm_to_cache(const int8_t* __restrict src, int8_t* __restrict dst,
        uint32_t size) {
	//check if the pointers are aligned to 32-bit word
    if ((((uint32_t)src & 3) == 0) && (((uint32_t)dst & 3) == 0)) {
        int remaining = size & (_VDSP_NUM_8BIT_LANES - 1);
        size -= remaining;
        int unroll_size = ((size / _VDSP_NUM_8BIT_LANES) & (unroll_factor - 1)) * _VDSP_NUM_8BIT_LANES;
        size -= unroll_size;
        remaining += unroll_size;
        MLI_PTR(int32_t) __restrict aligned_src_ptr = (MLI_PTR(int32_t))((uint32_t)src);
        int32_t* __restrict aligned_dst_ptr = (int32_t*)((uint32_t)dst);
        size = size / sizeof(int32_t);
        int idx = 0;
#pragma clang loop pipeline(pipeline_parallel_loop)
        for ( ;idx < size; idx += _VDSP_NUM_32BIT_LANES * unroll_factor) {
            auto src_v = mli_prv_load_1vec(&aligned_src_ptr[idx]);
             auto src_v1 = mli_prv_load_1vec(&aligned_src_ptr[idx + _VDSP_NUM_32BIT_LANES]);
             auto src_v2 = mli_prv_load_1vec(&aligned_src_ptr[idx + 2 * _VDSP_NUM_32BIT_LANES]);
             auto src_v3 = mli_prv_load_1vec(&aligned_src_ptr[idx + 3 * _VDSP_NUM_32BIT_LANES]);
             mli_prv_store_n_samples(&aligned_dst_ptr[idx], src_v);
             mli_prv_store_n_samples(&aligned_dst_ptr[idx + _VDSP_NUM_32BIT_LANES], src_v1);
             mli_prv_store_n_samples(&aligned_dst_ptr[idx + 2 * _VDSP_NUM_32BIT_LANES], src_v2);
             mli_prv_store_n_samples(&aligned_dst_ptr[idx + 3 * _VDSP_NUM_32BIT_LANES], src_v3);
        }
        if (remaining) {
            memcpy((void*)(&aligned_dst_ptr[idx]), (void*)(&aligned_src_ptr[idx]), remaining);
        }
     //check that pointers have the same offset from the alignment
    } else if ((((uint32_t)src & 3) == (((uint32_t)dst & 3)))){
        int align_size = (uint32_t)src & 3;
        memcpy((void*)dst, (void*)src, align_size);
        size -= align_size;
        int remaining = size & (_VDSP_NUM_8BIT_LANES - 1);
        size -= remaining;
        int unroll_size = ((size / _VDSP_NUM_8BIT_LANES) & (unroll_factor - 1)) * _VDSP_NUM_8BIT_LANES;
        size -= unroll_size;
        remaining += unroll_size;
        MLI_PTR(int32_t) __restrict aligned_src_ptr = (MLI_PTR(int32_t))((int32_t)&src[align_size]);
        int32_t* __restrict aligned_dst_ptr = (int32_t*)((int32_t)&dst[align_size]);
        size = size / sizeof(int32_t);
        int idx = 0;
#pragma clang loop pipeline(pipeline_parallel_loop)
        for ( ;idx < size; idx += _VDSP_NUM_32BIT_LANES * unroll_factor) {
            auto src_v = mli_prv_load_1vec(&aligned_src_ptr[idx]);
            auto src_v1 = mli_prv_load_1vec(&aligned_src_ptr[idx + _VDSP_NUM_32BIT_LANES]);
            auto src_v2 = mli_prv_load_1vec(&aligned_src_ptr[idx + 2 * _VDSP_NUM_32BIT_LANES]);
            auto src_v3 = mli_prv_load_1vec(&aligned_src_ptr[idx + 3 * _VDSP_NUM_32BIT_LANES]);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx], src_v);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx + _VDSP_NUM_32BIT_LANES], src_v1);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx + 2 * _VDSP_NUM_32BIT_LANES], src_v2);
            mli_prv_store_n_samples(&aligned_dst_ptr[idx + 3 * _VDSP_NUM_32BIT_LANES], src_v3);
        }
        if (remaining) {
            memcpy((void*)(&aligned_dst_ptr[idx]), (void*)(&aligned_src_ptr[idx]), remaining);
        }
    } else {
        memcpy((void*)dst, (void*)src, size);
    }
}


template<typename io_T>
static MLI_FORCE_INLINE void mli_mov_memcpy(mli_mov_handle_t* h, const io_T* __restrict src, io_T* __restrict dst, uint32_t size,
        uint32_t out_stride, uint32_t in_stride, bool src_in_vccm, bool dst_in_vccm,
        bool no_inner_src_stride, bool no_inner_dst_stride, bool small_size) {
#if USE_DMA
    // TODO program DMA
    h->state = MLI_MOV_STATE_DMA_CONFIGURED;
#else
    if (src_in_vccm && dst_in_vccm) {
        int idx_src = 0;
        int idx_dst = 0;
        MLI_PTR(io_T) __restrict dst_ptr = (MLI_PTR(io_T))dst;
        MLI_PTR(io_T) __restrict src_ptr = (MLI_PTR(io_T))src;
        //Dummy load to determine the number of lanes
        auto vec = mli_prv_load_1vec(src_ptr);
        int num_of_lanes = get_number_lanes(vec);
        int remaining = size & (num_of_lanes - 1);
        if (remaining) {
            if(no_inner_src_stride) {
                auto src_v = mli_prv_load_1vec(&src_ptr[idx_src]);
                if (no_inner_dst_stride) {
                       mli_prv_store_n_samples(&dst_ptr[idx_dst], src_v, remaining);
                } else {
                       mli_prv_stride_store_n_samples(&dst_ptr[idx_dst], src_v, out_stride, remaining);
                }
            } else {
                auto src_v = mli_prv_stride_load_1vec(&src_ptr[idx_src], in_stride);
                if (no_inner_dst_stride) {
                    mli_prv_store_n_samples(&dst_ptr[idx_dst], src_v, remaining);
                } else {
                          mli_prv_stride_store_n_samples(&dst_ptr[idx_dst], src_v, out_stride, remaining);

                }
            }
            idx_src += remaining * in_stride;
            idx_dst += remaining * out_stride;
        }
        for (int pos = remaining; pos < size; pos += num_of_lanes) {
            if(no_inner_src_stride) {
                auto src_v = mli_prv_load_1vec(&src_ptr[idx_src]);
                if (no_inner_dst_stride) {
                       mli_prv_store_n_samples(&dst_ptr[idx_dst], src_v);
                } else {
                       mli_prv_stride_store_n_samples(&dst_ptr[idx_dst], src_v, out_stride);
                }
            } else {
                auto src_v = mli_prv_stride_load_1vec(&src_ptr[idx_src], in_stride);
                if (no_inner_dst_stride) {
                       mli_prv_store_n_samples(&dst_ptr[idx_dst], src_v);
                } else {
                       mli_prv_stride_store_n_samples(&dst_ptr[idx_dst], src_v, out_stride);
                }
            }
            idx_src += num_of_lanes * in_stride;
            idx_dst += num_of_lanes * out_stride;
        }
    } else if (dst_in_vccm) {
        if (no_inner_src_stride && no_inner_dst_stride) {
            if (small_size) {
                memcpy((void*)dst, (void*)src, size * sizeof(io_T));
            } else {
                mli_mov_from_cache_to_vccm((int8_t*)src, (int8_t*)dst, size * sizeof(io_T));
            }
        } else {
            int idx_src = 0;
            int idx_dst = 0;
            io_T* __restrict src_ptr = (io_T*)src;
            MLI_PTR(io_T) __restrict dst_ptr = (MLI_PTR(io_T))dst;
            //Dummy load to determine the number of lanes
            auto vec = mli_prv_load_1vec(dst_ptr);
            int num_of_lanes = get_number_lanes(vec);
            int remaining = size & (num_of_lanes - 1);
            if (remaining) {
                auto src_v = mli_prv_stride_load_1vec(&src_ptr[idx_src], in_stride, remaining);
                if (no_inner_dst_stride) {
                          mli_prv_store_n_samples(&dst_ptr[idx_dst], src_v, remaining);
                } else {
                          mli_prv_stride_store_n_samples(&dst_ptr[idx_dst], src_v, out_stride, remaining);
                }
                idx_src += remaining * in_stride;
                idx_dst += remaining * out_stride;
            }
            for (int pos = remaining; pos < size; pos += num_of_lanes) {
                auto src_v = mli_prv_stride_load_1vec(&src_ptr[idx_src], in_stride);
                if (no_inner_dst_stride) {
                       mli_prv_store_n_samples(&dst_ptr[idx_dst], src_v);
                } else {
                       mli_prv_stride_store_n_samples(&dst_ptr[idx_dst], src_v, out_stride);
                }
                idx_src += num_of_lanes * in_stride;
                idx_dst += num_of_lanes * out_stride;
            }
        }
    } else if (src_in_vccm) {
        if (no_inner_src_stride &&no_inner_dst_stride) {
            if (small_size) {
                memcpy((void*)dst, (void*)src, size * sizeof(io_T));
            } else {
            mli_mov_from_vccm_to_cache((int8_t*)src, (int8_t*)dst, size * sizeof(io_T));
            }
        } else {
             int idx_src = 0;
             int idx_dst = 0;
             io_T* __restrict dst_ptr = (io_T*)dst;
             MLI_PTR(io_T) __restrict src_ptr = (MLI_PTR(io_T))src;
             //Dummy load to determine the number of lanes
             auto vec = mli_prv_load_1vec(src_ptr);
             int num_of_lanes = get_number_lanes(vec);
             int remaining = size & (num_of_lanes - 1);
             if (remaining) {
                 if(in_stride == 1) {
                     auto src_v = mli_prv_load_1vec(&src_ptr[idx_src]);
                     mli_prv_stride_store_n_samples(&dst_ptr[idx_dst], src_v, out_stride, remaining);
                 } else {
                     auto src_v = mli_prv_stride_load_1vec(&src_ptr[idx_src], in_stride, remaining);
                     mli_prv_stride_store_n_samples(&dst_ptr[idx_dst], src_v, out_stride, remaining);
                 }
                 idx_src += remaining * in_stride;
                 idx_dst += remaining * out_stride;
             }
             for (int pos = remaining; pos < size; pos += num_of_lanes) {
                 if(in_stride == 1) {
                     auto src_v = mli_prv_load_1vec(&src_ptr[idx_src]);
                     mli_prv_stride_store_n_samples(&dst_ptr[idx_dst], src_v, out_stride);
                 } else {
                     auto src_v = mli_prv_stride_load_1vec(&src_ptr[idx_src], in_stride);
                     mli_prv_stride_store_n_samples(&dst_ptr[idx_dst], src_v, out_stride);
                 }
                 idx_src += num_of_lanes * in_stride;
                 idx_dst += num_of_lanes * out_stride;
             }
        }
    } else {
        if (no_inner_src_stride && (no_inner_dst_stride)) {
            memcpy((void*)dst, (void*)src, size);
        } else {
            for (int i = 0; i < size; i++) {
                int src_idx = i * in_stride;
                int dst_idx = i * out_stride;
                dst[dst_idx] = src[src_idx];
            }
        }

    }
#endif
}




template<typename io_T>
static MLI_FORCE_INLINE void fill_inner_dimension_by_zeros(io_T* __restrict p, uint32_t size, uint32_t out_stride,
        const bool dst_in_vccm, bool no_inner_dst_stride) {
    if (dst_in_vccm) {
        //Dummy load to determine the number of lanes
        MLI_PTR(io_T)p_v = (MLI_PTR(io_T))p;
        auto vec = mli_prv_load_1vec(p_v);
        int num_of_lanes = get_number_lanes(vec);
        int remaining = size & (num_of_lanes - 1);
        int dst_idx = 0;
        if (remaining) {
            if (no_inner_dst_stride) {
                mli_prv_store_n_samples(&p_v[dst_idx], (decltype(vec))0, remaining);
            } else {
                mli_prv_stride_store_n_samples(&p_v[dst_idx], (decltype(vec))0, out_stride, remaining);
            }
            dst_idx += remaining * out_stride;
        }
        for (int pos = remaining; pos < size; pos += num_of_lanes) {
            if (no_inner_dst_stride) {
                mli_prv_store_n_samples(&p_v[dst_idx], (decltype(vec))0);
            } else {
                mli_prv_stride_store_n_samples(&p_v[dst_idx], (decltype(vec))0, out_stride);
            }
            dst_idx += num_of_lanes * out_stride;
        }

    } else {
        for (int i = 0; i < size; i++) {
            int dst_idx = i * out_stride;
            p[dst_idx] = 0;
        }
    }
}

template<typename io_T>
static MLI_FORCE_INLINE void mov_inner_loop (mli_mov_handle_t* h, const io_T* __restrict src, io_T* __restrict dst,
        uint32_t inner_dst_size, uint32_t inner_src_size,
        uint32_t inner_src_strde, uint32_t inner_dst_strde,
        uint32_t inner_src_offset, uint32_t inner_dst_offset,
        uint8_t inner_pre_padding, uint8_t inner_post_padding,
        uint32_t inner_subsample, uint32_t inner_src_shape,
        bool zero_inner_loop, bool src_in_vccm, bool dst_in_vccm,
        bool no_inner_src_stride, bool no_inner_dst_stride, bool small_size) {
    if (zero_inner_loop) {
        fill_inner_dimension_by_zeros<io_T>(dst, inner_dst_size, inner_dst_strde,
                dst_in_vccm, no_inner_dst_stride);
    } else {
        int inner_dst_pos =  inner_dst_offset * inner_dst_strde;
        int inner_src_pos = inner_src_offset - inner_pre_padding;
        uint32_t pre_padding_size = 0;
        //check if there will be pre_padding in inner dimension
        if(inner_src_pos < 0) {
            pre_padding_size += CEIL_DIV(-inner_src_pos, inner_subsample);
            fill_inner_dimension_by_zeros<io_T>(&dst[inner_dst_pos], pre_padding_size, inner_dst_strde,
                    dst_in_vccm, no_inner_dst_stride);
            inner_src_pos += pre_padding_size * inner_subsample;
            inner_dst_pos += pre_padding_size * inner_dst_strde;
        }

        uint32_t size_of_copy = CEIL_DIV((inner_src_size + inner_src_offset - inner_src_pos), inner_subsample);
        inner_src_pos *= inner_src_strde;
        uint32_t inner_src_step = inner_src_strde * inner_subsample;
        mli::mov::mli_mov_memcpy<io_T>(h, &src[inner_src_pos], &dst[inner_dst_pos], size_of_copy, inner_dst_strde,
                        inner_src_step, src_in_vccm, dst_in_vccm, no_inner_src_stride , no_inner_dst_stride, small_size);
        inner_dst_pos += size_of_copy * inner_dst_strde;
        inner_src_pos += size_of_copy * inner_src_step;

        if (inner_src_pos >= inner_src_shape) {
            uint32_t post_padding_size = inner_dst_size - pre_padding_size - size_of_copy;
            fill_inner_dimension_by_zeros<io_T>(&dst[inner_dst_pos], post_padding_size, inner_dst_strde,
                    dst_in_vccm, no_inner_dst_stride);
        }
    }
}

template<typename io_T>
static MLI_FORCE_INLINE void mov_inner_loop (mli_mov_handle_t* h, const io_T* __restrict src, io_T* __restrict dst,
        uint32_t inner_dst_size,
        uint32_t inner_src_strde, uint32_t inner_dst_strde,
        uint32_t inner_subsample,
        bool src_in_vccm, bool dst_in_vccm,
        bool no_inner_src_stride, bool no_inner_dst_stride, bool small_size) {

        uint32_t inner_src_step = inner_subsample * inner_src_strde;
        mli::mov::mli_mov_memcpy<io_T>(h, src, dst, inner_dst_size, inner_dst_strde,
                        inner_src_step, src_in_vccm, dst_in_vccm, no_inner_src_stride, no_inner_dst_stride, small_size);

 }



}
}
}

#endif //_MLI_MOV_VDSP_H_

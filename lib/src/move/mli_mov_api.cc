/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <stdbool.h>
#include <string.h>
#include "mli_api.h"
#include "mli_debug.h"
#include "mli_math_macros.h"
#include "mli_mem_info.h"
#include "mli_mov_api.h"
#include "mli_mov.h"
#include "mli_types.h"
#include "mli_check.h"
#include "mli_prv_load_store.h"
#include "mli_prv_tensor.h"
#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

// singleton for dma channel pool
static mli_mov_dma_pool_t dma_pool = MLI_MOV_DMA_POOL_INIT;

// singleton for callback functions
static mli_mov_cb_t callbacktable[MAX_DMA_CHAN] = {{0}};

//=====================================================================
// Public functions
//=====================================================================

//---------------------------------------------------------------------
// Synchronous data movement functions
//---------------------------------------------------------------------

/** 
 * @brief Synchronous copy from src to dst
 *
 * @detail This function will perform a data copy from the src tensor to the dst tensor
 * according to the settings in the cfg struct.
 * It assumes the destination tensor contains a valid pointer to a large enough buffer.
 * the size of this buffer is specified in the capacity field of the dst tensor.
 * the other fields of the dst tensor will be filled by the copy function.
 *
 * The function will return once the complete data transfer is finished.
 */
mli_status mli_mov_tensor_sync(const mli_tensor* src, const mli_mov_cfg_t* cfg, mli_tensor* dst) {
    mli_status retval = MLI_STATUS_OK;
    mli_mov_handle_t h = {0};
#if USE_DMA
    int num_dma_channels = 1;
#else
    int num_dma_channels = 0;
#endif

    if (retval == MLI_STATUS_OK)
        retval = mli_mov_acquire_handle(num_dma_channels, &h);
    if (retval == MLI_STATUS_OK)
        retval = mli_mov_prepare(&h, src, cfg, dst);
    if (retval == MLI_STATUS_OK)
        retval = mli_mov_start(&h, src, cfg, dst);
    if (retval == MLI_STATUS_OK)
        retval = mli_mov_wait(&h);

    if (retval == MLI_STATUS_OK) {
        retval = mli_mov_release_handle(&h);
    }else {
        // in case of a failure we still need to release the resources
        // but we want to report the failure, and not the result of
        // the release function.
        mli_mov_release_handle(&h);
    }
    return retval;
}

//---------------------------------------------------------------------
// Asynchronous data movement functions
//---------------------------------------------------------------------

/** 
 * @brief Prepare asynchronous copy from src to dst
 *
 * @detail This function will prepare a data copy from the src tensor to the dst tensor
 * according to the settings in the cfg struct.
 * It assumes the destination tensor contains a valid pointer to a large enough buffer.
 * the size of this buffer is specified in the capacity field of the dst tensor.
 * the other fields of the dst tensor will be filled by the copy function.
 *
 * The function returns after the transfer has been prepared.
 * In case the DMA is used, this function will not start the DMA transfer.
 */

mli_status mli_mov_prepare(mli_mov_handle_t* h, const mli_tensor* src, const mli_mov_cfg_t* cfg, mli_tensor* dst) {
    mli_status retval = MLI_STATUS_OK;
    int32_t src_mem_stride[MLI_MAX_RANK] = {0};
    // check tensor, check if handle is valid
    MLI_ASSERT(h != NULL);
    MLI_ASSERT(h->state == MLI_MOV_STATE_OPEN || h->state == MLI_MOV_STATE_DONE);
    retval = MLI_CHECK_STATUS(mli_chk_data_movement(src,cfg,dst),__func__);
    if (retval != MLI_STATUS_OK) return retval;
    // copy tensor parameters from source to destination and compute missing parameters.
    int rank = dst->rank = src->rank;
    dst->el_type = src->el_type;

    const bool src_in_vccm = mli_mem_is_inside_vccm(mli_prv_tensor_cast_data_ptr(src));
    const bool dst_in_vccm = mli_mem_is_inside_vccm(mli_prv_tensor_cast_data_ptr(dst));

    if ((src->el_type == MLI_EL_SA_8 || src->el_type == MLI_EL_SA_32) && (src->el_params.sa.dim != -1)) {
        if ((dst->el_params.sa.scale.mem.pi16 != src->el_params.sa.scale.mem.pi16) &&
                (dst->el_params.sa.scale.mem.pi16 != NULL)) {
            mli::mov::mli_mov_memcpy<int8_t>(h, src->el_params.sa.scale.mem.pi8,
                        dst->el_params.sa.scale.mem.pi8, src->el_params.sa.scale.capacity, 1, 1, false, false, true, true, false);
            mli::mov::mli_mov_memcpy<int8_t>(h, src->el_params.sa.zero_point.mem.pi8,
                        dst->el_params.sa.zero_point.mem.pi8, src->el_params.sa.zero_point.capacity, 1, 1, false, false, true, true, false);
            mli::mov::mli_mov_memcpy<int8_t>(h, src->el_params.sa.scale_frac_bits.mem.pi8,
                        dst->el_params.sa.scale_frac_bits.mem.pi8, src->el_params.sa.scale_frac_bits.capacity, 1, 1, false, false, true, true, false);
        } else {
            dst->el_params.sa.scale = src->el_params.sa.scale;
            dst->el_params.sa.zero_point = src->el_params.sa.zero_point;
            dst->el_params.sa.scale_frac_bits = src->el_params.sa.scale_frac_bits;
        }
        dst->el_params.sa.dim = src->el_params.sa.dim;
    } else {
        dst->el_params = src->el_params;
    }

    src_mem_stride[rank - 1] = src->mem_stride[rank - 1] != 0 ? src->mem_stride[rank - 1] : 1;
    for (int i = rank - 2; i >= 0; i--) {
        src_mem_stride[i] = src->mem_stride[i] != 0 ? src->mem_stride[i] : src_mem_stride[i+1] * src->shape[i+1];
    }

    const uint8_t* pdim = cfg->perm_dim;
    uint32_t dst_write_size[4] = {1, 1, 1, 1};
    uint32_t src_cpy_size[4] = {1, 1, 1, 1};
    for (int i = 0; i < rank; i++) {
        src_cpy_size[i] = cfg->size[pdim[i]] > 0 ? cfg->size[pdim[i]] : (src->shape[pdim[i]] - cfg->offset[pdim[i]]);
        dst_write_size[i] = CEIL_DIV((src_cpy_size[i] + cfg->padding_pre[pdim[i]] + cfg->padding_post[pdim[i]]),
                cfg->sub_sample_step[pdim[i]]);
        dst->shape[i] = dst_write_size[i] + cfg->dst_offset[i];
    }
    for (int i = rank; i < MLI_MAX_RANK; i++) {
        dst->shape[i] = 1;
        dst->mem_stride[i] = 0;
    }

    /* if destination memstride is provided in the configuration, use it.
       if not, check if the output tensor provides a mem stride.
       when no memstride is provided at all, compute the memstride based on the destination shape */
    if (cfg->dst_mem_stride[rank - 1] != 0) {
        dst->mem_stride[rank - 1] = cfg->dst_mem_stride[rank - 1];
    } else if (dst->mem_stride[rank - 1] == 0) {
        dst->mem_stride[rank - 1] = 1;
    }
    for (int i = rank - 2; i >= 0; i--) {
        if (cfg->dst_mem_stride[i] != 0) {
            dst->mem_stride[i] = cfg->dst_mem_stride[i];
        } else if (dst->mem_stride[i] == 0) {
            dst->mem_stride[i] = dst->mem_stride[i + 1] * dst->shape[i + 1];
        }
    }
    retval = MLI_CHECK_STATUS(mli_chk_data_movement_dst_tensor(dst), __func__);
    if (retval != MLI_STATUS_OK) {
        return retval;
    }
    // update state in the handle
    h->state = MLI_MOV_STATE_PREPARED;

    // copy tensor data, first check if it can be done in a single transfer.
    bool is_possible_in_single1d_transfer = true;
    bool no_padding = true;
    int32_t stride = 1;
    for (int i = rank - 1; i >=0; i--) {
        // for a single 1d copy all data needs to be continuous in memory
        // this means that the mem_stride of both source and destination
        // needs to match the product of the shape.
        // this also means that the shape of src and dst needs to be the same.
        is_possible_in_single1d_transfer &= (src_mem_stride[i] == stride) && (dst->mem_stride[i] == stride);
        is_possible_in_single1d_transfer &= (src->shape[i] == dst->shape[i]);
        no_padding &= !(cfg->padding_pre[i] || cfg->padding_post[i]);
        is_possible_in_single1d_transfer &= no_padding;
        is_possible_in_single1d_transfer &= (cfg->perm_dim[i] == i);
        stride *= src->shape[i];
    }

    if (is_possible_in_single1d_transfer) {
        int copy_size = mli_hlp_count_elem_num(src, 0);
        copy_size *= mli_hlp_tensor_element_size(src);
        mli::mov::mli_mov_memcpy<int8_t>(h, src->data.mem.pi8, dst->data.mem.pi8, copy_size, 1, 1, src_in_vccm, dst_in_vccm, true, true, false);

    } else {
        bool no_inner_src_stride = ((src_mem_stride[cfg->perm_dim[rank-1]] * cfg->sub_sample_step[cfg->perm_dim[rank-1]]) == 1);
        bool no_inner_dst_stride = (dst->mem_stride[rank-1] == 1);
        uint32_t elem_size = mli_hlp_tensor_element_size(src);

        if (no_inner_src_stride && no_inner_dst_stride) {
            if (src_in_vccm && dst_in_vccm) {
                mli::mov::mli_mov_prepare_run<true, true, true, true>(h, src, cfg, dst, dst_write_size, src_mem_stride, src_cpy_size,
                        no_padding, elem_size);
            } else {
                mli::mov::mli_mov_prepare_run<false, false, true, true>(h, src, cfg, dst, dst_write_size, src_mem_stride, src_cpy_size,
                        no_padding, elem_size);
            }
        } else {
            if (src_in_vccm && dst_in_vccm) {
                mli::mov::mli_mov_prepare_run<true, true, false, false>(h, src, cfg, dst, dst_write_size, src_mem_stride, src_cpy_size,
                        no_padding, elem_size);
            } else {
                mli::mov::mli_mov_prepare_run<false, false, false, false>(h, src, cfg, dst, dst_write_size, src_mem_stride, src_cpy_size,
                        no_padding, elem_size);
            }
        }
    }
    return retval;
}

/** 
 * @brief Register a callback for a datatransfer
 *
 * @detail This function will register a callback function that will be called after
 * the data transfer has been completed.
 * The callback function takes one parameter, and the value of cookie is passed to the callback function.
 */
mli_status mli_mov_registercallback(mli_mov_handle_t* h, void (*cb)(int32_t), int32_t cookie) {
    MLI_ASSERT(h != NULL);
    MLI_ASSERT(h->dma_ch < MAX_DMA_CHAN);
    // store the callback function ptr.
    callbacktable[h->dma_ch].cb = cb;
    callbacktable[h->dma_ch].cookie = cookie;
    // register isr to the DMA, and only call the cb once all channels are done
    // TODO
    return MLI_STATUS_OK;
}

/** 
 * @brief Start asynchronous copy from src to dst
 *
 * @detail This function will start the data copy from the src tensor to the dst tensor
 * as prepared by the prepare function. (and checks this)
 *
 * The function returns after the transfer has been started.
 */
mli_status mli_mov_start(mli_mov_handle_t* h, const mli_tensor* src, const mli_mov_cfg_t* cfg, mli_tensor* dst) {
    MLI_ASSERT(h != NULL);

#if USE_DMA
    if (h->state == MLI_MOV_STATE_DMA_CONFIGURED) {
        // trigger the start of the dma channel(s)
        // TODO
        h->state = MLI_MOV_STATE_DMA_RUNNING;
    } else
#endif
    {
    // in case DMA is not used, but direct copy was done, set state to DONE, and call callback.
        h->state = MLI_MOV_STATE_DONE;
        if (callbacktable[h->dma_ch].cb != NULL) {
            callbacktable[h->dma_ch].cb(callbacktable[h->dma_ch].cookie);
        }
    }
    return MLI_STATUS_OK;
}

/** 
 * @brief Polling function to detect if transfer has completed
 *
 * @detail This function will return true when the transfer is completed, and false in all
 * other cases
 */
bool mli_mov_isdone(mli_mov_handle_t* h) {
    bool done = false;
    MLI_ASSERT(h != NULL);
//    printf("isdone: state %d\n", h->state);
    if (h->state == MLI_MOV_STATE_DONE) {
        done = true;
    } else if (h->state == MLI_MOV_STATE_DMA_RUNNING) {
        // TODO: poll dma status
    } else {
        done = false;
    }
    return done;
}

/** 
 * @brief Synchronize to transfer complete
 *
 * @detail This function will do active polling and return after the transfer has completed.
 */
mli_status mli_mov_wait(mli_mov_handle_t* h) {
    MLI_ASSERT(h != NULL);

    while(!mli_mov_isdone(h)){
        //active wait
    }
    return MLI_STATUS_OK;
}


//---------------------------------------------------------------------
// functions to set available resources (e.g. dma channels)
//---------------------------------------------------------------------
/** 
 * @brief set dma channels that can be used by mli_mov functions
 *
 * @detail This function is used to set a pool of the dma channels
 * that can be used by the mli_mov functions.
 */
mli_status mli_mov_set_num_dma_ch(int ch_offset, int num_ch) {
    dma_pool.base_channel = ch_offset;
    dma_pool.pool_size = num_ch;
    for (int i = 0; i < num_ch; i++) {
        dma_pool.channel_status[i + ch_offset] = MLI_MOV_DMA_CH_AVAILABLE;
    }

    return MLI_STATUS_OK;
}

/** 
 * @brief Acquire dma channel(s)
 *
 * @detail This function finds the first available (block of) channel(s) in the pool.
 */
mli_status mli_mov_acquire_handle(int num_ch, mli_mov_handle_t* h) {
    mli_status retval = MLI_STATUS_NOT_ENGH_MEM; // TODO: add status for not enough dma channels
    MLI_ASSERT(h != NULL);
    // find first available channel, and check if enough adjacent channels are available.
    for (int i = dma_pool.base_channel; i < dma_pool.base_channel + dma_pool.pool_size; i++) {
        bool found = true;
        for (int ch_cnt = 0; ch_cnt < num_ch; ch_cnt++) {
            found &= (dma_pool.channel_status[i + ch_cnt] == MLI_MOV_DMA_CH_AVAILABLE);
        }
        if (found) {
            h->state = MLI_MOV_STATE_OPEN;
            h->dma_ch = i;
            h->num_ch = num_ch;
            for (int ch_cnt = 0; ch_cnt < num_ch; ch_cnt++) {
                dma_pool.channel_status[i + ch_cnt] = MLI_MOV_DMA_CH_IN_USE;
            }
            retval = MLI_STATUS_OK;
            break;
        }
    }

    return retval;
}

/** 
 * @brief Release dma channle(s)
 *
 * @detail This function will release the dma channels from the handle h back to the pool.
 */
mli_status mli_mov_release_handle(mli_mov_handle_t* h) {
    MLI_ASSERT(h != NULL);

    for (int ch_cnt = 0; ch_cnt < h->num_ch; ch_cnt++) {
        dma_pool.channel_status[h->dma_ch + ch_cnt] = MLI_MOV_DMA_CH_AVAILABLE;
    }
    h->state = MLI_MOV_STATE_INVALID;
    h->dma_ch = 0;
    h->num_ch = 0;
    return MLI_STATUS_OK;
}


//---------------------------------------------------------------------
// Helper functions to fill mli_mov_cfg_t
//---------------------------------------------------------------------

/** 
 * @brief Construction of cfg struct for full tensor copy
 *
 * @detail This function will fill the cfg struct with the values needed for a full tensor copy
 * it will put all the other fields to a neutral value.
 */
mli_status mli_mov_cfg_for_copy(mli_mov_cfg_t* cfg) {
    memset(cfg, 0, sizeof(mli_mov_cfg_t));
    for (int dim = 0; dim < MLI_MAX_RANK; dim++) {
        cfg->sub_sample_step[dim] = 1;
        cfg->perm_dim[dim] = dim;
    }

    return MLI_STATUS_OK;
}

/** 
 * @brief Construction of cfg struct for slicing
 *
 * @detail This function will fill the cfg struct with the values needed for copying a slice
 * from the source to the destination tensor
 */
mli_status mli_mov_cfg_for_slice(mli_mov_cfg_t* cfg, int* offsets, int* sizes, int* dst_mem_stride) {
    mli_status retval = MLI_STATUS_OK; 

    if (retval == MLI_STATUS_OK)
        retval = mli_mov_cfg_for_copy(cfg);
    for (int dim = 0; dim < MLI_MAX_RANK; dim++) {
        cfg->offset[dim] = offsets[dim];
        cfg->size[dim] = sizes[dim];
        cfg->dst_mem_stride[dim] = dst_mem_stride[dim];
    }
    return retval;
}

/** 
 * @brief Construction of cfg struct for concatenation
 *
 * @detail This function will fill the cfg struct with the values needed for copying a complete tensor
 * into a larger tensor at a specified position
 */
mli_status mli_mov_cfg_for_concat(mli_mov_cfg_t* cfg, int* dst_offsets, int* dst_mem_stride) {
    mli_status retval = MLI_STATUS_OK; 

    if (retval == MLI_STATUS_OK)
        retval = mli_mov_cfg_for_copy(cfg);
    for (int dim = 0; dim < MLI_MAX_RANK; dim++) {
        cfg->dst_offset[dim] = dst_offsets[dim];
        cfg->dst_mem_stride[dim] = dst_mem_stride[dim];
    }
    return retval;
}

/** 
 * @brief Construction of cfg struct for subsampling
 *
 * @detail This function will fill the cfg struct with the values needed for subsampling a tensor
 * a subsample step of 3 means that every third sample is copied to the output.
 */
mli_status mli_mov_cfg_for_subsample(mli_mov_cfg_t* cfg, int* sub_sample_step, int* dst_mem_stride) {

    mli_status retval = MLI_STATUS_OK; 

    if (retval == MLI_STATUS_OK)
        retval = mli_mov_cfg_for_copy(cfg);
    for (int dim = 0; dim < MLI_MAX_RANK; dim++) {
        cfg->sub_sample_step[dim] = sub_sample_step[dim];
        cfg->dst_mem_stride[dim] = dst_mem_stride[dim];
    }
    return retval;
}

/** 
 * @brief Construction of cfg struct for permutaion or transposing a tensor
 *
 * @detail This function will fill the cfg struct with the values needed for reordering the order of the dimensions in a tensor
 */
mli_status mli_mov_cfg_for_permute(mli_mov_cfg_t* cfg, uint8_t* perm_dim) {
    memset(cfg, 0, sizeof(mli_mov_cfg_t));
    for (int dim = 0; dim < MLI_MAX_RANK; dim++) {
        cfg->sub_sample_step[dim] = 1;
        cfg->perm_dim[dim] = perm_dim[dim];
    }

    return MLI_STATUS_OK;
}

/** 
 * @brief Construction of cfg struct for padding
 *
 * @detail This function will fill the cfg struct with the values needed adding zero padding to a tensor in CHW layout
 */
mli_status mli_mov_cfg_for_padding2d_chw(mli_mov_cfg_t* cfg, uint8_t padleft, uint8_t padright, uint8_t padtop, uint8_t padbot, int* dst_mem_stride) {
    mli_status retval = MLI_STATUS_OK; 

    if (retval == MLI_STATUS_OK)
        retval = mli_mov_cfg_for_copy(cfg);
    for (int dim = 0; dim < MLI_MAX_RANK; dim++) {
        cfg->dst_mem_stride[dim] = dst_mem_stride[dim];
    }
    cfg->padding_pre[FMAP_W_DIM_CHW] = padleft;
    cfg->padding_post[FMAP_W_DIM_CHW] = padright;
    cfg->padding_pre[FMAP_H_DIM_CHW] = padtop;
    cfg->padding_post[FMAP_H_DIM_CHW] = padbot;
    return retval;
}

/** 
 * @brief Construction of cfg struct for padding
 *
 * @detail This function will fill the cfg struct with the values needed adding zero padding to a tensor in HWC layout
 */
mli_status mli_mov_cfg_for_padding2d_hwc(mli_mov_cfg_t* cfg, uint8_t padleft, uint8_t padright, uint8_t padtop, uint8_t padbot, int* dst_mem_stride) {
    mli_status retval = MLI_STATUS_OK; 

    if (retval == MLI_STATUS_OK)
        retval = mli_mov_cfg_for_copy(cfg);
    for (int dim = 0; dim < MLI_MAX_RANK; dim++) {
        cfg->dst_mem_stride[dim] = dst_mem_stride[dim];
    }
    cfg->padding_pre[FMAP_W_DIM_HWC] = padleft;
    cfg->padding_post[FMAP_W_DIM_HWC] = padright;
    cfg->padding_pre[FMAP_H_DIM_HWC] = padtop;
    cfg->padding_post[FMAP_H_DIM_HWC] = padbot;
    return retval;
}

/** 
 * @brief Construction of cfg struct
 *
 * @detail This function will fill the cfg struct
 */

mli_status mli_mov_cfg_all(
    mli_mov_cfg_t* cfg,
    int* offsets,
    int* sizes,
    int* subsample_step,
    int* dst_offsets,
    int* dst_mem_stride,
    uint8_t* perm_dim,
    uint8_t padleft,
    uint8_t padright,
    uint8_t padtop,
    uint8_t padbot) {
    mli_status retval = MLI_STATUS_OK;
    memset(cfg, 0, sizeof(mli_mov_cfg_t));
    for (int dim = 0; dim < MLI_MAX_RANK; dim++) {
        cfg->offset[dim] = offsets[dim];
        cfg->size[dim] = sizes[dim];
        cfg->dst_mem_stride[dim] = dst_mem_stride[dim];
        cfg->dst_offset[dim] = dst_offsets[dim];
        cfg->sub_sample_step[dim] = subsample_step[dim];
        cfg->perm_dim[dim] = perm_dim[dim];
    }
    cfg->padding_pre[FMAP_W_DIM_HWC] = padleft;
    cfg->padding_post[FMAP_W_DIM_HWC] = padright;
    cfg->padding_pre[FMAP_H_DIM_HWC] = padtop;
    cfg->padding_post[FMAP_H_DIM_HWC] = padbot;

    return retval;
}

#pragma MLI_CODE_SECTION_END()
#ifdef __cplusplus
}
#endif

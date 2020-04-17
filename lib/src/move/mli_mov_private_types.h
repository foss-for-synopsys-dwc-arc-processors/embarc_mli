/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_MOV_PRIVATE_TYPES_H_
#define _MLI_MOV_PRIVATE_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

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



#ifdef __cplusplus
}
#endif

#endif //_MLI_MOV_PRIVATE_TYPES_H_
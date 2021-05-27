/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_MEM_INFO_H_
#define _MLI_MEM_INFO_H_

#include "mli_config.h"
#include "mli_types.h"

#ifdef _ARC
#include <arc/arc_reg.h>

#define SUPPORT_NON_ALIGNMENT  22
#define DISABLE_ALIGNMENT_CHECK 19
#endif

static MLI_FORCE_INLINE bool mli_mem_is_inside_dccm(const void *ptr) {
#ifdef core_config_dccm_size
    return ((uint32_t)ptr >= core_config_dccm_base) &&
           ((uint32_t)ptr < core_config_dccm_base + core_config_dccm_size);
#else
    return false;
#endif
}

static MLI_FORCE_INLINE bool mli_mem_is_inside_yccm(const void *ptr) {
#ifdef core_config_xy_y_base
    return ((uint32_t)ptr >= core_config_xy_y_base) &&
           ((uint32_t)ptr < core_config_xy_y_base + core_config_xy_size);
#else
    return false;
#endif
}

static MLI_FORCE_INLINE bool mli_mem_is_inside_xccm(const void *ptr) {
#ifdef core_config_xy_x_base
    return ((uint32_t)ptr >= core_config_xy_x_base) &&
           ((uint32_t)ptr < core_config_xy_x_base + core_config_xy_size);
#else
    return false;
#endif
}

static MLI_FORCE_INLINE bool mli_mem_is_inside_vccm(const void *ptr) {
#ifdef core_config_vec_mem_size
    return ((uint32_t)ptr >= core_config_vec_mem_base) &&
           ((uint32_t)ptr < core_config_vec_mem_base + core_config_vec_mem_size);
#else
    return false;
#endif
}

static MLI_FORCE_INLINE bool mli_mem_is_inside_ccm(const void *ptr) {
    return mli_mem_is_inside_dccm(ptr)
        || mli_mem_is_inside_xccm(ptr)
        || mli_mem_is_inside_yccm(ptr)
        || mli_mem_is_inside_vccm(ptr);
}

static MLI_FORCE_INLINE mli_status mli_mem_chk_ptr(void *p, uint32_t align_mask, bool check_bank) {
#if MLI_PTR_IS_VCCM
    bool is_inside_vccm = mli_mem_is_inside_vccm(p);
    if (check_bank && (!is_inside_vccm))
        return MLI_STATUS_MEM_BANK_MISMATCH;
    //Check the alignment if the pointer is inside the VCCM memory or the non_alignment isn't supported
    if (is_inside_vccm ||
        (((_lr(ISA_CONFIG)&(1<<SUPPORT_NON_ALIGNMENT)) == 0) || ((_lr(STATUS32)&(1<<DISABLE_ALIGNMENT_CHECK)) == 0))) {
        if (((uint32_t)p & align_mask) != 0)
            return MLI_STATUS_MISALIGNMENT_ERROR;
    }
#endif
#if MLI_PTR_IS_XY
    if (check_bank && (!mli_mem_is_inside_ccm(p)))
        return MLI_STATUS_MEM_BANK_MISMATCH;
#endif
#if (PLATFORM == V2DSP_XY) || ((PLATFORM == V2DSP_VECTOR) && (!MLI_PTR_IS_VCCM))
    if (((_lr(ISA_CONFIG)&(1<<SUPPORT_NON_ALIGNMENT)) == 0) || ((_lr(STATUS32)&(1<<DISABLE_ALIGNMENT_CHECK)) == 0)) {
        if (((uint32_t)p & align_mask) != 0)
            return MLI_STATUS_MISALIGNMENT_ERROR;
    }
#endif
    return MLI_STATUS_OK;
}

#endif // _MLI_MEM_INFO_H_

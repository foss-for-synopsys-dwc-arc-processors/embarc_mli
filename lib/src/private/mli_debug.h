/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_DEBUG_H_
#define _MLI_DEBUG_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include "mli_config.h"
#include "mli_types.h"

#define MLI_DBG_ENABLE_MESSAGES      (MLI_DEBUG_MODE == DBG_MODE_FULL || MLI_DEBUG_MODE == DBG_MODE_DEBUG)
#define MLI_DBG_ENABLE_ASSERTS       (MLI_DEBUG_MODE == DBG_MODE_FULL || MLI_DEBUG_MODE == DBG_MODE_DEBUG || MLI_DEBUG_MODE == DBG_MODE_ASSERT)
#define MLI_DBG_ENABLE_RETURNCODES   (MLI_DEBUG_MODE != DBG_MODE_RELEASE)
#define MLI_DBG_ENABLE_EXTRA_ASSERTS (MLI_DEBUG_MODE == DBG_MODE_FULL)

// Printf wrapper: works only in DBG_MODE_FULL and DBG_MODE_DEBUG
#if MLI_DBG_ENABLE_MESSAGES
#    include <stdio.h>
#    define MLI_PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#    define MLI_PRINTF(fmt, ...) ((void) 0)
#endif

// Printf wrapper for stack trace printing (__PRETTY_FUNCTION__ works on GCC/LLVM based compilers)
#if MLI_DEBUG_ENABLE_STACK_TRACE_MESSAGES == 1
#    include <stdio.h>
#    define MLI_PRINTF_FUNC() printf("%s\n", __PRETTY_FUNCTION__ )
#else
#    define MLI_PRINTF_FUNC() ((void) 0)
#endif

// Assert wrapper: works only in DBG_MODE_FULL and DBG_MODE_DEBUG
#if MLI_DBG_ENABLE_ASSERTS
#    include <assert.h>
#    define MLI_ASSERT(a) assert(a)
#else
#    define MLI_ASSERT(a) (void)(a)
#endif

// Extra Assert wrapper: works only in DBG_MODE_FULL
// used for checks inside performance critical loops.
#if MLI_DBG_ENABLE_EXTRA_ASSERTS
#    include <assert.h>
#    define MLI_EXTRA_ASSERT(a) assert(a)
#else
#    define MLI_EXTRA_ASSERT(a) (void)(a)
#endif

#if MLI_DBG_ENABLE_RETURNCODES
#    define MLI_CHECK(cond, msg)        mli_check(cond, msg, "", #cond, __func__)
#    define MLI_CHECK2(cond, msg, msg2) mli_check(cond, msg, msg2, #cond, __func__)
#    define MLI_CHECK_STATUS(stat, msg) mli_check_status(stat, msg, __func__)
#else
#    define MLI_CHECK(cond, msg)        ((void)(cond), false)
#    define MLI_CHECK2(cond, msg, msg2) ((void)(cond), false)
#    define MLI_CHECK_STATUS(stat, msg) (MLI_STATUS_OK)
#endif

#define MLI_CHECK_AND_FIX(variable, value) (variable = value, MLI_CHECK(variable == value, "Expected fixed value for specialized function"))

static MLI_FORCE_INLINE mli_status mli_check_status(
        mli_status stat, const char* msg, const char* funcname) {
    if (stat != MLI_STATUS_OK){
        //printf("%s: %s: %s\n", funcname, msg, err_status_to_string[status]); TODO
        MLI_PRINTF("MLI_CHECK: %s: %s: %s\n", funcname, msg, "");
        MLI_ASSERT(0);
    }
    return stat;
}

static MLI_FORCE_INLINE bool mli_check(
        int cond,
        const char* msg,
        const char* msg2,
        const char* condstring,
        const char* funcname) {
    if (!cond){
        MLI_PRINTF("MLI_CHECK: %s: condition (%s) failed: %s %s\n", funcname, condstring, msg, msg2);
        MLI_ASSERT(0);
    }
    return !cond;
}

#ifdef __cplusplus
}
#endif

#endif /* DEBUG_H_ */

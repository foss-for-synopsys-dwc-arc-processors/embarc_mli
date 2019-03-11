/*
 *  Copyright (c) 2019, Synopsys, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1) Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2)  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3) Neither the name of the <ORGANIZATION> nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef _MLI_KRN_ELTWISE_H_
#define _MLI_KRN_ELTWISE_H_

#include "mli_debug.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_load_store.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {

typedef enum {
    ELTWISE_ADD = 0,
    ELTWISE_SUB,
    ELTWISE_MUL,
    ELTWISE_MAX,
    ELTWISE_MIN
} mli_eltwise_type;

#pragma Code(".mli_lib")
//======================================================
//
//======================================================
#ifdef __Xxy
template <typename io_T>
static inline void __attribute__ ((always_inline)) eltwise_op_sub_fx (
        const MLI_PTR(int8_t) op1, 
        const MLI_PTR(int8_t) op2, 
        MLI_PTR(int8_t) out, 
        const int op1_size, 
        const int op2_size) {
    // Simple broadcast (vector on scalar)
    //==============================================
    if (op2_size == 1) {
        const io_T broadcast_val = *(const io_T *)op2;
        // Vector minus scalar
        for (int idx = 0; idx < op1_size; idx++) out[idx] = mli_math_sub_fx(op1[idx], broadcast_val);
    } else if (op1_size == 1) {
        const io_T broadcast_val = *(const io_T *)op1;
        // Scalar minus Vector
        for (int idx = 0; idx < op2_size; idx++) out[idx] = mli_math_sub_fx(broadcast_val, op2[idx]);
    } else {
        // Elemetnwise between tensors of the same shape
        //==============================================
        MLI_ASSERT(op1_size == op2_size);

        for (int idx = 0; idx < op1_size; idx++) {
            *out++ = mli_math_sub_fx(*op1++, *op2++);
        }
    }
}

template <typename io_T>
static inline void __attribute__ ((always_inline)) eltwise_op_add_fx (
        const MLI_PTR(int8_t) op1, 
        const MLI_PTR(int8_t) op2, 
        MLI_PTR(int8_t) out, 
        const int op1_size, 
        const int op2_size) {
    // Simple broadcast (vector on scalar)
    //==============================================
    if (op1_size == 1 || op2_size == 1) {
        const int8_t broadcast_val = (op1_size > op2_size) ? (*(const io_T *)op2) : (*(const io_T *)op1);
        const MLI_PTR(int8_t) vec = (op1_size > op2_size) ? (MLI_PTR(int8_t))op1 : (MLI_PTR(int8_t))op2;
        const int out_size = MAX(op1_size, op2_size);

        for (int idx = 0; idx < out_size; idx++) {
            *out++ = mli_math_add_fx(*vec++, broadcast_val);
        }
    }
    // Elemetnwise between tensors of the same shape
    //==============================================
    else {
        MLI_ASSERT(op1_size == op2_size);

        for (int idx = 0; idx < op1_size; idx++) {
            *out++ = mli_math_add_fx(*op1++, *op2++);
        }
    }
}

template <typename io_T>
static inline void __attribute__ ((always_inline)) eltwise_op_max_fx (
        const MLI_PTR(int8_t) op1, 
        const MLI_PTR(int8_t) op2, 
        MLI_PTR(int8_t) out, 
        const int op1_size, 
        const int op2_size) {
    // Simple broadcast (vector on scalar)
    //==============================================
    if (op1_size == 1 || op2_size == 1) {
        const int8_t broadcast_val = (op1_size > op2_size) ? (*(const io_T *)op2) : (*(const io_T *)op1);
        const MLI_PTR(int8_t) vec = (op1_size > op2_size) ? (MLI_PTR(int8_t))op1 : (MLI_PTR(int8_t))op2;
        const int out_size = MAX(op1_size, op2_size);

        for (int idx = 0; idx < out_size; idx++) {
            *out++ = mli_math_max_fx(*vec++, broadcast_val);
        }
    }
    // Elemetnwise between tensors of the same shape
    //==============================================
    else {
        MLI_ASSERT(op1_size == op2_size);

        for (int idx = 0; idx < op1_size; idx++) {
            *out++ = mli_math_max_fx(*op1++, *op2++);
        }
    }
}

template <typename io_T>
static inline void __attribute__ ((always_inline)) eltwise_op_min_fx (
        const MLI_PTR(int8_t) op1, 
        const MLI_PTR(int8_t) op2, 
        MLI_PTR(int8_t) out, 
        const int op1_size, 
        const int op2_size) {
    // Simple broadcast (vector on scalar)
    //==============================================
    if (op1_size == 1 || op2_size == 1) {
        const int8_t broadcast_val = (op1_size > op2_size) ? (*(const io_T *)op2) : (*(const io_T *)op1);
        const MLI_PTR(int8_t) vec = (op1_size > op2_size) ? (MLI_PTR(int8_t))op1 : (MLI_PTR(int8_t))op2;
        const int out_size = MAX(op1_size, op2_size);

        for (int idx = 0; idx < out_size; idx++) {
            *out++ = mli_math_min_fx(*vec++, broadcast_val);
        }
    }
    // Elemetnwise between tensors of the same shape
    //==============================================
    else {
        MLI_ASSERT(op1_size == op2_size);

        for (int idx = 0; idx < op1_size; idx++) {
            *out++ = mli_math_min_fx(*op1++, *op2++);
        }
    }
}
#endif

template <typename io_T>
static inline void __attribute__ ((always_inline)) eltwise_op_add_fx (
        const MLI_PTR(io_T) op1, 
        const MLI_PTR(io_T) op2, 
        MLI_PTR(io_T) out, 
        const int op1_size, 
        const int op2_size) {
    // Simple broadcast (vector on scalar)
    //==============================================
    if (op1_size == 1 || op2_size == 1) {
        const io_T broadcast_val = (op1_size > op2_size) ? (*(const io_T *)op2) : (*(const io_T *)op1);
        const MLI_PTR(io_T) vec = (op1_size > op2_size) ? (MLI_PTR(io_T))op1 : (MLI_PTR(io_T))op2;
        const int out_size = MAX(op1_size, op2_size);

        const v2q15_t broadcast_val_v2 = fx_create_v2q15(broadcast_val, broadcast_val);
        for (int idx = 0; idx < out_size / 2; idx++) {
            mli_prv_store_2_samples(out, fx_add_v2q15(broadcast_val_v2, mli_prv_load_2_samples(vec)));
            vec += 2;
            out += 2;
        }
        if (out_size & 1) {
            *out++ = mli_math_add_fx(*vec++, broadcast_val);
        }
    }
    // Elemetnwise between tensors of the same shape
    //==============================================
    else {
        MLI_ASSERT(op1_size == op2_size);

        for (int idx = 0; idx < op1_size / 2; idx++) {
            mli_prv_store_2_samples(out, mli_prv_load_add_vec2(op1, op2));
            op1 += 2;
            op2 += 2;
            out += 2;
        }
        if (op1_size & 1) {
            *out++ = mli_math_add_fx(*op1++, *op2++);
        }
    }
}

template <typename io_T>
static inline void __attribute__ ((always_inline)) eltwise_op_sub_fx (
        const MLI_PTR(io_T) op1, 
        const MLI_PTR(io_T) op2, 
        MLI_PTR(io_T) out, 
        const int op1_size, 
        const int op2_size) {
    // Simple broadcast (vector on scalar)
    //==============================================
    if (op2_size == 1) {
        const io_T broadcast_val = *(const io_T *)op2;
        const v2q15_t broadcast_val_v2 = fx_create_v2q15(broadcast_val, broadcast_val);
        // Vector minus scalar
        for (int idx = 0; idx < op1_size / 2; idx++) {
            mli_prv_store_2_samples(out, fx_sub_v2q15(mli_prv_load_2_samples(op1), broadcast_val_v2));
            op1 += 2;
            out += 2;
        }
        if (op1_size & 1) {
            *out++ = mli_math_sub_fx(*op1++, broadcast_val);
        }
    } else if (op1_size == 1) {
        const io_T broadcast_val = *(const io_T *)op1;
        const v2q15_t broadcast_val_v2 = fx_create_v2q15(broadcast_val, broadcast_val);
        // Scalar minus Vector
        for (int idx = 0; idx < op2_size / 2; idx++) {
            mli_prv_store_2_samples(out, fx_sub_v2q15(broadcast_val_v2, mli_prv_load_2_samples(op2)));
            op2 += 2;
            out += 2;
        }
        if (op2_size & 1) {
            *out++ = mli_math_sub_fx(broadcast_val, *op2++);
        }
    } else {
        // Elemetnwise between tensors of the same shape
        //==============================================
        MLI_ASSERT(op1_size == op2_size);

        for (int idx = 0; idx < op1_size / 2; idx++) {
            mli_prv_store_2_samples(out, mli_prv_load_sub_vec2(op1, op2));
            op1 += 2;
            op2 += 2;
            out += 2;
        }
        if (op1_size & 1) {
            *out++ = mli_math_sub_fx(*op1++, *op2++);
        }
    }
}

template <typename io_T>
static inline void __attribute__ ((always_inline)) eltwise_op_max_fx (
        const MLI_PTR(io_T) op1, 
        const MLI_PTR(io_T) op2, 
        MLI_PTR(io_T) out, 
        const int op1_size, 
        const int op2_size) {
    // Simple broadcast (vector on scalar)
    //==============================================
    if (op1_size == 1 || op2_size == 1) {
        const io_T broadcast_val = (op1_size > op2_size) ? (*(const io_T *)op2) : (*(const io_T *)op1);
        const MLI_PTR(io_T) vec = (op1_size > op2_size) ? (MLI_PTR(io_T))op1 : (MLI_PTR(io_T))op2;
        const int out_size = MAX(op1_size, op2_size);

        const v2q15_t broadcast_val_v2 = fx_create_v2q15(broadcast_val, broadcast_val);
        for (int idx = 0; idx < out_size / 2; idx++) {
            mli_prv_store_2_samples(out, fx_max_v2q15(broadcast_val_v2, mli_prv_load_2_samples(vec)));
            vec += 2;
            out += 2;
        }
        if (out_size & 1) {
            *out++ = mli_math_max_fx(*vec++, broadcast_val);
        }
    } else {
    // Elemetnwise between tensors of the same shape
    //==============================================
        MLI_ASSERT(op1_size == op2_size);

        for (int idx = 0; idx < op1_size / 2; idx++) {
            mli_prv_store_2_samples(out, mli_prv_load_max_vec2(op1, op2));
            op1 += 2;
            op2 += 2;
            out += 2;
        }
        if (op1_size & 1) {
            *out++ = mli_math_max_fx(*op1++, *op2++);
        }
    }
}

template <typename io_T>
static inline void __attribute__ ((always_inline)) eltwise_op_min_fx (
        const MLI_PTR(io_T) op1, 
        const MLI_PTR(io_T) op2, 
        MLI_PTR(io_T) out, 
        const int op1_size, 
        const int op2_size) {
    // Simple broadcast (vector on scalar)
    //==============================================
    if (op1_size == 1 || op2_size == 1) {
        const io_T broadcast_val = (op1_size > op2_size) ? (*(const io_T *)op2) : (*(const io_T *)op1);
        const MLI_PTR(io_T) vec = (op1_size > op2_size) ? (MLI_PTR(io_T))op1 : (MLI_PTR(io_T))op2;
        const int out_size = MAX(op1_size, op2_size);

        const v2q15_t broadcast_val_v2 = fx_create_v2q15(broadcast_val, broadcast_val);
        for (int idx = 0; idx < out_size / 2; idx++) {
            mli_prv_store_2_samples(out, fx_min_v2q15(broadcast_val_v2, mli_prv_load_2_samples(vec)));
            vec += 2;
            out += 2;
        }
        if (out_size & 1) {
            *out++ = mli_math_min_fx(*vec++, broadcast_val);
        }
    } else {
    // Elemetnwise between tensors of the same shape
    //==============================================
        MLI_ASSERT(op1_size == op2_size);

        for (int idx = 0; idx < op1_size / 2; idx++) {
            mli_prv_store_2_samples(out, mli_prv_load_min_vec2(op1, op2));
            op1 += 2;
            op2 += 2;
            out += 2;
        }
        if (op1_size & 1) {
            *out++ = mli_math_min_fx(*op1++, *op2++);
        }
    }
}

template <typename io_T>
static inline void __attribute__ ((always_inline)) eltwise_op_mul_fx (
        const MLI_PTR(io_T) op1, 
        const MLI_PTR(io_T) op2, 
        MLI_PTR(io_T) out, 
        const int op1_size, 
        const int op2_size, 
        const int mul_out_shift) {
    // Simple broadcast (vector on scalar)
    //==============================================
    if (op1_size == 1 || op2_size == 1) {
        const io_T broadcast_val = (op1_size > op2_size) ? (*(const io_T *)op2) : (*(const io_T *)op1);
        const MLI_PTR(io_T) vec = (op1_size > op2_size) ? op1 : op2;
        const int out_size = MAX(op1_size, op2_size);
        v2q15_t broadcast_val_v2 = {broadcast_val, broadcast_val};

        if ((out_size & 0x3) || (out_size < 0x7)) {
            for (int j = 0; j < (out_size & 0x3); j++) {
                auto acc = mli_prv_init_accu((io_T)0);
                mli_prv_load_mac(&acc, vec++, (const MLI_PTR(io_T) __restrict) & broadcast_val);
                mli_prv_clip_and_store_output(out++, &acc, mul_out_shift);
            }
            for (int j = 0; j < (out_size & ~0x3) / 2; j++) {
                auto acc_v = mli_prv_init_accu_v((io_T)0);
                mli_math_mac_fx_vec2(&acc_v, mli_prv_load_2_samples(vec), broadcast_val_v2);
                mli_prv_clip_and_store_output_v(out, &acc_v, mul_out_shift);
                vec += 2;
                out += 2;
            }
        } else {
#pragma clang loop unroll_count(2)
            for (int idx = 0; idx < out_size / 2; idx++) {
                auto acc_v = mli_prv_init_accu_v((io_T)0);
                mli_math_mac_fx_vec2(&acc_v, mli_prv_load_2_samples(vec), broadcast_val_v2);
                mli_prv_clip_and_store_output_v(out, &acc_v, mul_out_shift);
                vec += 2;
                out += 2;
            }
        }
    } else {
    // Elemetnwise between tensors of the same shape
    //==============================================
        MLI_ASSERT(op1_size == op2_size);

        if (op1_size & 0x3) {
            for (int j = 0; j < (op1_size & 0x3); j++) {
                auto acc = mli_prv_init_accu((io_T)0);
                mli_prv_load_mac(&acc, op1++, op2++);
                mli_prv_clip_and_store_output(out++, &acc, mul_out_shift);
            }

#pragma clang loop unroll_count(2)
            for (int j = 0; j < (op1_size & ~0x3) / 2; j++) {
                auto acc_v = mli_prv_init_accu_v((io_T)0);
                mli_math_mac_fx_vec2(&acc_v, mli_prv_load_2_samples(op1), mli_prv_load_2_samples(op2));
                mli_prv_clip_and_store_output_v(out, &acc_v, mul_out_shift);
                op1 += 2;
                op2 += 2;
                out += 2;
            }
        } else {
#pragma clang loop unroll_count(2)
            for (int idx = 0; idx < op1_size / 2; idx++) {
                auto acc_v = mli_prv_init_accu_v((io_T)0);
                mli_math_mac_fx_vec2(&acc_v, mli_prv_load_2_samples(op1), mli_prv_load_2_samples(op2));
                mli_prv_clip_and_store_output_v(out, &acc_v, mul_out_shift);
                op1 += 2;
                op2 += 2;
                out += 2;
            }
        }
    }
}

template <typename io_T>
static inline void __attribute__ ((always_inline)) eltwise_op_mul_with_restricts_fx (
        const MLI_PTR(io_T) __restrict op1, 
        const MLI_PTR(io_T) __restrict op2, 
        MLI_PTR(io_T) __restrict out, 
        const int op1_size, 
        const int op2_size, 
        const int mul_out_shift) {
    // Simple broadcast (vector on scalar)
    //==============================================
    if (op1_size == 1 || op2_size == 1) {
        const io_T broadcast_val = (op1_size > op2_size) ? (*(const io_T *)op2) : (*(const io_T *)op1);
        const MLI_PTR(io_T) vec = (op1_size > op2_size) ? op1 : op2;
        const int out_size = MAX(op1_size, op2_size);
        v2q15_t broadcast_val_v2 = {broadcast_val, broadcast_val};

        if ((out_size & 0x3) || (out_size < 0x7)) {
            for (int j = 0; j < (out_size & 0x3); j++) {
                auto acc = mli_prv_init_accu((io_T)0);
                mli_prv_load_mac(&acc, vec++, (const MLI_PTR(io_T) __restrict) & broadcast_val);
                mli_prv_clip_and_store_output(out++, &acc, mul_out_shift);
            }
            for (int j = 0; j < (out_size & ~0x3) / 2; j++) {
                auto acc_v = mli_prv_init_accu_v((io_T)0);
                mli_math_mac_fx_vec2(&acc_v, mli_prv_load_2_samples(vec), broadcast_val_v2);
                mli_prv_clip_and_store_output_v(out, &acc_v, mul_out_shift);
                vec += 2;
                out += 2;
            }
        } else {
#pragma clang loop unroll_count(2)
            for (int idx = 0; idx < out_size / 2; idx++) {
                auto acc_v = mli_prv_init_accu_v((io_T)0);
                mli_math_mac_fx_vec2(&acc_v, mli_prv_load_2_samples(vec), broadcast_val_v2);
                mli_prv_clip_and_store_output_v(out, &acc_v, mul_out_shift);
                vec += 2;
                out += 2;
            }
        }
    } else {
    // Elemetnwise between tensors of the same shape
    //==============================================
        MLI_ASSERT(op1_size == op2_size);

        if (op1_size & 0x3) {
            for (int j = 0; j < (op1_size & 0x3); j++) {
                auto acc = mli_prv_init_accu((io_T)0);
                mli_prv_load_mac(&acc, op1++, op2++);
                mli_prv_clip_and_store_output(out++, &acc, mul_out_shift);
            }

#pragma clang loop unroll_count(2)
            for (int j = 0; j < (op1_size & ~0x3) / 2; j++) {
                auto acc_v = mli_prv_init_accu_v((io_T)0);
                mli_math_mac_fx_vec2(&acc_v, mli_prv_load_2_samples(op1), mli_prv_load_2_samples(op2));
                mli_prv_clip_and_store_output_v(out, &acc_v, mul_out_shift);
                op1 += 2;
                op2 += 2;
                out += 2;
            }
        } else {
#pragma clang loop unroll_count(2)
            for (int idx = 0; idx < op1_size / 2; idx++) {
                auto acc_v = mli_prv_init_accu_v((io_T)0);
                mli_math_mac_fx_vec2(&acc_v, mli_prv_load_2_samples(op1), mli_prv_load_2_samples(op2));
                mli_prv_clip_and_store_output_v(out, &acc_v, mul_out_shift);
                op1 += 2;
                op2 += 2;
                out += 2;
            }
        }
    }
}

//======================================================
//
//======================================================
template <typename io_T, mli_eltwise_type func_type>
static inline void eltwise_prepare_and_run_fx(const mli_tensor *in1, const mli_tensor *in2, mli_tensor *out) {
    mli_prv_fx_init_dsp_ctrl();
    // Extract general parameters for function
    uint32_t in1_sz = mli_prv_count_elem_num(in1);
    uint32_t in2_sz = mli_prv_count_elem_num(in2);

    // Extract in/out pointers to mem
    const MLI_PTR(io_T) inp1_ptr = (const MLI_PTR(io_T))(in1->data);
    const MLI_PTR(io_T) inp2_ptr = (const MLI_PTR(io_T))(in2->data);
    MLI_PTR(io_T) out_ptr = (MLI_PTR(io_T))(out->data);

    // Extract in/out as scalar values
    const io_T in1_scalar = (io_T)((intptr_t)(in1->data));
    const io_T in2_scalar = (io_T)((intptr_t)(in2->data));
    io_T out_scalar;

    inp1_ptr = (in1->rank != 0) ? inp1_ptr : (const MLI_PTR(io_T)) & in1_scalar;
    inp2_ptr = (in2->rank != 0) ? inp2_ptr : (const MLI_PTR(io_T)) & in2_scalar;
    out_ptr = (out->capacity > 0) ? out_ptr : (MLI_PTR(io_T)) & out_scalar;

    // Calc outshift for MUL operation
    const int mul_out_shift = mli_prv_calc_shift(in1, in2, out);
    if (func_type == ELTWISE_ADD) {
        eltwise_op_add_fx<io_T>(inp1_ptr, inp2_ptr, out_ptr, in1_sz, in2_sz);
    } else if (func_type == ELTWISE_SUB) {
        eltwise_op_sub_fx<io_T>(inp1_ptr, inp2_ptr, out_ptr, in1_sz, in2_sz);
    } else if (func_type == ELTWISE_MAX) {
        eltwise_op_max_fx<io_T>(inp1_ptr, inp2_ptr, out_ptr, in1_sz, in2_sz);
    } else if (func_type == ELTWISE_MIN) {
        eltwise_op_min_fx<io_T>(inp1_ptr, inp2_ptr, out_ptr, in1_sz, in2_sz);
    } else if (func_type == ELTWISE_MUL) {
        if ((inp1_ptr == inp2_ptr) || (inp1_ptr == out_ptr) || (inp2_ptr == out_ptr))
            eltwise_op_mul_fx<io_T>(inp1_ptr, inp2_ptr, out_ptr, in1_sz, in2_sz, mul_out_shift);
        else
            eltwise_op_mul_with_restricts_fx<io_T>(inp1_ptr, inp2_ptr, out_ptr, in1_sz, in2_sz, mul_out_shift);
    }
    // Fill output tensor parameters
    //======================================
    if (out->capacity == 0) {
        // In case we calculated 1 scalar value
        out->data = mli_math_cast_scalar_to_ptr_fx(out_scalar);
        out->rank = 0;
    } else {
        const unsigned *shape_ptr = (in1_sz > in2_sz) ? in1->shape : in2->shape;
        int rank = (in1_sz > in2_sz) ? (int)in1->rank : (int)in2->rank;
        for (int k = 0; k < rank; k++) out->shape[k] = shape_ptr[k];
        out->rank = rank;
    }
    if (func_type != ELTWISE_MUL) out->el_params.fx.frac_bits = in1->el_params.fx.frac_bits;
    out->el_type = in1->el_type;
}

#pragma Code()

}  // namespace mli

#endif  // _MLI_KRN_ELTWISE_H_

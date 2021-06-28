/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_DOTPROD_VDSP_H_
#define _MLI_KRN_DOTPROD_VDSP_H_

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "mli_types.h"
#include "mli_prv_load_store.h"

namespace mli {
namespace krn {
namespace vdsp {

MLI_FORCE_INLINE grp_vNx4accint_t init_accu_grp(vNx4accint_t accu) {
    grp_vNx4accint_t r;
    r.accu0 = accu;
    r.accu1 = accu;
    r.accu2 = accu;
    r.accu3 = accu;
    return r;
}

MLI_FORCE_INLINE grp_vNx2accint_t init_accu_grp(vNx2accint_t accu) {
    grp_vNx2accint_t r;
    r.accu0 = accu;
    r.accu1 = accu;
    r.accu2 = accu;
    r.accu3 = accu;
    return r;
}

MLI_FORCE_INLINE grp_vNx4accshort_t init_accu_grp(vNx4accshort_t accu) {
    grp_vNx4accshort_t r;
    r.accu0 = accu;
    r.accu1 = accu;
    r.accu2 = accu;
    r.accu3 = accu;
    return r;
}

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod1D_v(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int vals,
        const int in_step,
        const int krn_step) {

    // optimal unroll is 16, but for targets with smaller vector size, it needs to be reduced.
    auto dummy = mli_prv_load_1vec(in);
    constexpr int max_unroll = sizeof(dummy) / sizeof(io_T);
    constexpr int ch_unroll = MIN(16, max_unroll);
    int idx;

    for (idx = 0; idx < vals - (ch_unroll - 1); idx+=ch_unroll) {
        auto multi_in = mli_prv_load_1vec(in);
#pragma clang loop unroll(full)
        for (int i = 0; i < ch_unroll; i++) {
            accu = mli_prv_mac_load_v_s(accu, krn, (io_T)multi_in[i]);
            krn += krn_step;
        } // ch_unroll
        in += in_step * ch_unroll;
    } // vals

    for ( ; idx < vals; idx++) {
        accu = mli_prv_mac_load_v_s(accu, krn, *in);
        in += in_step;
        krn += krn_step;
    }
    return accu;
}

// this function unrolls in the output width dimension to allow to use multiple accumulators in parallel.
// this way the vector loads of the weights can be re-used,
// and it also unrolls in the input channel dimension to combine scalar loads
template <int unroll, typename io_T, typename w_T, typename grpacc_T>
static MLI_FORCE_INLINE grpacc_T dotprod1D_v_unroll(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        grpacc_T accu,
        const int vals,
        const int in_step,
        const int in_unroll_step,
        const int krn_step) {

    // optimal unroll is 8, but for targets with smaller vector size, it needs to be reduced.
    auto dummy = mli_prv_load_1vec(in);
    constexpr int max_unroll = sizeof(dummy) / sizeof(io_T);
    constexpr int ch_unroll = MIN(8, max_unroll);
    int idx;

    for (idx = 0; idx < vals - (ch_unroll - 1); idx+=ch_unroll) {
        auto multi_in0 = mli_prv_load_1vec(in);
        auto multi_in1 = mli_prv_load_1vec(in + in_unroll_step);
        auto multi_in2 = mli_prv_load_1vec(in + 2 * in_unroll_step);
        auto multi_in3 = mli_prv_load_1vec(in + 3 * in_unroll_step);
#pragma clang loop unroll(full)
        for (int i = 0; i < ch_unroll; i++) {
            accu.accu0 = mli_prv_mac_load_v_s(accu.accu0, krn, (io_T)multi_in0[i]);
            if (unroll > 1) {
                accu.accu1 = mli_prv_mac_load_v_s(accu.accu1, krn, (io_T)multi_in1[i]);
            }
            if (unroll > 2) {
                accu.accu2 = mli_prv_mac_load_v_s(accu.accu2, krn, (io_T)multi_in2[i]);
            }
            if (unroll > 3) {
                accu.accu3 = mli_prv_mac_load_v_s(accu.accu3, krn, (io_T)multi_in3[i]);
            }

            krn += krn_step;
        } // ch_unroll
        in += in_step * ch_unroll;
    } // vals

    for ( ; idx < vals; idx++) {
        accu.accu0 = mli_prv_mac_load_v_s(accu.accu0, krn, in);
        if (unroll > 1) {
            accu.accu1 = mli_prv_mac_load_v_s(accu.accu1, krn, in + in_unroll_step);
        }
        if (unroll > 2) {
            accu.accu2 = mli_prv_mac_load_v_s(accu.accu2, krn, in + 2 * in_unroll_step);
        }
        if (unroll > 3) {
            accu.accu3 = mli_prv_mac_load_v_s(accu.accu3, krn, in + 3 * in_unroll_step);
        }
        in += in_step;
        krn += krn_step;
    }
    return accu;
}

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D_vv(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step) {
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;

    for (int row = 0; row < height; row++) {
#pragma clang loop pipeline(enable)
#pragma clang loop pipeline_options(0x10)
        for (int clmn = 0; clmn < width; clmn++) {
            accu = mli_prv_mac_load_v_v(accu, krn, in);
            in += in_col_step;
            krn += kern_col_step;
        }
        in += in_row_step;
        krn += kern_row_step;
    }
    return accu;
}

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D_vv_unrolled(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step) {
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"

#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            accu = mli_prv_mac_load_v_v(accu, krn, in);
            in += in_col_step;
            krn += kern_col_step;
        }
        in += in_row_step;
        krn += kern_row_step;
    }
    return accu;
#pragma clang diagnostic pop
}

template <int unroll, typename io_T, typename w_T, typename grpacc_T>
static MLI_FORCE_INLINE grpacc_T dotprod2D_vv_wunroll(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        grpacc_T accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int in_unroll_step,
        int kern_col_step,
        int kern_row_step) {
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;

    for (int row = 0; row < height; row++) {
        for (int clmn = 0; clmn < width; clmn++) {
            accu.accu0 = mli_prv_mac_load_v_v(accu.accu0, krn, in);
            if (unroll > 1) {
                accu.accu1 = mli_prv_mac_load_v_v(accu.accu1, krn, in + in_unroll_step);
            }
            if (unroll > 2) {
                accu.accu2 = mli_prv_mac_load_v_v(accu.accu2, krn, in + 2 * in_unroll_step);
            }
            if (unroll > 3) {
                accu.accu3 = mli_prv_mac_load_v_v(accu.accu3, krn, in + 3 * in_unroll_step);
            }
            in += in_col_step;
            krn += kern_col_step;
        }
        in += in_row_step;
        krn += kern_row_step;
    }
    return accu;
}

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D_vv_ptrvector(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step) {
    int in_row_step_orig = in_row_step;
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;

    vNint_t addr_vec = 0;
    int i = 0;
    int offset = in_row_step_orig * sizeof(io_T);
#pragma clang loop unroll(full)
    for (int row = 1; row < height; row++) {
        addr_vec[i++] = offset;
        offset += in_row_step_orig * sizeof(io_T);
    }
    i = 0;
    addr_vec += (int)in;

    for (int clmn = 0; clmn < width; clmn++) {
        accu = mli_prv_mac_load_v_v(accu, krn, in);
        in += in_col_step;
        krn += kern_col_step;
    }
    krn += kern_row_step;

#pragma clang loop unroll(full)
    for (int row = 1; row < height; row++) {
        MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T))addr_vec[i++];
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            accu = mli_prv_mac_load_v_v(accu, krn, in_ptr);
            in_ptr += in_col_step;
            krn += kern_col_step;
        }
        krn += kern_row_step;
    }
    return accu;
}

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE acc_T dotprod3D_v_pad (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T accu) {
    in_ch_step -= height * in_row_step;
    kern_ch_step -= height * kern_row_step;
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;

    __builtin_assume (height > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    for (int ch = 0; ch < channels; ch++) {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                accu = mli_prv_mac_load_v_s(accu, krn, in);
                in += in_col_step;
                krn += kern_col_step;
            }
            in += in_row_step;
            krn += kern_row_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
#pragma clang diagnostic pop
    return accu;
}

static MLI_FORCE_INLINE vNx2short_t make_vindex(
        int width,
        int height,
        int in_col_step,
        int in_row_step,
        int unroll = 1,
        int in_unroll_step = 0) {
    vNx2short_t vindex;
    MLI_ASSERT(width * height * unroll <= (sizeof(vNx2short_t) / sizeof(short)));
    int idx = 0;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"

#pragma clang loop unroll(full)
    for (int unroll_idx = 0; unroll_idx < unroll; unroll_idx++) {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                vindex[idx] = unroll_idx * in_unroll_step + row * in_row_step + clmn * in_col_step;
                idx++;
            }
        }
    }
#pragma clang diagnostic pop
    return vindex;
}

static MLI_FORCE_INLINE vNx4short_t make_vindex2(
        int width,
        int height,
        int in_col_step,
        int in_row_step,
        int unroll = 1,
        int in_unroll_step = 0) {
    vNx4short_t vindex;
    int vec_length = (sizeof(vNx2short_t) / sizeof(short));
    int idx = 0;
    MLI_ASSERT(width * height * unroll <= 2 * vec_length);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"

#pragma clang loop unroll(full)
    for (int unroll_idx = 0; unroll_idx < unroll; unroll_idx++) {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                if (idx < vec_length) {
                    vindex.lo[idx] = unroll_idx * in_unroll_step + row * in_row_step + clmn * in_col_step;
                } else {
                    vindex.hi[idx - vec_length] = unroll_idx * in_unroll_step + row * in_row_step + clmn * in_col_step;
                }
                idx++;
            }
        }
    }
#pragma clang diagnostic pop
    return vindex;
}

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE acc_T dotprod3D_v_variable_krn_sz (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T accu) {

    kern_row_step -= width * kern_col_step;
    in_row_step -= width * in_col_step;

    __builtin_assume (height > 0);

    for (int row = 0; row < height; row++) {
        for (int clmn = 0; clmn < width; clmn++) {
            accu = dotprod1D_v(in, krn, accu, channels, in_ch_step, kern_ch_step);
            krn += kern_col_step;
            in += in_col_step;
        }
        krn += kern_row_step;
        in += in_row_step;
    }

    return accu;
}

template <int unroll, typename in_T, typename w_T, typename grpacc_T >
static MLI_FORCE_INLINE grpacc_T dotprod3D_v_variable_krn_sz_unroll (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int in_unroll_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        grpacc_T accu) {
    kern_row_step -= width * kern_col_step;
    in_row_step -= width * in_col_step;

    __builtin_assume (height > 0);

    for (int row = 0; row < height; row++) {
        for (int clmn = 0; clmn < width; clmn++) {
            accu = dotprod1D_v_unroll<unroll>(in, krn, accu, channels, in_ch_step, in_unroll_step, kern_ch_step);
            krn += kern_col_step;
            in += in_col_step;
        }
        krn += kern_row_step;
        in += in_row_step;
    }

    return accu;
}

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE acc_T dotprod3D_v_pad_gather1 (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T accu) {
    // construct gather vector with pointers
    const vNx2int_t vindex = to_vNx2int_t(make_vindex(width, height, in_col_step, in_row_step));

    kern_ch_step -= height * kern_row_step;
    kern_row_step -= width * kern_col_step;

    __builtin_assume (height > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    for (int ch = 0; ch < channels; ch++) {
        // gather load w x h samples from input
        auto in_gather = mli_prv_gather_load_nx2_samples(in, vindex, width * height);
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                // get input sample from gather vector.
                in_T input = in_gather[row * width + clmn];
                accu = mli_prv_mac_load_v_s(accu, krn, input);
                krn += kern_col_step;
            }
            krn += kern_row_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
#pragma clang diagnostic pop
    return accu;
}

/* gather2 is used for larger kernels.
because of the vector selection logic this code is most beneficial for
fixed kernel size where the loops can be fully unrolled and the vector selection
logic can be optimized away.

for char it could be beneficial to make a special version that loads all samples in a single vector.
although the downside is that this way the element get from the first part of the
vector is stalling until the complete vector is loaded.
so there is some benefit in loading second half in a different vector.

*/
template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE acc_T dotprod3D_v_pad_gather2 (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T accu) {
    // construct gather vector with pointers
    const vNx4int_t vindex = to_vNx4int_t(make_vindex2(width, height, in_col_step, in_row_step));
    int vec_length = (sizeof(vNx2short_t) / sizeof(short));

    kern_ch_step -= height * kern_row_step;
    kern_row_step -= width * kern_col_step;

    __builtin_assume (height > 0);
    for (int ch = 0; ch < channels; ch++) {
        // gather load w x h samples from input
        auto in_gather_lo = mli_prv_gather_load_nx2_samples(in, vindex.lo, width * height);
        auto in_gather_hi = mli_prv_gather_load_nx2_samples(in, vindex.hi, width * height);
// This function is intended to be used only with fixed kernel sizes because of the complex indexing
// that is why the warnings about unroll not possible are not switched off here.
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                // get input sample from gather vector.
                int idx = row * width + clmn;
                in_T input = idx >= vec_length ? in_gather_hi[idx - vec_length] : in_gather_lo[idx];
                accu = mli_prv_mac_load_v_s(accu, krn, input);
                krn += kern_col_step;
            }
            krn += kern_row_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
    return accu;
}

template < typename in_T, typename w_T, typename acc_T, bool fixed_size >
static MLI_FORCE_INLINE acc_T dotprod3D_v (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T accu) {
/* Optimized version will use a gather load to combine the scalar loads.
   the number of loads required depends on the kernel width, height and unroll factor.
   The number of loads available in the gather load depends on the vector length.
   For both 8bit and 16bit loads, the number of elemens that can be loaded in a single
   gather instruction is the same.
   */
    int num_loads_single_gather = _VDSP_NUM_16BIT_LANES;
    int required_loads = width * height;

    if (required_loads <= num_loads_single_gather) {
        return dotprod3D_v_pad_gather1(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
    } else if (fixed_size && (required_loads <= 2 * num_loads_single_gather)) {
        return dotprod3D_v_pad_gather2(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
    } else {
        return dotprod3D_v_pad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
    }
}

template < typename in_T, typename w_T, typename grpacc_T, int unroll >
static MLI_FORCE_INLINE grpacc_T dotprod3D_v_pad_gather1_unroll (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        int unroll_step,
        int unroll_1,
        int required_loads,
        int kernel_size,
        int ext_width,
        grpacc_T accu) {
    // construct gather vector with pointers
    const vNx2int_t vindex = to_vNx2int_t(make_vindex(ext_width, height, in_col_step, in_row_step, unroll_1, unroll_step));

    kern_ch_step -= height * kern_row_step;
    kern_row_step -= width * kern_col_step;

    __builtin_assume (height > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    for (int ch = 0; ch < channels; ch++) {
        // gather load w x h samples from input
        auto in_gather = mli_prv_gather_load_nx2_samples(in, vindex, required_loads);
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                // get input sample from gather vector.
                in_T input = in_gather[row * ext_width + clmn];
                accu.accu0 = mli_prv_mac_load_v_s(accu.accu0, krn, input);
                if (unroll > 1) {
                    in_T input = in_gather[row * ext_width + clmn + 1 * kernel_size];
                    accu.accu1 = mli_prv_mac_load_v_s(accu.accu1, krn, input);
                }
                if (unroll > 2) {
                    in_T input = in_gather[row * ext_width + clmn + 2 * kernel_size];
                    accu.accu2 = mli_prv_mac_load_v_s(accu.accu2, krn, input);
                }
                if (unroll > 3) {
                    in_T input = in_gather[row * ext_width + clmn + 3 * kernel_size];
                    accu.accu3 = mli_prv_mac_load_v_s(accu.accu3, krn, input);
                }
                krn += kern_col_step;
            }
            krn += kern_row_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
#pragma clang diagnostic pop
    return accu;
}

/* gather2 is used for larger kernels.
because of the vector selection logic this code is most beneficial for
fixed kernel size where the loops can be fully unrolled and the vector selection
logic can be optimized away.

for char it could be beneficial to make a special version that loads all samples in a single vector.
although the downside is that this way the element get from the first part of the
vector is stalling until the complete vector is loaded.
so there is some benefit in loading second half in a different vector.

*/
template < typename in_T, typename w_T, typename grpacc_T, int unroll >
static MLI_FORCE_INLINE grpacc_T dotprod3D_v_pad_gather2_unroll (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        int unroll_step,
        int unroll_1,
        int required_loads,
        int kernel_size,
        int ext_width,
        grpacc_T accu) {
    // construct gather vector with pointers
    const vNx4int_t vindex = to_vNx4int_t(make_vindex2(ext_width, height, in_col_step, in_row_step, unroll_1, unroll_step));
    int vec_length = (sizeof(vNx2short_t) / sizeof(short));

    kern_ch_step -= height * kern_row_step;
    kern_row_step -= width * kern_col_step;

    __builtin_assume (height > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    for (int ch = 0; ch < channels; ch++) {
        // gather load w x h samples from input
        auto in_gather_lo = mli_prv_gather_load_nx2_samples(in, vindex.lo, vec_length);
        auto in_gather_hi = mli_prv_gather_load_nx2_samples(in, vindex.hi, required_loads - vec_length);
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                // get input sample from gather vector.
                int idx = row * ext_width + clmn;
                in_T input = idx >= vec_length ? in_gather_hi[idx - vec_length] : in_gather_lo[idx];
                accu.accu0 = mli_prv_mac_load_v_s(accu.accu0, krn, input);

                if (unroll > 1) {
                    int idx = row * ext_width + clmn + 1 * kernel_size;
                    in_T input = idx >= vec_length ? in_gather_hi[idx - vec_length] : in_gather_lo[idx];
                    accu.accu1 = mli_prv_mac_load_v_s(accu.accu1, krn, input);
                }
                if (unroll > 2) {
                    int idx = row * ext_width + clmn + 2 * kernel_size;
                    in_T input = idx >= vec_length ? in_gather_hi[idx - vec_length] : in_gather_lo[idx];
                    accu.accu2 = mli_prv_mac_load_v_s(accu.accu2, krn, input);
                }
                if (unroll > 3) {
                    int idx = row * ext_width + clmn + 3 * kernel_size;
                    in_T input = idx >= vec_length ? in_gather_hi[idx - vec_length] : in_gather_lo[idx];
                    accu.accu3 = mli_prv_mac_load_v_s(accu.accu3, krn, input);
                }
                krn += kern_col_step;
            }
            krn += kern_row_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
#pragma clang diagnostic pop
    return accu;
}

template <int unroll, bool fixed_size, typename in_T, typename w_T, typename grpacc_T >
static MLI_FORCE_INLINE grpacc_T dotprod3D_v_unroll (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int in_unroll_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        int unroll_step,
        int unroll_1,
        int required_loads,
        int kernel_size,
        int ext_width,
        grpacc_T accu) {
/* Optimized version will use a gather load to combine the scalar loads.
   the number of loads required depends on the kernel width, height and unroll factor.
   The number of loads available in the gather load depends on the vector length.
   For both 8bit and 16bit loads, the number of elemens that can be loaded in a single
   gather instruction is the same.
   */
    int num_loads_single_gather = _VDSP_NUM_16BIT_LANES;

    if (required_loads <= num_loads_single_gather) {
        return dotprod3D_v_pad_gather1_unroll<in_T, w_T, grpacc_T, unroll>(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step,
                kern_col_step, kern_row_step, kern_ch_step, unroll_step, unroll_1, required_loads, kernel_size, ext_width, accu);
    } else if (fixed_size && (required_loads <= 2 * num_loads_single_gather)) {
        return dotprod3D_v_pad_gather2_unroll<in_T, w_T, grpacc_T, unroll>(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step,
                kern_col_step, kern_row_step, kern_ch_step, unroll_step, unroll_1, required_loads, kernel_size, ext_width, accu);
    } else {
        grpacc_T r;
        r.accu0 = dotprod3D_v_pad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu0);
        if (unroll > 1) {
            in += in_unroll_step;
            r.accu1 = dotprod3D_v_pad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu1);
        }
        if (unroll > 2) {
            in += in_unroll_step;
            r.accu2 = dotprod3D_v_pad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu2);
        }
        if (unroll > 3) {
            in += in_unroll_step;
            r.accu3 = dotprod3D_v_pad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu3);
        }
        return r;
    }
}

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE acc_T dotprod3D_v_nopad_gather1 (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T accu) {
    // construct gather vector with pointers
    const vNx2int_t vindex = to_vNx2int_t(make_vindex(width, height, in_col_step, in_row_step));

    MLI_ASSERT(kern_row_step == width * kern_col_step);
    kern_ch_step -= height * kern_row_step;

    __builtin_assume (height * width > 0);
    __builtin_assume (channels > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    for (int ch = 0; ch < channels; ch++) {
        // gather load w x h samples from input
        auto in_gather = mli_prv_gather_load_nx2_samples(in, vindex, width * height);
#pragma clang loop unroll(full)
        // with nopad version row and col loop can be combined.
        for (int idx = 0; idx < height * width; idx++) {
            // get input sample from gather vector.
            in_T input = in_gather[idx];
            accu = mli_prv_mac_load_v_s(accu, krn, input);
            krn += kern_col_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
#pragma clang diagnostic pop
    return accu;
}

/* This function is unrolled in the height dimension
   in_height_step is the number of elements to the next set of input samples
   This can be different from the in_row_step because the dilation factor can
   be different than the stride.*/
template < typename in_T, typename w_T, typename grpacc_T, int unroll>
static MLI_FORCE_INLINE grpacc_T dotprod3D_v_nopad_gather1_unroll (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int in_unroll_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        grpacc_T accu) {
    // construct gather vector with pointers
    const vNx2int_t vindex = to_vNx2int_t(make_vindex(width, height, in_col_step, in_row_step, unroll, in_unroll_step));
    int kernel_size = width * height;

    MLI_ASSERT(kern_row_step == width * kern_col_step);
    kern_ch_step -= height * kern_row_step;

    __builtin_assume (height * width > 0);
    __builtin_assume (channels > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    for (int ch = 0; ch < channels; ch++) {
        // gather load w x h samples from input
        auto in_gather = mli_prv_gather_load_nx2_samples(in, vindex, width * height * unroll);
#pragma clang loop unroll(full)
        // with nopad version row and col loop can be combined.
        for (int idx = 0; idx < height * width; idx++) {
            // get input sample from gather vector.
            in_T input = in_gather[idx];
            accu.accu0 = mli_prv_mac_load_v_s(accu.accu0, krn, input);

            if (unroll > 1) {
                in_T input = in_gather[idx + kernel_size];
                accu.accu1 = mli_prv_mac_load_v_s(accu.accu1, krn, input);
            }
            if (unroll > 2) {
                in_T input = in_gather[idx + 2 * kernel_size];
                accu.accu2 = mli_prv_mac_load_v_s(accu.accu2, krn, input);
            }
            if (unroll > 3) {
                in_T input = in_gather[idx + 3 * kernel_size];
                accu.accu3 = mli_prv_mac_load_v_s(accu.accu3, krn, input);
            }

            krn += kern_col_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
#pragma clang diagnostic pop
    return accu;
}

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE acc_T dotprod3D_v_nopad_gather2 (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T accu) {
    // construct gather vector with pointers
    const vNx4int_t vindex = to_vNx4int_t(make_vindex2(width, height, in_col_step, in_row_step));
    int vec_length = (sizeof(vNx2short_t) / sizeof(short));

    MLI_ASSERT(height * width > vec_length);
    MLI_ASSERT(height * width <= 2 * vec_length);
    MLI_ASSERT(kern_row_step == width * kern_col_step);
    kern_ch_step -= height * kern_row_step;

    __builtin_assume (height * width > 0);
    __builtin_assume (channels > 0);
    for (int ch = 0; ch < channels; ch++) {
        // gather load w x h samples from input
        auto in_gather_lo = mli_prv_gather_load_nx2_samples(in, vindex.lo, vec_length);
#pragma clang loop unroll(full)
        // with nopad version row and col loop can be combined.
        for (int idx = 0; idx < vec_length; idx++) {
            // get input sample from gather vector.
            in_T input = in_gather_lo[idx];
            accu = mli_prv_mac_load_v_s(accu, krn, input);
            krn += kern_col_step;
        }
        auto in_gather_hi = mli_prv_gather_load_nx2_samples(in, vindex.hi, width * height - vec_length);
        // remainder is in the second part of the vindex vector.
        for (int idx = 0; idx < (height * width - vec_length); idx++) {
            // get input sample from gather vector.
            in_T input = in_gather_hi[idx];
            accu = mli_prv_mac_load_v_s(accu, krn, input);
            krn += kern_col_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
    return accu;
}

/* This function is unrolled in the height dimension
   in_height_step is the number of elements to the next set of input samples
   This can be different from the in_row_step because the dilation factor can
   be different than the stride.*/
template < typename in_T, typename w_T, typename grpacc_T, int unroll>
static MLI_FORCE_INLINE grpacc_T dotprod3D_v_nopad_gather2_unroll (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int in_unroll_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        grpacc_T accu) {
    // construct gather vector with pointers
    int half_unroll = (unroll + 1)/2;
    const vNx2int_t vindex = to_vNx2int_t(make_vindex(width, height, in_col_step, in_row_step, half_unroll, in_unroll_step));
    int kernel_size = width * height;

    kern_ch_step -= height * kern_row_step;

    __builtin_assume (height * width > 0);
    __builtin_assume (channels > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    for (int ch = 0; ch < channels; ch++) {
        // gather load w x h samples from input
        auto in_gatherA = mli_prv_gather_load_nx2_samples(in, vindex, width * height * half_unroll);
        auto in_gatherB = mli_prv_gather_load_nx2_samples(in + half_unroll * in_unroll_step, vindex, width * height * (unroll - half_unroll));

        // with nopad version row and col loop can be combined.
        for (int idx = 0; idx < height * width; idx++) {
            // get input sample from gather vector.
            in_T input = in_gatherA[idx];
            accu.accu0 = mli_prv_mac_load_v_s(accu.accu0, krn, input);

            if (unroll == 2) {
                input = in_gatherB[idx];
                accu.accu1 = mli_prv_mac_load_v_s(accu.accu1, krn, input);
            }
            if (unroll == 3) {
                input = in_gatherA[idx + kernel_size];
                accu.accu1 = mli_prv_mac_load_v_s(accu.accu1, krn, input);
                input = in_gatherB[idx];
                accu.accu2 = mli_prv_mac_load_v_s(accu.accu2, krn, input);
            }
            if (unroll == 4) {
                input = in_gatherA[idx + kernel_size];
                accu.accu1 = mli_prv_mac_load_v_s(accu.accu1, krn, input);
                input = in_gatherB[idx];
                accu.accu2 = mli_prv_mac_load_v_s(accu.accu2, krn, input);
                input = in_gatherB[idx + kernel_size];
                accu.accu3 = mli_prv_mac_load_v_s(accu.accu3, krn, input);
            }

            krn += kern_col_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
#pragma clang diagnostic pop
    return accu;
}

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE acc_T dotprod3D_v_nopad (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T accu) {
/* Optimized version will use a gather load to combine the scalar loads.
   the number of loads required depends on the kernel width, height and unroll factor.
   The number of loads available in the gather load depends on the vector length.
   For both 8bit and 16bit loads, the number of elemens that can be loaded in a single
   gather instruction is the same.
   */
    int num_loads_single_gather = _VDSP_NUM_16BIT_LANES;
    int required_loads = width * height;

    if (required_loads <= num_loads_single_gather) {
        return dotprod3D_v_nopad_gather1(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
    } else {
        return dotprod3D_v(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
    }
}

/* This function is unrolled 4x in the height dimension (H4)
   in_height_step is the number of elements to the next set of input samples
   This can be different from the in_row_step because the dilation factor can
   be different than the stride.*/
template < int unroll, typename in_T, typename w_T, typename accgrp_T >
static MLI_FORCE_INLINE accgrp_T dotprod3D_v_nopad_unroll (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        int in_unroll_step,
        accgrp_T accu) {
/* Optimized version will use a gather load to combine the scalar loads.
   the number of loads required depends on the kernel width, height and unroll factor.
   The number of loads available in the gather load depends on the vector length.
   For both 8bit and 16bit loads, the number of elemens that can be loaded in a single
   gather instruction is the same.
   */
    int num_loads_single_gather = _VDSP_NUM_16BIT_LANES;
    int required_loads = width * height * unroll;

    MLI_ASSERT(unroll <= 4);

    if ((required_loads <= num_loads_single_gather)) {
        return dotprod3D_v_nopad_gather1_unroll<in_T, w_T, accgrp_T, unroll>(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, in_unroll_step, kern_col_step, kern_row_step, kern_ch_step, accu);
//    } else if (((required_loads <= num_loads_single_gather * 2)) {
//        return dotprod3D_v_nopad_gather2_unroll<in_T, w_T, accgrp_T, unroll>(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, in_unroll_step, kern_col_step, kern_row_step, kern_ch_step, accu);
    } else {
        accgrp_T r;
        r.accu0 = dotprod3D_v_nopad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu0);
        if (unroll > 1) {
            in += in_unroll_step;
            r.accu1 = dotprod3D_v_nopad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu1);
        }
        if (unroll > 2) {
            in += in_unroll_step;
            r.accu2 = dotprod3D_v_nopad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu2);
        }
        if (unroll > 3) {
            in += in_unroll_step;
            r.accu3 = dotprod3D_v_nopad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu3);
        }
        return r;
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_DOTPROD_VDSP_H_

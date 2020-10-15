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

    for (int idx = 0; idx < vals; idx++) {
        accu = mli_prv_mac_load_v_s(accu, krn, in);
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
        int in_row_step,
        int in_col_step) {
    vNx2short_t vindex;
    MLI_ASSERT(width * height <= (sizeof(vNx2short_t) / sizeof(short)));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"

#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            vindex[row * width + clmn] = row * in_row_step + clmn * in_col_step;
        }
    }
#pragma clang diagnostic pop
    return vindex;
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
    const vNx2int_t vindex = to_vNx2int_t(make_vindex(width, height, in_row_step, in_col_step));

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

template < typename in_T, typename w_T, typename acc_T >
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
    } else {
        return dotprod3D_v_pad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
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
    const vNx2int_t vindex = to_vNx2int_t(make_vindex(width, height, in_row_step, in_col_step));

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
template < typename in_T, typename w_T, typename grpacc_T, int unrollH>
static MLI_FORCE_INLINE grpacc_T dotprod3D_v_nopad_gather1_unrollH (
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
        grpacc_T accu) {
    // construct gather vector with pointers
    const vNx2int_t vindex = to_vNx2int_t(make_vindex(width, height + unrollH - 1, in_row_step, in_col_step));

    MLI_ASSERT(kern_row_step == width * kern_col_step);
    kern_ch_step -= height * kern_row_step;

    __builtin_assume (height * width > 0);
    __builtin_assume (channels > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    for (int ch = 0; ch < channels; ch++) {
        // gather load w x h samples from input
        auto in_gather = mli_prv_gather_load_nx2_samples(in, vindex, width * (height + unrollH - 1));
#pragma clang loop unroll(full)
        // with nopad version row and col loop can be combined.
        for (int idx = 0; idx < height * width; idx++) {
            // get input sample from gather vector.
            in_T input = in_gather[idx];
            accu.accu0 = mli_prv_mac_load_v_s(accu.accu0, krn, input);

            if (unrollH > 1) {
                in_T input = in_gather[idx + width];
                accu.accu1 = mli_prv_mac_load_v_s(accu.accu1, krn, input);
            }
            if (unrollH > 2) {
                in_T input = in_gather[idx + 2 * width];
                accu.accu2 = mli_prv_mac_load_v_s(accu.accu2, krn, input);
            }
            if (unrollH > 3) {
                in_T input = in_gather[idx + 3 * width];
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
//    } else if (required_loads <= num_loads_single_gather * 2) {
//        return dotprod3D_v_nopad_gather2(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
    } else {
        return dotprod3D_v(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
    }
}

/* This function is unrolled 2x in the height dimension (H2)
   in_height_step is the number of elements to the next set of input samples
   This can be different from the in_row_step because the dilation factor can
   be different than the stride.*/
template < typename in_T, typename w_T, typename accgrp_T >
static MLI_FORCE_INLINE accgrp_T dotprod3D_v_nopad_unrollH2 (
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
        int in_height_step,
        accgrp_T accu) {
/* Optimized version will use a gather load to combine the scalar loads.
   the number of loads required depends on the kernel width, height and unroll factor.
   The number of loads available in the gather load depends on the vector length.
   For both 8bit and 16bit loads, the number of elemens that can be loaded in a single
   gather instruction is the same.
   */
    constexpr int unroll = 4;
    int num_loads_single_gather = _VDSP_NUM_16BIT_LANES;
    int required_loads = width * (height + unroll - 1);

    if ((required_loads <= num_loads_single_gather) && (in_height_step == in_row_step)) {
        return dotprod3D_v_nopad_gather1_unrollH<in_T, w_T, accgrp_T, unroll>(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
//    } else if ((required_loads <= num_loads_single_gather * 2) && (in_height_step == in_row_step)) {
//        return dotprod3D_v_nopad_gather2_unrollH(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
    } else {
        accgrp_T r;
        r.accu0 = dotprod3D_v_nopad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu0);
        in += in_height_step;
        r.accu1 = dotprod3D_v_nopad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu1);
        return r;
    }

}

/* This function is unrolled 4x in the height dimension (H4)
   in_height_step is the number of elements to the next set of input samples
   This can be different from the in_row_step because the dilation factor can
   be different than the stride.*/
template < typename in_T, typename w_T, typename accgrp_T >
static MLI_FORCE_INLINE accgrp_T dotprod3D_v_nopad_unrollH4 (
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
        int in_height_step,
        accgrp_T accu) {
/* Optimized version will use a gather load to combine the scalar loads.
   the number of loads required depends on the kernel width, height and unroll factor.
   The number of loads available in the gather load depends on the vector length.
   For both 8bit and 16bit loads, the number of elemens that can be loaded in a single
   gather instruction is the same.
   */
    constexpr int unroll = 4;
    int num_loads_single_gather = _VDSP_NUM_16BIT_LANES;
    int required_loads = width * (height + unroll - 1);

    if ((required_loads <= num_loads_single_gather) && (in_height_step == in_row_step)) {
        return dotprod3D_v_nopad_gather1_unrollH<in_T, w_T, accgrp_T, unroll>(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
//    } else if ((required_loads <= num_loads_single_gather * 2) && (in_height_step == in_row_step)) {
//        return dotprod3D_v_nopad_gather2_unrollH(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu);
    } else {
        accgrp_T r;
        r.accu0 = dotprod3D_v_nopad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu0);
        in += in_height_step;
        r.accu1 = dotprod3D_v_nopad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu1);
        in += in_height_step;
        r.accu2 = dotprod3D_v_nopad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu2);
        in += in_height_step;
        r.accu3 = dotprod3D_v_nopad(in, krn, width, height, channels, in_col_step, in_row_step, in_ch_step, kern_col_step, kern_row_step, kern_ch_step, accu.accu3);
        return r;
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_DOTPROD_VDSP_H_

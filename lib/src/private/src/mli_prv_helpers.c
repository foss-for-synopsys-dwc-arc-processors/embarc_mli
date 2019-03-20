/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_private_types.h"
#include "mli_math_macros.h"

#pragma Code(".mli_lib")

//============================================================
//
//============================================================
mli_minmax_t
mli_prv_get_relu_min_max (const mli_relu_cfg * cfg, const mli_tensor * out) {
    mli_minmax_t val_limit;
    int min_val, max_val;
    switch (out->el_type) {
    case MLI_EL_FX_8:
        min_val = INT8_MIN;
        max_val = INT8_MAX;
        break;
    case MLI_EL_FX_16:
        min_val = INT16_MIN;
        max_val = INT16_MAX;
        break;
    default:
        MLI_ASSERT(0);             /* unsupported element type */
    }

    switch (cfg->type) {
    case MLI_RELU_GEN:
        val_limit.min = 0;
        val_limit.max = max_val;
        break;
    case MLI_RELU_6:
        val_limit.min = 0;
        val_limit.max = MIN (6 << (int) out->el_params.fx.frac_bits, max_val);
        break;
    case MLI_RELU_1:
        val_limit.min = (uint16_t) MAX (-(1 << (int) out->el_params.fx.frac_bits), min_val);
        val_limit.max = (uint16_t) MIN (1 << (int) out->el_params.fx.frac_bits, max_val);
        break;
    default:
        // For leaky and param relu there is no saturation in the function domain.
        // only container type limitations (8bit or 16 bit)
        val_limit.min = min_val;
        val_limit.max = max_val;
    }

    return val_limit;
}
#pragma Code()


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


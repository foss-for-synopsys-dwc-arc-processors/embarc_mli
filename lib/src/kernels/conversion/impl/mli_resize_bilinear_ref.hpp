/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */

#ifndef _MLI_RESIZE_BILINEAR_REF_HPP_
#define _MLI_RESIZE_BILINEAR_REF_HPP_

#include "mli_types.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_mem_info.h"

namespace snps_arc::metaware::mli {
namespace krn {
namespace ref {

// TODO: change mli_tensor to Tensor
// TODO: change BHWC to BHWGC
mli_status mli_resize_bilinear(const mli_tensor* in, const ResizeOpConfig& cfg, mli_tensor* out) {

    mli_prv_fx_init_dsp_ctrl();

    auto in_prv = mli_prv_get_generic_tensor<MLI_PTR(int8_t)>(in);
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(int32_t)>(out);

    int one_fx = (1 << cfg.shift);
    int row_fx, row_int, delta_row_fx, input_row0_int, input_row1_int;
    int col_fx, col_int, delta_col_fx, input_col0_int, input_col1_int;
    int8_t v00, v01, v10, v11;
    int32_t out_val;
    int b, h, w, c;
    for (b = 0; b < out_prv.shape[kTensorBatchDim]; b++) {
        for (h = 0; h < out_prv.shape[kTensorHeightDim]; h++) {
            row_fx = h * cfg.stride[0] + cfg.offset[0];
            row_int = row_fx >> cfg.shift;
            delta_row_fx = row_fx - (row_int << cfg.shift);
            input_row0_int = MIN(MAX(row_int, 0), in_prv.shape[kTensorHeightDim] - 1);
            input_row1_int = MIN(row_int + 1, in_prv.shape[kTensorHeightDim] - 1);
            for (w = 0; w < out_prv.shape[kTensorWidthDim]; w++) {
                col_fx = w * cfg.stride[1] + cfg.offset[1];
                col_int = col_fx >> cfg.shift;
                delta_col_fx = col_fx - (col_int << cfg.shift);
                input_col0_int = MIN(MAX(col_int, 0), in_prv.shape[kTensorWidthDim] - 1);
                input_col1_int = MIN(col_int + 1, in_prv.shape[kTensorWidthDim] - 1);
                for (c = 0; c < out_prv.shape[kTensorChannelDim]; c++) {
                    // read the nearest 4 input values around the output point
                    v00 = mli_prv_tensor_read(in_prv, b, input_row0_int, input_col0_int, c);
                    v01 = mli_prv_tensor_read(in_prv, b, input_row0_int, input_col1_int, c);
                    v10 = mli_prv_tensor_read(in_prv, b, input_row1_int, input_col0_int, c);
                    v11 = mli_prv_tensor_read(in_prv, b, input_row1_int, input_col1_int, c);

                    // compute and write output point
                    out_val = v00 * (one_fx - delta_row_fx) * (one_fx - delta_col_fx) +
                              v01 * (one_fx - delta_row_fx) * delta_col_fx            +
                              v10 * delta_row_fx            * (one_fx - delta_col_fx) +
                              v11 * delta_row_fx            * delta_col_fx;
                    mli_prv_tensor_write(out_val, out_prv, b, h, w, c);

                }
            }
        }
    }

    return MLI_STATUS_OK;
}


} // namespace ref
} // namespace krn
} // namespace snps_arc::metaware::mli

#endif // _MLI_RESIZE_BILINEAR_REF_HPP_
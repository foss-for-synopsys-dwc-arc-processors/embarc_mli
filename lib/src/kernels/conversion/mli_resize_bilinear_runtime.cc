/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */

#include <cstring>

#include "mli_debug.h"
#include "mli_ref_runtime_api.hpp"
#include "mli_resize_bilinear.hpp"
#include "mli_ref_private_types.hpp"

namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::snps_arc::metaware::mli::krn;

// TODO: remove this after standart version of ResizeBilinear with Issue() will be added
void run_mli_resize_bilinear_standalone(const mli_tensor* in, const ResizeOpConfig& cfg, mli_tensor* out){


  MLI_ASSERT((cfg.stride[0] > 0) && (cfg.stride[1] > 0));
  MLI_ASSERT((cfg.shift >= 1) && (cfg.shift <= 11));

  // resizing factor is limited to 1/16 downscaling
  if (cfg.shift >= 4) {
    MLI_ASSERT(cfg.stride[0] > (1 << (cfg.shift - 4)));
    MLI_ASSERT(cfg.stride[1] > (1 << (cfg.shift - 4)));
  }
  // resizing factor is limited to x16 upscaling
  const int16_t val_16_fx = 16 << cfg.shift;
  MLI_ASSERT(cfg.stride[0] < val_16_fx);
  MLI_ASSERT(cfg.stride[1] < val_16_fx);

  // offset range is limited to maximum 16 pixels 
  MLI_ASSERT(cfg.offset[0] > -val_16_fx && cfg.offset[0] < val_16_fx);
  MLI_ASSERT(cfg.offset[1] > -val_16_fx && cfg.offset[1] < val_16_fx);

  mli_krn::mli_resize_bilinear(in, cfg, out);
}

}  // namespace snps_arc::metaware::mli::ref


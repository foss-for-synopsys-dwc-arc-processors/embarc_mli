/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */
#include <cstring>

#include "mli_ref_runtime_api.hpp"
#include "mli_ref_compiler_api.hpp"
#include "mli_ref_private_types.hpp"

namespace snps_arc::metaware::mli::ref {

ResizeBilinear_CS::ResizeBilinear_CS(const lib_mli::PlatformDescription pd,
                                     const TensorIterator<NoBuffer, kResizeBilinearRank, kResizeBilinearIterRank> &in,
                                     const ResizeOpConfig &cfg,
                                     const TensorIterator<NoBuffer, kResizeBilinearRank, kResizeBilinearIterRank> &out)
                                     : m_cfg(cfg), m_in(in), m_out(out), m_pd(pd) {

  MLI_ASSERT((m_cfg.stride[0] > 0) && (m_cfg.stride[1] > 0));
  MLI_ASSERT((m_cfg.shift >= 1) && (m_cfg.shift <= 11));

  // resizing factor is limited to 1/16 downscaling
  if (m_cfg.shift >= 4) {
    MLI_ASSERT(m_cfg.stride[0] > (1 << (m_cfg.shift - 4)));
    MLI_ASSERT(m_cfg.stride[1] > (1 << (m_cfg.shift - 4)));
  }
 
  // resizing factor is limited to x16 upscaling
  const int16_t val_16_fx = 16 << cfg.shift;
  MLI_ASSERT(m_cfg.stride[0] < val_16_fx);
  MLI_ASSERT(m_cfg.stride[1] < val_16_fx);

  // offset range is limited to maximum 16 pixels 
  MLI_ASSERT((m_cfg.offset[0] > -val_16_fx) && (m_cfg.offset[0] < val_16_fx));
  MLI_ASSERT((m_cfg.offset[1] > -val_16_fx) && (m_cfg.offset[1] < val_16_fx));
}

mli_status ResizeBilinear_CS::AttachBufferOffsets(const OffsetBuffer &input,
                                                  const OffsetBuffer &output,
                                                  const OffsetBuffer &ctrl_buffer) {

    m_in.set_buf(input);
    m_out.set_buf(output);

    return MLI_STATUS_OK;
}

mli_status ResizeBilinear_CS::GetKernelPrivateData(void *kernel_private_data_buffer) {

    MLI_ASSERT(m_in.get_elem_size() == sizeof(int8_t) && m_out.get_elem_size() == sizeof(int32_t));
    MLI_ASSERT(m_in.get_tensor().get_rank() == m_out.get_tensor().get_rank());

    ResizeBilinearPrivateData opaque_obj;
    opaque_obj.input = m_in;
    opaque_obj.output = m_out;
    opaque_obj.config = m_cfg;

    std::memcpy(kernel_private_data_buffer, (void *)&opaque_obj, sizeof(opaque_obj));

    return MLI_STATUS_OK;
}

unsigned ResizeBilinear_CS::GetKernelPrivateDataSize() const {
    return sizeof(ResizeBilinearPrivateData);
}

unsigned ResizeBilinear_CS::GetRuntimeObjectSize() const {
    return sizeof(ResizeBilinear);
}

}  // namespace snps_arc::metaware::mli::ref

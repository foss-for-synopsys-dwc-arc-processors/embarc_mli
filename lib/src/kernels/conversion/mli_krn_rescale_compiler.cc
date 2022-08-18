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

Rescale_CS::Rescale_CS(const lib_mli::PlatformDescription pd,
                       const Tensor<NoBuffer, 4> input_shape,
                       const RescaleConfig &cfg,
                       const Tensor<NoBuffer, 4> output_tile_shape)
                       : m_config {cfg},
                         m_pd {pd} {

    uint32_t in_shape[4];
    uint32_t out_shape[4];
    int32_t in_stride[4];
    int32_t out_stride[4];

    for (uint32_t i = 0; i < 4; ++i) {
        in_shape[i] = input_shape.get_dim(i);
        in_stride[i] = input_shape.get_mem_stride(i);
        out_shape[i] = output_tile_shape.get_dim(i);
        out_stride[i] = output_tile_shape.get_mem_stride(i);
    }

    m_input = Tensor<OffsetBuffer, 4>(OffsetBuffer(), in_shape, in_stride, input_shape.get_rank());
    m_output = Tensor<OffsetBuffer, 4>(OffsetBuffer(), out_shape, out_stride, output_tile_shape.get_rank());

    m_input_buffer_size =
    service::GetBufferSize(input_shape.get_rank(), in_shape, in_stride);
    m_output_buffer_size =
    service::GetBufferSize(output_tile_shape.get_rank(), out_shape, out_stride);

    uint32_t params_elem_num;
    if (cfg.axis < 0) {
      params_elem_num = 1;
    } else {
      params_elem_num = in_shape[cfg.axis];
    }

    // size_in_elements
    m_params_elem_num = params_elem_num;
    // size_in_bytes = No.of elements multplied by params elements' sizes
    m_encoded_params_buffer_size = params_elem_num *
            (sizeof(int32_t) + sizeof(int16_t) + sizeof(int8_t) + sizeof(int8_t));

    m_use_tiling = false;
    for (int i = 0; i < 4; i++) {
      m_tile_total_output_size[i] = 0;
      m_tile_iteration_order[i] = 0;
      m_tile_output_first_inc[i] = 0;
      m_tile_output_inc[i] = 0;
    };
}

mli_status Rescale_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input,
                                           const Tensor<OffsetBuffer, 4> &output,
                                           const OffsetBuffer &encoded_params,
                                           const OffsetBuffer &ctrl_buffer) {
    MLI_ASSERT(output.get_buf().get_size() == m_output_buffer_size * output.get_elem_size());

    m_input.set_buf(input.get_buf());
    m_output.set_buf(output.get_buf());
    m_encoded_params = encoded_params;

    return MLI_STATUS_OK;
}

mli_status Rescale_CS::GetKernelPrivateData( void *kernel_private_data_buffer ) {

    RescalePrivateData opaque_obj;

    MLI_ASSERT(m_input.get_rank() == m_output.get_rank());
    if (m_use_tiling) {
      for (uint32_t i = 0; i < m_input.get_rank(); i++) {
        MLI_ASSERT(m_output.get_dim(i) == MAX(m_tile_output_first_inc[i], m_tile_output_inc[i]));
      }
    }
    else {
      for (uint32_t i = 0; i < m_input.get_rank(); i++) {
        MLI_ASSERT(m_output.get_dim(i) == m_input.get_dim(i));
      }
    }
    opaque_obj.io_rank = m_input.get_rank();

    opaque_obj.input_buffer = m_input.get_buf();
    opaque_obj.output_buffer = m_output.get_buf();
    opaque_obj.encoded_params_buffer = m_encoded_params;

    opaque_obj.params_elem_num = m_params_elem_num;

    opaque_obj.input_b = m_input.get_dim(mli::kTensorBatchDim);
    opaque_obj.input_h = m_input.get_dim(mli::kTensorHeightDim);
    opaque_obj.input_w = m_input.get_dim(mli::kTensorWidthDim);
    opaque_obj.input_c = m_input.get_dim(mli::kTensorChannelDim);

    opaque_obj.output_b = m_output.get_dim(mli::kTensorBatchDim);
    opaque_obj.output_h = m_output.get_dim(mli::kTensorHeightDim);
    opaque_obj.output_w = m_output.get_dim(mli::kTensorWidthDim);
    opaque_obj.output_c = m_output.get_dim(mli::kTensorChannelDim);

    opaque_obj.input_b_stride = m_input.get_mem_stride(mli::kTensorBatchDim);
    opaque_obj.input_h_stride = m_input.get_mem_stride(mli::kTensorHeightDim);
    opaque_obj.input_w_stride = m_input.get_mem_stride(mli::kTensorWidthDim);
    opaque_obj.input_c_stride = m_input.get_mem_stride(mli::kTensorChannelDim);

    opaque_obj.output_b_stride = m_output.get_mem_stride(mli::kTensorBatchDim);
    opaque_obj.output_h_stride = m_output.get_mem_stride(mli::kTensorHeightDim);
    opaque_obj.output_w_stride = m_output.get_mem_stride(mli::kTensorWidthDim);
    opaque_obj.output_c_stride = m_output.get_mem_stride(mli::kTensorChannelDim);

    // Rescale configuration
    opaque_obj.rescale_axis = m_config.axis;

    opaque_obj.m_use_tiling = m_use_tiling;
    for (int i = 0; i < 4; i++) {
      opaque_obj.m_tile_total_output_size[i] = m_tile_total_output_size[i];
      opaque_obj.m_tile_iteration_order[i] = m_tile_iteration_order[i];
      opaque_obj.m_tile_output_first_inc[i] = m_tile_output_first_inc[i];
      opaque_obj.m_tile_output_inc[i] = m_tile_output_inc[i];
    }

    std::memcpy(kernel_private_data_buffer, (void *)&opaque_obj, sizeof(opaque_obj));

    return MLI_STATUS_OK;
}

mli_status Rescale_CS::EncodeParams(const Tensor<Buffer, 1> &in_bias,
                                    const Tensor<Buffer, 1> &scale,
                                    const Tensor<Buffer, 1> &shift,
                                    const Tensor<Buffer, 1> &out_bias,
                                    Buffer &encoded_params) {

    uint32_t i, j, last_count;
    uint32_t inbias_size = in_bias.get_buf().get_size();
    uint32_t scale_size = scale.get_buf().get_size();
    uint32_t shift_size = shift.get_buf().get_size();
    uint32_t outbias_size = out_bias.get_buf().get_size();

    m_params_elem_num = in_bias.get_buf().get_size() / in_bias.get_buf().get_elem_size();

    for (i = 0, j = 0; i < inbias_size; i+=4, j++) {
        int32_t val = in_bias.read<int32_t>(j);
        int8_t* pval = reinterpret_cast<int8_t*>(&val);
        encoded_params.write<int8_t>(i + 0, *(pval + 0));
        encoded_params.write<int8_t>(i + 1, *(pval + 1));
        encoded_params.write<int8_t>(i + 2, *(pval + 2));
        encoded_params.write<int8_t>(i + 3, *(pval + 3));
    }
    for (last_count = i, j = 0; i < (scale_size + last_count); i+=2, j++) {
        int16_t val = scale.read<int16_t>(j);
        int8_t* pval = reinterpret_cast<int8_t*>(&val);
        encoded_params.write<int8_t>(i + 0, *(pval + 0));
        encoded_params.write<int8_t>(i + 1, *(pval + 1));
    }
    for (last_count = i, j = 0; i < (shift_size + last_count); i+=1, j++) {
        encoded_params.write<int8_t>(i, shift.read<int8_t>(j));
    }
    for (last_count = i, j = 0; i < (outbias_size + last_count); i+=1, j++) {
        encoded_params.write<int8_t>(i, out_bias.read<int8_t>(j));
    }

    return MLI_STATUS_OK;
}

unsigned Rescale_CS::GetKernelPrivateDataSize() const {
    return sizeof(RescalePrivateData);
}

unsigned Rescale_CS::GetRuntimeObjectSize() const {
    return sizeof(Rescale);
}

unsigned Rescale_CS::GetInputBufferSize() const {
    return m_input_buffer_size;
}

unsigned Rescale_CS::GetOutputBufferSize() const {
    return m_output_buffer_size;
}

unsigned Rescale_CS::GetEncodedParamsSize() const {
    return m_encoded_params_buffer_size; // in bytes
}

mli_status Rescale_CS::SetIterators(uint32_t output_total_size[4],
                                    uint32_t iteration_order[4],
                                    uint32_t output_first_inc[4],
                                    uint32_t output_inc[4]) {
  m_use_tiling = true;
  for (int i = 0; i < 4; i++) {
    m_tile_total_output_size[i] = output_total_size[i];
    m_tile_iteration_order[i] = iteration_order[i];
    m_tile_output_first_inc[i] = output_first_inc[i];
    m_tile_output_inc[i] = output_inc[i];
  }
  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref

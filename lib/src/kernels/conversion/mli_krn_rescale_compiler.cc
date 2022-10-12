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
                       const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& input,
                       const RescaleConfig& cfg,
                       const TensorIterator<NoBuffer, kRescaleParamRank, kRescaleIterRank>& enc_param,
                       const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& output)
  : m_config(cfg),
    m_input(input),
    m_enc_param(enc_param),
    m_output(output),
    m_pd(pd) {

    uint32_t io_rank = input.get_tensor().get_rank();
    MLI_ASSERT(io_rank == output.get_tensor().get_rank());
    MLI_ASSERT(io_rank <= kRescaleRank);

    uint32_t in_shape[kRescaleRank];
    uint32_t out_shape[kRescaleRank];
    int32_t in_stride[kRescaleRank];
    int32_t out_stride[kRescaleRank];
    input.get_full_shape(in_shape);
    input.get_mem_strides(in_stride);
    output.get_full_shape(out_shape);
    output.get_mem_strides(out_stride);

    m_input_buffer_size =
      service::GetBufferSize(io_rank, in_shape, in_stride);
    m_output_buffer_size =
      service::GetBufferSize(io_rank, out_shape, out_stride);

    m_params_elem_num = cfg.axis == kPerTensorQuantDim ? 1 : in_shape[cfg.axis];
    /*
     * It should be
     * "m_encoded_params_buffer_size = sizeof(input) + sizeof(int16_t) + sizeof(int8_t) + sizeof(output)",
     * but NoBuffer is passed in ctor, so m_encoded_params_buffer_size is set to maximum possible size.
     * later it's changed in EncodeParams() to actual (smaller or equal) size.
     */ 
    m_encoded_params_buffer_size = m_params_elem_num *
      (sizeof(int32_t) + sizeof(int16_t) + sizeof(int8_t) + sizeof(int32_t));
}

mli_status Rescale_CS::AttachBufferOffsets(const OffsetBuffer& input,
                                           const OffsetBuffer& output,
                                           const OffsetBuffer& encoded_params,
                                           const OffsetBuffer& metadata) {

    MLI_ASSERT(input.get_size() / input.get_elem_size() == output.get_size() / output.get_elem_size());
    MLI_ASSERT(m_encoded_params_buffer_size == encoded_params.get_size());

    m_input.set_buf(input);
    m_output.set_buf(output);
    m_enc_param.set_buf(encoded_params);

    return MLI_STATUS_OK;
}

mli_status Rescale_CS::GetKernelPrivateData( void *kernel_private_data_buffer ) {

    RescalePrivateData opaque_obj;
    opaque_obj.size = sizeof(RescalePrivateData);

    assert(m_input.get_tensor().get_rank() == m_output.get_tensor().get_rank());
    
    for (uint32_t i = 0; i < m_input.get_tensor().get_rank(); i++) {
      MLI_ASSERT(m_output.get_dim(i) == m_input.get_dim(i));
    }
    opaque_obj.input = m_input;
    opaque_obj.output = m_output;
    opaque_obj.enc_param = m_enc_param;
    opaque_obj.rescale_axis = m_config.axis;

    std::memcpy(kernel_private_data_buffer, (void *)&opaque_obj, sizeof(opaque_obj));

    return MLI_STATUS_OK;
}

mli_status Rescale_CS::EncodeParams(const Tensor<Buffer, kRescaleParamRank> &in_bias,
                                    const Tensor<Buffer, kRescaleParamRank> &scale,
                                    const Tensor<Buffer, kRescaleParamRank> &shift,
                                    const Tensor<Buffer, kRescaleParamRank> &out_bias,
                                    Buffer &encoded_params) {

    MLI_ASSERT(m_params_elem_num == in_bias.get_dim(0));
    MLI_ASSERT(m_params_elem_num == scale.get_dim(0));
    MLI_ASSERT(m_params_elem_num == shift.get_dim(0));
    MLI_ASSERT(m_params_elem_num == out_bias.get_dim(0));

    uint32_t in_bias_elem_size = in_bias.get_elem_size();
    uint32_t out_bias_elem_size = out_bias.get_elem_size();
    m_encoded_params_buffer_size = m_params_elem_num *
      (in_bias_elem_size + sizeof(int16_t) + sizeof(int8_t) + out_bias_elem_size);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < m_params_elem_num; i++) {
      if (in_bias_elem_size == sizeof(int8_t)) {
        encoded_params.write_obj(offset, in_bias.read<int8_t>(i));
      }
      else if (in_bias_elem_size == sizeof(int32_t)) {
        encoded_params.write_obj(offset, in_bias.read<int32_t>(i));
      }
      else {
        MLI_ASSERT(0);
      }
      offset += in_bias_elem_size;
    }

    for (uint32_t i = 0; i < m_params_elem_num; i++) {
      encoded_params.write_obj(offset, scale.read<int16_t>(i));
      offset += sizeof(int16_t);
    }

    for (uint32_t i = 0; i < m_params_elem_num; i++) {
      encoded_params.write_obj(offset, shift.read<int8_t>(i));
      offset += sizeof(int8_t);
    }

    for (uint32_t i = 0; i < m_params_elem_num; i++) {
      if (out_bias_elem_size == sizeof(int8_t)) {
        encoded_params.write_obj(offset, out_bias.read<int8_t>(i));
      }
      else if (out_bias_elem_size == sizeof(int32_t)) {
        encoded_params.write_obj(offset, out_bias.read<int32_t>(i));
      }
      else {
        MLI_ASSERT(0);
      }
      offset += out_bias_elem_size;
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

}  // namespace snps_arc::metaware::mli::ref

/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include <cstring>

#include "mli_ref_compiler_api.hpp"
#include "mli_ref_runtime_api.hpp"
#include "mli_ref_private_types.hpp"

namespace snps_arc::metaware::mli::ref {

Prelu_CS::Prelu_CS(const lib_mli::PlatformDescription pd,
                   const TensorIterator<NoBuffer, 4, 4> &input,
                   const PreluOpConfig &cfg,
                   const TensorIterator<NoBuffer, 4, 4> &output)
                   : m_pd(pd) {
    DEPRECATED_METHOD
    OffsetBuffer tmp_buffer;
    Tensor<OffsetBuffer, 4> tmp_in_tensor = Tensor<OffsetBuffer, 4>(tmp_buffer ,input.get_tensor());
    Tensor<OffsetBuffer, 4> tmp_out_tensor = Tensor<OffsetBuffer, 4>(tmp_buffer ,output.get_tensor());
    Tensor<OffsetBuffer, kPreluRank> in_tns = tmp_in_tensor.split(3, 1, true);
    Tensor<OffsetBuffer, kPreluRank> out_tns = tmp_out_tensor.split(3, 1, true);
    m_input = TensorIterator<OffsetBuffer, kPreluRank, kPreluIterRank>(in_tns);
    m_output = TensorIterator<OffsetBuffer, kPreluRank, kPreluIterRank>(out_tns);
    uint32_t io_rank = m_input.get_tensor().get_rank();
    MLI_ASSERT(io_rank == m_output.get_tensor().get_rank());
    MLI_ASSERT(io_rank <= kPreluRank);

    uint32_t in_shape[kPreluRank];
    m_input.get_full_shape(in_shape);

    m_config.axis = io_rank - 1;
    
    // size_in_bytes = No.of elements multplied by params elements' sizes
    m_encoded_params_buffer_size = in_shape[io_rank - 1] *
            (sizeof(int32_t) + sizeof(int16_t) + sizeof(int16_t) + sizeof(int8_t) + sizeof(int8_t) + sizeof(int8_t));
}

Prelu_CS::Prelu_CS(const lib_mli::PlatformDescription pd,
                   const TensorIterator<NoBuffer, kPreluRank, kPreluIterRank> &input,
                   const PreluOpConfig &cfg,
                   const TensorIterator<NoBuffer, kPreluRank, kPreluIterRank> &output)
                   : m_config(cfg),
                     m_input(input),
                     m_output(output),
                     m_pd(pd) {

    uint32_t io_rank = input.get_tensor().get_rank();
    MLI_ASSERT(io_rank == output.get_tensor().get_rank());
    MLI_ASSERT(io_rank <= kPreluRank);

    uint32_t in_shape[kPreluRank];
    uint32_t out_shape[kPreluRank];
    int32_t in_stride[kPreluRank];
    int32_t out_stride[kPreluRank];
    input.get_full_shape(in_shape);
    input.get_mem_strides(in_stride);
    output.get_full_shape(out_shape);
    output.get_mem_strides(out_stride);
    
    // size_in_bytes = No.of elements multplied by params elements' sizes
    m_encoded_params_buffer_size = in_shape[io_rank - 1] *
            (sizeof(int32_t) + sizeof(int16_t) + sizeof(int16_t) + sizeof(int8_t) + sizeof(int8_t) + sizeof(int8_t));
}

mli_status Prelu_CS::AttachBufferOffsets(const OffsetBuffer &input,
                                         const OffsetBuffer &output,
                                         const OffsetBuffer &params,
                                         const OffsetBuffer &ctrl_buffer) {
    m_input.set_buf(input);
    m_output.set_buf(output);
    m_encoded_params = params;

    return MLI_STATUS_OK;
}

mli_status Prelu_CS::GetKernelPrivateData( void *kernel_private_data_buffer ) {

    PreluPrivateData opaque_obj;
    MLI_ASSERT(kernel_private_data_buffer != nullptr);
    MLI_ASSERT(m_input.get_tensor().get_rank() == m_output.get_tensor().get_rank());

    for(uint32_t i = 0; i < m_input.get_tensor().get_rank(); i++) {
        MLI_ASSERT(m_input.get_dim(i) == m_output.get_dim(i));
    }
    uint32_t io_rank = m_input.get_tensor().get_rank();
    opaque_obj.input = m_input;
    opaque_obj.output = m_output;
    opaque_obj.encoded_params_buffer = m_encoded_params;
    
    opaque_obj.tile_params_max_elem_num = (uint32_t) MAX(m_input.get_config().get_first_inc(io_rank - 1),
                                                         m_input.get_config().get_inc(io_rank - 1));
    if (!opaque_obj.tile_params_max_elem_num) opaque_obj.tile_params_max_elem_num = m_input.get_tensor().get_dim(kGroupTensorChannelDim);
    MLI_ASSERT(opaque_obj.tile_params_max_elem_num > 0);
    // Prelu configuration
    opaque_obj.prelu_axis = m_config.axis;

    std::memcpy(kernel_private_data_buffer, (void *)&opaque_obj, opaque_obj.size);

    return MLI_STATUS_OK;
}

mli_status Prelu_CS::EncodeParams(Tensor<Buffer, kPreluParamRank> &bias,
                                  Tensor<Buffer, kPreluParamRank> &posscale,
                                  Tensor<Buffer, kPreluParamRank> &negscale,
                                  Tensor<Buffer, kPreluParamRank> &posshift,
                                  Tensor<Buffer, kPreluParamRank> &negshift,
                                  Tensor<Buffer, kPreluParamRank> &asymm,
                                  Buffer &encoded_params) {
    uint32_t i, j, last_count;
    uint32_t in_bias_size = bias.get_buf().get_size();
    uint32_t posscale_size = posscale.get_buf().get_size();
    uint32_t negscale_size = negscale.get_buf().get_size();
    uint32_t posshift_size = posshift.get_buf().get_size();
    uint32_t negshift_size = negshift.get_buf().get_size();
    uint32_t out_bias_size = asymm.get_buf().get_size();

    for (i = 0, j = 0; i < in_bias_size; i+=4, j++) {
        int32_t val = bias.read<int32_t>(j);
        int8_t* pval = reinterpret_cast<int8_t*>(&val);
        encoded_params.write<int8_t>(i + 0, *(pval + 0));
        encoded_params.write<int8_t>(i + 1, *(pval + 1));
        encoded_params.write<int8_t>(i + 2, *(pval + 2));
        encoded_params.write<int8_t>(i + 3, *(pval + 3));
    }
    for (last_count = i, j = 0; i < (posscale_size + last_count); i+=2, j++) {
        int16_t val = posscale.read<int16_t>(j);
        int8_t* pval = reinterpret_cast<int8_t*>(&val);
        encoded_params.write<int8_t>(i + 0, *(pval + 0));
        encoded_params.write<int8_t>(i + 1, *(pval + 1));
    }
    for (last_count = i, j = 0; i < (negscale_size + last_count); i+=2, j++) {
        int16_t val = negscale.read<int16_t>(j);
        int8_t* pval = reinterpret_cast<int8_t*>(&val);
        encoded_params.write<int8_t>(i + 0, *(pval + 0));
        encoded_params.write<int8_t>(i + 1, *(pval + 1));
    }
    for (last_count = i, j = 0; i < (posshift_size + last_count); i+=1, j++) {
        encoded_params.write<int8_t>(i, posshift.read<int8_t>(j));
    }
    for (last_count = i, j = 0; i < (negshift_size + last_count); i+=1, j++) {
        encoded_params.write<int8_t>(i, negshift.read<int8_t>(j));
    }
    for (last_count = i, j = 0; i < (out_bias_size + last_count); i+=1, j++) {
        encoded_params.write<int8_t>(i, asymm.read<int8_t>(j));
    }

    return MLI_STATUS_OK;
}
      
unsigned Prelu_CS::GetKernelPrivateDataSize() const {
    return sizeof(PreluPrivateData);
}

unsigned Prelu_CS::GetRuntimeObjectSize() const {
    return sizeof(Prelu);
}

unsigned Prelu_CS::GetParamsBufferSize() {
    return 0;
}

unsigned Prelu_CS::GetEncodedParamsSize() {
    return m_encoded_params_buffer_size;
}

}
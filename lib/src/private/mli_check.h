/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_CHECK_H_
#define _MLI_CHECK_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "mli_types.h"

/**
 * @brief
 * @param[in]
 * @param[in]
 * @param[out]
 * @return
 *
 * @details
 *
 *
 *
 */
    mli_status mli_chk_tensor(const mli_tensor * in, bool check_bank = true);

    mli_status mli_chk_lut(const mli_lut * lut, int buff_size);


/**
 * @brief
 * @param[in]
 * @param[in]
 * @param[out]
 * @return
 *
 * @details
 *
 *
 *
 */
mli_status mli_chk_conv2d_hwcn_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_conv2d_hwcn_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_conv2d_hwcn_fx16_fx8_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_depthwise_conv2d_hwcn_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_depthwise_conv2d_hwcn_fx16_fx8_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_depthwise_conv2d_hwcn_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_group_conv2d_hwcn_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_group_conv2d_hwcn_fx16_fx8_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_group_conv2d_hwcn_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_transpose_conv2d_hwcn_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_transpose_conv2d_hwcn_fx16_fx8_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_transpose_conv2d_hwcn_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_transpose_conv2d_hwcn_k2x2_str2(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out);

mli_status mli_chk_transpose_conv2d_hwcn_k4x4_str2(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out);

mli_status mli_chk_maxpool_hwc_fx8(
        const mli_tensor *in,
        const mli_pool_cfg *cfg,
        const mli_tensor *out,
        const uint32_t kernel_size = 0);
mli_status mli_chk_maxpool_hwc_fx16(
        const mli_tensor *in,
        const mli_pool_cfg *cfg,
        const mli_tensor *out,
        const uint32_t kernel_size = 0);
mli_status mli_chk_maxpool_hwc_sa8(
        const mli_tensor *in,
        const mli_pool_cfg *cfg,
        const mli_tensor *out,
        const uint32_t kernel_size = 0);

mli_status mli_chk_avepool_hwc_fx8(
        const mli_tensor *in,
        const mli_pool_cfg *cfg,
        const mli_tensor *out,
        const uint32_t kernel_size = 0);
mli_status mli_chk_avepool_hwc_fx16(
        const mli_tensor *in,
        const mli_pool_cfg *cfg,
        const mli_tensor *out,
        const uint32_t kernel_size = 0);
mli_status mli_chk_avepool_hwc_sa8(
        const mli_tensor *in,
        const mli_pool_cfg *cfg,
        const mli_tensor *out,
        const uint32_t kernel_size = 0);
        
mli_status mli_chk_fully_connected_fx8w16d(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_fully_connected_cfg * cfg,
        mli_tensor * out);

mli_status mli_chk_fully_connected_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_fully_connected_cfg * cfg,
        mli_tensor * out);

mli_status mli_chk_fully_connected_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_fully_connected_cfg * cfg,
        mli_tensor * out);

mli_status mli_chk_fully_connected_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_fully_connected_cfg * cfg,
        mli_tensor * out);

mli_status mli_chk_relu_fx8(const mli_tensor * in, const mli_relu_cfg * cfg, mli_tensor * out);
mli_status mli_chk_relu_fx16(const mli_tensor * in, const mli_relu_cfg * cfg, mli_tensor * out);
mli_status mli_chk_relu_sa8(const mli_tensor * in, const mli_relu_cfg * cfg, mli_tensor * out);

mli_status mli_chk_basic_activation_fx8(const mli_tensor * in, mli_tensor * out);
mli_status mli_chk_basic_activation_fx16(const mli_tensor * in, mli_tensor * out);
mli_status mli_chk_basic_activation_sa8(const mli_tensor * in, mli_tensor * out);
mli_status mli_chk_softmax_fx8(const mli_tensor * in, const mli_softmax_cfg* cfg, mli_tensor * out);
mli_status mli_chk_softmax_fx16(const mli_tensor * in, const mli_softmax_cfg* cfg, mli_tensor * out);
mli_status mli_chk_softmax_sa8(const mli_tensor * in, const mli_softmax_cfg* cfg, mli_tensor * out);
mli_status mli_chk_l2_normalize_fx16(const mli_tensor * in, const mli_l2_normalize_cfg* cfg, mli_tensor * out);
mli_status mli_chk_l2_normalize_sa8(const mli_tensor * in, const mli_l2_normalize_cfg* cfg, mli_tensor * out);
mli_status mli_chk_leaky_relu(const mli_tensor * in, const mli_tensor * slope_coeff, mli_tensor * out);
mli_status mli_chk_leaky_relu_fx8(const mli_tensor * in, const mli_tensor * slope_coeff, mli_tensor * out);
mli_status mli_chk_leaky_relu_fx16(const mli_tensor * in, const mli_tensor * slope_coeff, mli_tensor * out);
mli_status mli_chk_leaky_relu_sa8(const mli_tensor * in, const mli_tensor * slope_coeff, mli_tensor * out);
mli_status mli_chk_eltwise_fx8(const mli_tensor * left, const mli_tensor * right, mli_tensor * out);
mli_status mli_chk_eltwise_fx16(const mli_tensor * left, const mli_tensor * right, mli_tensor * out);
mli_status mli_chk_eltwise_sa8(const mli_tensor * left, const mli_tensor * right, mli_tensor * out);
mli_status mli_chk_eltwise_maxmin_fx8(const mli_tensor * left, const mli_tensor * right, mli_tensor * out);
mli_status mli_chk_eltwise_maxmin_fx16(const mli_tensor * left, const mli_tensor * right, mli_tensor * out);
mli_status mli_chk_eltwise_maxmin_sa8(const mli_tensor * left, const mli_tensor * right, mli_tensor * out);

mli_status mli_chk_prelu_fx8(
        const mli_tensor * in, 
        const mli_tensor * slope_coeff, 
        const mli_prelu_cfg *cfg, 
        mli_tensor * out);

mli_status mli_chk_prelu_fx16(
        const mli_tensor * in, 
        const mli_tensor * slope_coeff, 
        const mli_prelu_cfg *cfg, 
        mli_tensor * out);

mli_status mli_chk_prelu_sa8(
        const mli_tensor * in, 
        const mli_tensor * slope_coeff, 
        const mli_prelu_cfg *cfg, 
        mli_tensor * out);

mli_status mli_chk_rnn_dense_fx16(
        const mli_tensor **in,
        const mli_tensor **weights,
        const mli_tensor *bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out);

mli_status mli_chk_rnn_dense_fx16_fx8_fx8(
        const mli_tensor **in,
        const mli_tensor **weights,
        const mli_tensor *bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out);

mli_status mli_chk_rnn_dense_sa8_sa8_sa32(
        const mli_tensor **in,
        const mli_tensor **weights,
        const mli_tensor *bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out);

mli_status mli_chk_lstm_cell_fx16(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * cell,
        mli_tensor * out);

mli_status mli_chk_lstm_cell_fx16_fx8_fx8(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * cell,
        mli_tensor * out);

mli_status mli_chk_lstm_cell_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * cell,
        mli_tensor * out);

mli_status mli_chk_gru_cell_fx16(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out);

mli_status mli_chk_gru_cell_fx16_fx8_fx8(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out);

mli_status mli_chk_gru_cell_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out);

mli_status mli_chk_concat(const mli_tensor ** inputs, const mli_concat_cfg * cfg, mli_tensor * out);
mli_status mli_chk_concat_fx8(const mli_tensor ** inputs, const mli_concat_cfg * cfg, mli_tensor * out);
mli_status mli_chk_concat_fx16(const mli_tensor ** inputs, const mli_concat_cfg * cfg, mli_tensor * out);
mli_status mli_chk_padding2d_chw(const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out);
mli_status mli_chk_padding2d_chw_fx8(const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out);
mli_status mli_chk_padding2d_chw_fx16(const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out);
mli_status mli_chk_padding2d_hwc(const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out);
mli_status mli_chk_padding2d_hwc_fx8(const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out);
mli_status mli_chk_padding2d_hwc_fx16(const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out);
mli_status mli_chk_permute(const mli_tensor * in, const mli_permute_cfg * cfg, mli_tensor * out);
mli_status mli_chk_permute_sa8(const mli_tensor * in, const mli_permute_cfg * cfg, mli_tensor * out);
mli_status mli_chk_permute_fx8(const mli_tensor * in, const mli_permute_cfg * cfg, mli_tensor * out);
mli_status mli_chk_permute_fx16(const mli_tensor * in, const mli_permute_cfg * cfg, mli_tensor * out);

mli_status mli_chk_count_elem_num(const mli_tensor *in, uint32_t start_dim);
mli_status mli_chk_convert_tensor(const mli_tensor *in, mli_tensor *out);
mli_status mli_chk_point_to_subtensor(const mli_tensor *in, const mli_point_to_subtsr_cfg *cfg, mli_tensor *out);
mli_status mli_chk_create_subtensor(const mli_tensor *in, const mli_sub_tensor_cfg *cfg, mli_tensor *out);
mli_status mli_chk_data_movement(const mli_tensor *in, const mli_mov_cfg_t *cfg, mli_tensor *out);
mli_status mli_chk_data_movement_dst_tensor(const mli_tensor *t);

mli_status mli_chk_argmax_sa8(const mli_tensor *in, const mli_argmax_cfg *cfg, mli_tensor *out);
mli_status mli_chk_argmax_fx16(const mli_tensor *in, const mli_argmax_cfg *cfg, mli_tensor *out);

#ifdef __cplusplus
}
#endif

#endif                          // _MLI_CHECK_H_

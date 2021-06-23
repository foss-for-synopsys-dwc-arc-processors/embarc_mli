/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

/**
 *  @author Yaroslav Donskov <yaroslav@synopsys.com>
 */


#include <cstdio>
#include <cassert>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "mli_api.h"
#include "mli_config.h"
#include "weights.h"
#include "util.h"
#include "model.h"
#ifndef _MSC_VER
#include <arc/arc_timer.h>
#define T0_COUNT 0x21
#endif  // !_MSC_VER

#define NUM_TILES_Y 2
#define NUM_TILES_X 2
#define TILE_OVERLAP_Y 8
#define TILE_OVERLAP_X 8
	
#define VIRTUAL_BUFFER_SIZE 22528
#define MAX_IR_SIZE 114688
#define MAX_TILING_CHANNELS 28
#define TILING_INPUT_WIDTH ( IMAGE_SIDE / NUM_TILES_X + TILE_OVERLAP_X )
#define TILING_INPUT_HEIGHT ( IMAGE_SIDE / NUM_TILES_Y + TILE_OVERLAP_Y )
#define TILE_OUTPUT_WIDTH 32
#define TILE_OUTPUT_HEIGHT 32
#define TILE_OUTPUT_CHANNELS 32
#define MAX_IR_WIDTH_TILING (64 / NUM_TILES_X + TILE_OVERLAP_X)
#define MAX_IR_HEIGHT_TILING (64 / NUM_TILES_Y + TILE_OVERLAP_Y)
#define MAX_IR_SIZE_TILING ( MAX_IR_HEIGHT_TILING * MAX_IR_WIDTH_TILING * MAX_TILING_CHANNELS )
#define MAX(a, b) ((a) > (b) ? (a) : b)
#define BUFFER_SIZE  ( MAX( (VIRTUAL_BUFFER_SIZE * 6), (MAX_IR_SIZE_TILING * 3 + 2 * VIRTUAL_BUFFER_SIZE) ) )
//============================================================
static d_type _X tensors_buffer[BUFFER_SIZE];
static mli_tensor x_tensor;
static mli_tensor y_tensor;
static mli_tensor z_tensor;
static mli_tensor w_tensor;
static mli_tensor q_tensor;
static mli_tensor v_tensor;
//============================================================
static const mli_conv2d_cfg first_conv_cfg = {
	{MLI_RELU_GEN},
	2, 2,
	1, 2, 1, 2
};
static const mli_conv2d_cfg conv_cfg = {
	{MLI_RELU_NONE},
	1, 1
};
static const mli_conv2d_cfg dconv_cfg = {
	{MLI_RELU_NONE},
	1, 1,
	1, 1, 1, 1
};
static const mli_conv2d_cfg dconv_cfg_stride = {
	{MLI_RELU_NONE},
	2, 2,
	0, 1, 0, 1
};
static const mli_pool_cfg pool_cfg = {
	2, 2, 2, 2
};
//============================================================
static mli_tensor conv_weights_tensor;
static mli_tensor conv_bias_tensor;
static mli_tensor conv_1_weights_tensor;
static mli_tensor conv_1_bias_tensor;
static mli_tensor conv_2_weights_tensor;
static mli_tensor conv_2_bias_tensor;
static mli_tensor conv_3_weights_tensor;
static mli_tensor conv_3_bias_tensor;
static mli_tensor conv_4_weights_tensor;
static mli_tensor conv_4_bias_tensor;
static mli_tensor conv_5_weights_tensor;
static mli_tensor conv_5_bias_tensor;
static mli_tensor conv_6_weights_tensor;
static mli_tensor conv_6_bias_tensor;
static mli_tensor conv_7_weights_tensor;
static mli_tensor conv_7_bias_tensor;
static mli_tensor conv_8_weights_tensor;
static mli_tensor conv_8_bias_tensor;
static mli_tensor conv_9_weights_tensor;
static mli_tensor conv_9_bias_tensor;
static mli_tensor conv_10_weights_tensor;
static mli_tensor conv_10_bias_tensor;
static mli_tensor conv_11_weights_tensor;
static mli_tensor conv_11_bias_tensor;
static mli_tensor conv_12_weights_tensor;
static mli_tensor conv_12_bias_tensor;
static mli_tensor conv_13_weights_tensor;
static mli_tensor conv_13_bias_tensor;
static mli_tensor conv_14_weights_tensor;
static mli_tensor conv_14_bias_tensor;
static mli_tensor conv_15_weights_tensor;
static mli_tensor conv_15_bias_tensor;
static mli_tensor conv_16_weights_tensor;
static mli_tensor conv_16_bias_tensor;
static mli_tensor conv_17_weights_tensor;
static mli_tensor conv_17_bias_tensor;
static mli_tensor conv_18_weights_tensor;
static mli_tensor conv_18_bias_tensor;
static mli_tensor conv_19_weights_tensor;
static mli_tensor conv_19_bias_tensor;
static mli_tensor conv_20_weights_tensor;
static mli_tensor conv_20_bias_tensor;
static mli_tensor dconv_1_weights_tensor;
static mli_tensor dconv_1_bias_tensor;
static mli_tensor dconv_2_weights_tensor;
static mli_tensor dconv_2_bias_tensor;
static mli_tensor dconv_3_weights_tensor;
static mli_tensor dconv_3_bias_tensor;
static mli_tensor dconv_4_weights_tensor;
static mli_tensor dconv_4_bias_tensor;
static mli_tensor dconv_5_weights_tensor;
static mli_tensor dconv_5_bias_tensor;
static mli_tensor dconv_6_weights_tensor;
static mli_tensor dconv_6_bias_tensor;
static mli_tensor dconv_7_weights_tensor;
static mli_tensor dconv_7_bias_tensor;
static mli_tensor dconv_8_weights_tensor;
static mli_tensor dconv_8_bias_tensor;
static mli_tensor dconv_9_weights_tensor;
static mli_tensor dconv_9_bias_tensor;
static mli_tensor dconv_10_weights_tensor;
static mli_tensor dconv_10_bias_tensor;
static mli_tensor dconv_11_weights_tensor;
static mli_tensor dconv_11_bias_tensor;
static mli_tensor dconv_12_weights_tensor;
static mli_tensor dconv_12_bias_tensor;
static mli_tensor dconv_13_weights_tensor;
static mli_tensor dconv_13_bias_tensor;
static mli_tensor dconv_14_weights_tensor;
static mli_tensor dconv_14_bias_tensor;
static mli_tensor dconv_15_weights_tensor;
static mli_tensor dconv_15_bias_tensor;
static mli_tensor dconv_16_weights_tensor;
static mli_tensor dconv_16_bias_tensor;
//============================================================

static void init_intermediate_tensor(mli_tensor * tensor, int size){
	tensor->data.capacity = size * sizeof(d_type);
	tensor->el_type = D_EL_TYPE;
	tensor->rank = 3;
	tensor->el_params.sa.dim = -1;	
}

#include "idx_file.h"
void DebugWriteResultToFile(const char* filename, const mli_tensor* tensor) {
#if 1
  uint32_t arr_size = mli_hlp_count_elem_num(tensor, 0);
  assert(tensor->el_type == MLI_EL_SA_8);
  tIdxDataType d_type = IDX_DT_BYTE_1B;

  tIdxRetVal ret_status = idx_file_write_completely(
      filename, (void *)tensor->data.mem.pi8, &arr_size, 1, d_type);
  if (ret_status != IDX_ERR_NONE)
    printf("TENSOR FILE WRITE FAILED: %s", filename);
#endif
}


static void init_conv_w_tensor(
	mli_tensor * tensor, void * w, void * s, void * fb,
	int kx, int ky, int ic, int oc
	){
	tensor->data.mem.void_p = w;
	tensor->data.capacity = kx * ky * ic * oc * sizeof(w_type);
	tensor->rank = CONV_W_RANK;
	tensor->shape[0] = kx;
	tensor->shape[1] = ky;
	tensor->shape[2] = ic;
	tensor->shape[3] = oc;	
	tensor->el_type = W_EL_TYPE;
	tensor->el_params.sa.dim = 3;
	tensor->el_params.sa.scale.mem.void_p = s;
	tensor->el_params.sa.scale_frac_bits.mem.void_p = fb;
	tensor->el_params.sa.zero_point.mem.void_p = (void*) zeros;
}

static void init_conv_b_tensor(
	mli_tensor * tensor, void * b, void * s, void * fb,
	int oc
	){
	tensor->data.mem.void_p = b;
	tensor->data.capacity = oc * sizeof(b_type);
	tensor->rank = CONV_B_RANK;
	tensor->shape[0] = oc;
	tensor->el_type = B_EL_TYPE;
	tensor->el_params.sa.dim = 0;
	tensor->el_params.sa.scale.mem.void_p = s;
	tensor->el_params.sa.scale_frac_bits.mem.void_p = fb;
	tensor->el_params.sa.zero_point.mem.void_p = (void*) zeros;
}

static void init_tensors() {

	init_intermediate_tensor(&x_tensor, 2 * VIRTUAL_BUFFER_SIZE);
	init_intermediate_tensor(&y_tensor, 2 * VIRTUAL_BUFFER_SIZE);
	init_intermediate_tensor(&z_tensor, 2 * VIRTUAL_BUFFER_SIZE);
	init_intermediate_tensor(&w_tensor, 2 * VIRTUAL_BUFFER_SIZE);
	init_intermediate_tensor(&q_tensor, VIRTUAL_BUFFER_SIZE);
	init_intermediate_tensor(&v_tensor, VIRTUAL_BUFFER_SIZE);

	// 1st conv
	init_conv_w_tensor(
		&conv_weights_tensor, (void*) conv_weights, (void*) conv_weights_scales, 
		(void*) conv_weights_fb,
		LAYER_1_KX, LAYER_1_KY, LAYER_1_IC, LAYER_1_OC
	);
	init_conv_b_tensor(
		&conv_bias_tensor, (void*) conv_bias, (void*) conv_bias_scales,
		(void*) conv_bias_fb, LAYER_1_OC
	);

	// blaze block 1
	init_conv_w_tensor(
		&dconv_1_weights_tensor, (void*)dconv_1_weights, (void*)dconv_1_weights_scales,
		(void*)dconv_1_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_1_IC
	);
	init_conv_b_tensor(
		&dconv_1_bias_tensor, (void*)dconv_1_bias, (void*)dconv_1_bias_scales,
		(void*)dconv_1_bias_fb, BB_1_IC
	);
	init_conv_w_tensor(
		&conv_1_weights_tensor, (void*)conv_1_weights, (void*)conv_1_weights_scales,
		(void*)conv_1_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_1_IC, BB_1_OC
	);
	init_conv_b_tensor(
		&conv_1_bias_tensor, (void*)conv_1_bias, (void*)conv_1_bias_scales,
		(void*)conv_1_bias_fb, BB_1_OC
	);

	// blaze block 2
	init_conv_w_tensor(
		&dconv_2_weights_tensor, (void*)dconv_2_weights, (void*)dconv_2_weights_scales,
		(void*)dconv_2_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_2_IC
	);
	init_conv_b_tensor(
		&dconv_2_bias_tensor, (void*)dconv_2_bias, (void*)dconv_2_bias_scales,
		(void*)dconv_2_bias_fb, BB_2_IC
	);
	init_conv_w_tensor(
		&conv_2_weights_tensor, (void*)conv_2_weights, (void*)conv_2_weights_scales,
		(void*)conv_2_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_2_IC, BB_2_OC
	);
	init_conv_b_tensor(
		&conv_2_bias_tensor, (void*)conv_2_bias, (void*)conv_2_bias_scales,
		(void*)conv_2_bias_fb, BB_2_OC
	);

	// blaze block 3
	init_conv_w_tensor(
		&dconv_3_weights_tensor, (void*)dconv_3_weights, (void*)dconv_3_weights_scales,
		(void*)dconv_3_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_3_IC
	);
	init_conv_b_tensor(
		&dconv_3_bias_tensor, (void*)dconv_3_bias, (void*)dconv_3_bias_scales,
		(void*)dconv_3_bias_fb, BB_3_IC
	);
	init_conv_w_tensor(
		&conv_3_weights_tensor, (void*)conv_3_weights, (void*)conv_3_weights_scales,
		(void*)conv_3_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_3_IC, BB_3_OC
	);
	init_conv_b_tensor(
		&conv_3_bias_tensor, (void*)conv_3_bias, (void*)conv_3_bias_scales,
		(void*)conv_3_bias_fb, BB_3_OC
	);

	// blaze block 4
	init_conv_w_tensor(
		&dconv_4_weights_tensor, (void*)dconv_4_weights, (void*)dconv_4_weights_scales,
		(void*)dconv_4_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_4_IC
	);
	init_conv_b_tensor(
		&dconv_4_bias_tensor, (void*)dconv_4_bias, (void*)dconv_4_bias_scales,
		(void*)dconv_4_bias_fb, BB_4_IC
	);
	init_conv_w_tensor(
		&conv_4_weights_tensor, (void*)conv_4_weights, (void*)conv_4_weights_scales,
		(void*)conv_4_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_4_IC, BB_4_OC
	);
	init_conv_b_tensor(
		&conv_4_bias_tensor, (void*)conv_4_bias, (void*)conv_4_bias_scales,
		(void*)conv_4_bias_fb, BB_4_OC
	);

	// blaze block 5
	init_conv_w_tensor(
		&dconv_5_weights_tensor, (void*)dconv_5_weights, (void*)dconv_5_weights_scales,
		(void*)dconv_5_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_5_IC
	);
	init_conv_b_tensor(
		&dconv_5_bias_tensor, (void*)dconv_5_bias, (void*)dconv_5_bias_scales,
		(void*)dconv_5_bias_fb, BB_5_IC
	);
	init_conv_w_tensor(
		&conv_5_weights_tensor, (void*)conv_5_weights, (void*)conv_5_weights_scales,
		(void*)conv_5_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_5_IC, BB_5_OC
	);
	init_conv_b_tensor(
		&conv_5_bias_tensor, (void*)conv_5_bias, (void*)conv_5_bias_scales,
		(void*)conv_5_bias_fb, BB_5_OC
	);

	// blaze block 6
	init_conv_w_tensor(
		&dconv_6_weights_tensor, (void*)dconv_6_weights, (void*)dconv_6_weights_scales,
		(void*)dconv_6_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_6_IC
	);
	init_conv_b_tensor(
		&dconv_6_bias_tensor, (void*)dconv_6_bias, (void*)dconv_6_bias_scales,
		(void*)dconv_6_bias_fb, BB_6_IC
	);
	init_conv_w_tensor(
		&conv_6_weights_tensor, (void*)conv_6_weights, (void*)conv_6_weights_scales,
		(void*)conv_6_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_6_IC, BB_6_OC
	);
	init_conv_b_tensor(
		&conv_6_bias_tensor, (void*)conv_6_bias, (void*)conv_6_bias_scales,
		(void*)conv_6_bias_fb, BB_6_OC
	);

	// blaze block 7
	init_conv_w_tensor(
		&dconv_7_weights_tensor, (void*)dconv_7_weights, (void*)dconv_7_weights_scales,
		(void*)dconv_7_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_7_IC
	);
	init_conv_b_tensor(
		&dconv_7_bias_tensor, (void*)dconv_7_bias, (void*)dconv_7_bias_scales,
		(void*)dconv_7_bias_fb, BB_7_IC
	);
	init_conv_w_tensor(
		&conv_7_weights_tensor, (void*)conv_7_weights, (void*)conv_7_weights_scales,
		(void*)conv_7_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_7_IC, BB_7_OC
	);
	init_conv_b_tensor(
		&conv_7_bias_tensor, (void*)conv_7_bias, (void*)conv_7_bias_scales,
		(void*)conv_7_bias_fb, BB_7_OC
	);

	// blaze block 8
	init_conv_w_tensor(
		&dconv_8_weights_tensor, (void*)dconv_8_weights, (void*)dconv_8_weights_scales,
		(void*)dconv_8_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_8_IC
	);
	init_conv_b_tensor(
		&dconv_8_bias_tensor, (void*)dconv_8_bias, (void*)dconv_8_bias_scales,
		(void*)dconv_8_bias_fb, BB_8_IC
	);
	init_conv_w_tensor(
		&conv_8_weights_tensor, (void*)conv_8_weights, (void*)conv_8_weights_scales,
		(void*)conv_8_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_8_IC, BB_8_OC
	);
	init_conv_b_tensor(
		&conv_8_bias_tensor, (void*)conv_8_bias, (void*)conv_8_bias_scales,
		(void*)conv_8_bias_fb, BB_8_OC
	);

	// blaze block 9
	init_conv_w_tensor(
		&dconv_9_weights_tensor, (void*)dconv_9_weights, (void*)dconv_9_weights_scales,
		(void*)dconv_9_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_9_IC
	);
	init_conv_b_tensor(
		&dconv_9_bias_tensor, (void*)dconv_9_bias, (void*)dconv_9_bias_scales,
		(void*)dconv_9_bias_fb, BB_9_IC
	);
	init_conv_w_tensor(
		&conv_9_weights_tensor, (void*)conv_9_weights, (void*)conv_9_weights_scales,
		(void*)conv_9_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_9_IC, BB_9_OC
	);
	init_conv_b_tensor(
		&conv_9_bias_tensor, (void*)conv_9_bias, (void*)conv_9_bias_scales,
		(void*)conv_9_bias_fb, BB_9_OC
	);

	// blaze block 10
	init_conv_w_tensor(
		&dconv_10_weights_tensor, (void*)dconv_10_weights, (void*)dconv_10_weights_scales,
		(void*)dconv_10_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_10_IC
	);
	init_conv_b_tensor(
		&dconv_10_bias_tensor, (void*)dconv_10_bias, (void*)dconv_10_bias_scales,
		(void*)dconv_10_bias_fb, BB_10_IC
	);
	init_conv_w_tensor(
		&conv_10_weights_tensor, (void*)conv_10_weights, (void*)conv_10_weights_scales,
		(void*)conv_10_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_10_IC, BB_10_OC
	);
	init_conv_b_tensor(
		&conv_10_bias_tensor, (void*)conv_10_bias, (void*)conv_10_bias_scales,
		(void*)conv_10_bias_fb, BB_10_OC
	);

	// blaze block 11
	init_conv_w_tensor(
		&dconv_11_weights_tensor, (void*)dconv_11_weights, (void*)dconv_11_weights_scales,
		(void*)dconv_11_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_11_IC
	);
	init_conv_b_tensor(
		&dconv_11_bias_tensor, (void*)dconv_11_bias, (void*)dconv_11_bias_scales,
		(void*)dconv_11_bias_fb, BB_11_IC
	);
	init_conv_w_tensor(
		&conv_11_weights_tensor, (void*)conv_11_weights, (void*)conv_11_weights_scales,
		(void*)conv_11_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_11_IC, BB_11_OC
	);
	init_conv_b_tensor(
		&conv_11_bias_tensor, (void*)conv_11_bias, (void*)conv_11_bias_scales,
		(void*)conv_11_bias_fb, BB_11_OC
	);

	// blaze block 12
	init_conv_w_tensor(
		&dconv_12_weights_tensor, (void*)dconv_12_weights, (void*)dconv_12_weights_scales,
		(void*)dconv_12_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_12_IC
	);
	init_conv_b_tensor(
		&dconv_12_bias_tensor, (void*)dconv_12_bias, (void*)dconv_12_bias_scales,
		(void*)dconv_12_bias_fb, BB_12_IC
	);
	init_conv_w_tensor(
		&conv_12_weights_tensor, (void*)tensors_buffer, (void*)conv_12_weights_scales,
		(void*)conv_12_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_12_IC, BB_12_OC
	);
	init_conv_b_tensor(
		&conv_12_bias_tensor, (void*)conv_12_bias, (void*)conv_12_bias_scales,
		(void*)conv_12_bias_fb, BB_12_OC
	);

	// blaze block 13
	init_conv_w_tensor(
		&dconv_13_weights_tensor, (void*)dconv_13_weights, (void*)dconv_13_weights_scales,
		(void*)dconv_13_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_13_IC
	);
	init_conv_b_tensor(
		&dconv_13_bias_tensor, (void*)dconv_13_bias, (void*)dconv_13_bias_scales,
		(void*)dconv_13_bias_fb, BB_13_IC
	);
	init_conv_w_tensor(
		&conv_13_weights_tensor, (void*)tensors_buffer, (void*)conv_13_weights_scales,
		(void*)conv_13_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_13_IC, BB_13_OC
	);
	init_conv_b_tensor(
		&conv_13_bias_tensor, (void*)conv_13_bias, (void*)conv_13_bias_scales,
		(void*)conv_13_bias_fb, BB_13_OC
	);

	// blaze block 14
	init_conv_w_tensor(
		&dconv_14_weights_tensor, (void*)dconv_14_weights, (void*)dconv_14_weights_scales,
		(void*)dconv_14_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_14_IC
	);
	init_conv_b_tensor(
		&dconv_14_bias_tensor, (void*)dconv_14_bias, (void*)dconv_14_bias_scales,
		(void*)dconv_14_bias_fb, BB_14_IC
	);
	init_conv_w_tensor(
		&conv_14_weights_tensor, (void*)tensors_buffer, (void*)conv_14_weights_scales,
		(void*)conv_14_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_14_IC, BB_14_OC
	);
	init_conv_b_tensor(
		&conv_14_bias_tensor, (void*)conv_14_bias, (void*)conv_14_bias_scales,
		(void*)conv_14_bias_fb, BB_14_OC
	);

	// blaze block 15
	init_conv_w_tensor(
		&dconv_15_weights_tensor, (void*)dconv_15_weights, (void*)dconv_15_weights_scales,
		(void*)dconv_15_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_15_IC
	);
	init_conv_b_tensor(
		&dconv_15_bias_tensor, (void*)dconv_15_bias, (void*)dconv_15_bias_scales,
		(void*)dconv_15_bias_fb, BB_15_IC
	);
	init_conv_w_tensor(
		&conv_15_weights_tensor, (void*)tensors_buffer, (void*)conv_15_weights_scales,
		(void*)conv_15_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_15_IC, BB_15_OC
	);
	init_conv_b_tensor(
		&conv_15_bias_tensor, (void*)conv_15_bias, (void*)conv_15_bias_scales,
		(void*)conv_15_bias_fb, BB_15_OC
	);

	// blaze block 16
	init_conv_w_tensor(
		&dconv_16_weights_tensor, (void*)dconv_16_weights, (void*)dconv_16_weights_scales,
		(void*)dconv_16_weights_fb,
		BB_DCONV_KY, BB_DCONV_KX, 1, BB_16_IC
	);
	init_conv_b_tensor(
		&dconv_16_bias_tensor, (void*)dconv_16_bias, (void*)dconv_16_bias_scales,
		(void*)dconv_16_bias_fb, BB_16_IC
	);
	init_conv_w_tensor(
		&conv_16_weights_tensor, (void*)tensors_buffer, (void*)conv_16_weights_scales,
		(void*)conv_16_weights_fb,
		BB_CONV_KX, BB_CONV_KY, BB_16_IC, BB_16_OC
	);
	init_conv_b_tensor(
		&conv_16_bias_tensor, (void*)conv_16_bias, (void*)conv_16_bias_scales,
		(void*)conv_16_bias_fb, BB_16_OC
	);
	
	// 1st output conv
	init_conv_w_tensor(
		&conv_17_weights_tensor, (void*)conv_17_weights, (void*)conv_17_weights_scales,
		(void*)conv_17_weights_fb,
		BB_CONV_KX, BB_CONV_KY, OCONV_1_IC, OCONV_1_OC
	);
	init_conv_b_tensor(
		&conv_17_bias_tensor, (void*)conv_17_bias, (void*)conv_17_bias_scales,
		(void*)conv_17_bias_fb, OCONV_1_OC
	);
	
	// 2nd output conv
	init_conv_w_tensor(
		&conv_18_weights_tensor, (void*)conv_18_weights, (void*)conv_18_weights_scales,
		(void*)conv_18_weights_fb,
		BB_CONV_KX, BB_CONV_KY, OCONV_2_IC, OCONV_2_OC
	);
	init_conv_b_tensor(
		&conv_18_bias_tensor, (void*)conv_18_bias, (void*)conv_18_bias_scales,
		(void*)conv_18_bias_fb, OCONV_2_OC
	);

	// 3d output conv
	init_conv_w_tensor(
		&conv_19_weights_tensor, (void*)tensors_buffer, (void*)conv_19_weights_scales,
		(void*)conv_19_weights_fb,
		BB_CONV_KX, BB_CONV_KY, OCONV_3_IC, OCONV_3_OC
	);
	init_conv_b_tensor(
		&conv_19_bias_tensor, (void*)conv_19_bias, (void*)conv_19_bias_scales,
		(void*)conv_19_bias_fb, OCONV_3_OC
	);
	
	// 4th output conv
	init_conv_w_tensor(&conv_20_weights_tensor, (void*)tensors_buffer, (void*)conv_20_weights_scales,
		(void*)conv_20_weights_fb,
		BB_CONV_KX, BB_CONV_KY, OCONV_4_IC, OCONV_4_OC
	);
	init_conv_b_tensor(
		&conv_20_bias_tensor, (void*)conv_20_bias, (void*)conv_20_bias_scales,
		(void*)conv_20_bias_fb, OCONV_4_OC
	);

}

static void pad_channels_sa8(mli_tensor * tensor, uint32_t pad_c, int8_t zero_val){
	int h = tensor->shape[0];
	int w = tensor->shape[1];
	int c = tensor->shape[2];
	int new_c = (c + pad_c);
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			for (int k = c; k < new_c; k++){
				tensor->data.mem.pi8[i * w * new_c + j * new_c + k] = zero_val;
			}
		}
	}
}


static int blazeblock(
    mli_tensor* tensors[], int input_index,
    mli_tensor* dconv_weights,  mli_tensor* dconv_biases,
    mli_tensor* conv_weights,  mli_tensor* conv_biases,
	s_type dconv_scale, f_type dconv_fb, s_type dconv_zp,
	s_type conv_scale, f_type conv_fb, s_type conv_zp, 
	s_type relu_scale, f_type relu_fb, s_type relu_zp,
    bool reduce, bool reduce_next_block,
	int pad_channels, int pad_channels_next_block,
	bool profiling, bool detailed_profiling
){

#ifndef _MSC_VER
	unsigned t0, t1, t2, t3, t4, t6;
	if (profiling){
		t0 = _lr(T0_COUNT);
	}
#endif	// !_MSC_VER

	mli_tensor * input = tensors[input_index];
	int temp_1_index = 0;
	int temp_2_index = 0;
	mli_tensor * temp1 = nullptr;
	mli_tensor * temp2 = nullptr;
	for (int i = 0; i < 3; i++) {
		if (i == input_index) continue;
		if (temp1 == nullptr) {
			temp1 = tensors[i];
			temp_1_index = i;
		}
		else {
			temp2 = tensors[i];
			temp_2_index = i;
		}
	}

	// dconv 3x3: input -> temp1
	const mli_conv2d_cfg * cfg = reduce ? &dconv_cfg_stride : &dconv_cfg;
	temp1->el_params.sa.scale.mem.i16 = dconv_scale;
	temp1->el_params.sa.scale_frac_bits.mem.i8 = dconv_fb;
	temp1->el_params.sa.zero_point.mem.i16 = dconv_zp;
	mli_status status = mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32_k3x3(input, dconv_weights, dconv_biases, cfg, temp1);
	assert(status == MLI_STATUS_OK);

#ifndef _MSC_VER
	if (detailed_profiling){
		t1 = _lr(T0_COUNT);
		printf("dconv %.2f\n", (float)(t1 - t0) / (float) 1000000);
	}
#endif	// !_MSC_VER
	

	// conv 1x1: temp1 -> temp2
	temp2->el_params.sa.scale.mem.i16 = conv_scale;
	temp2->el_params.sa.scale_frac_bits.mem.i8 = conv_fb;
	temp2->el_params.sa.zero_point.mem.i16 = conv_zp;
	status = mli_krn_conv2d_hwcn_sa8_sa8_sa32_k1x1(temp1, conv_weights, conv_biases, &conv_cfg, temp2);
	assert(status == MLI_STATUS_OK);

#ifndef _MSC_VER
	if (detailed_profiling){
		t2 = _lr(T0_COUNT);
		printf("conv %.2f\n", (float)(t2 - t1) / (float) 1000000);
	}
#endif	// !_MSC_VER

	mli_tensor * buffer_a = input;
	mli_tensor * buffer_b = temp2;
	mli_tensor * buffer_c = temp1;
	int result_index = temp_1_index;
	if (reduce){
		// maxpool 2x2: input -> temp1
		if (pad_channels){
			// for channels padding in current block
			int c = input->shape[2] + pad_channels;
			temp1->mem_stride[0] = input->shape[1] / 2 * c;
			temp1->mem_stride[1] = c;
			temp1->mem_stride[2] = 1;
		}
		status = mli_krn_maxpool_hwc_sa8_k2x2(input, &pool_cfg, temp1);
		assert(status == MLI_STATUS_OK);
		buffer_a = temp1;
		buffer_c = input;
		result_index = input_index;
    }

#ifndef _MSC_VER
	if (detailed_profiling){
		t3 = _lr(T0_COUNT);
		printf("conv %.2f\n", (float)(t3 - t2) / (float) 1000000);
	}
#endif	// !_MSC_VER

	// pad
	assert( buffer_a->shape[0] == buffer_b->shape[0]);
	assert( buffer_a->shape[1] == buffer_b->shape[1]);
	if (pad_channels){		
		int8_t pad_zero_val = float2sa_(
			0.0f, buffer_a->el_params.sa.scale.mem.i16,
			buffer_a->el_params.sa.scale_frac_bits.mem.i8,
			buffer_a->el_params.sa.zero_point.mem.i16
		);
		// TODO: maybe try to change pad_channels_sa8 to eltwise mult part of buffer_a tensor with scalar zero
		pad_channels_sa8(buffer_a, pad_channels, pad_zero_val);
		buffer_a->shape[2] += pad_channels;
		buffer_a->mem_stride[0] = 0;
		buffer_a->mem_stride[1] = 0;
		buffer_a->mem_stride[2] = 0;
	}

#ifndef _MSC_VER
	if (detailed_profiling){
		t4 = _lr(T0_COUNT);
		printf("pad %.2f\n", (float)(t4 - t3) / (float) 1000000);
	}
#endif	// !_MSC_VER

	// add with relu-like scale 
	buffer_c->el_params.sa.scale.mem.i16 = relu_scale;
	buffer_c->el_params.sa.scale_frac_bits.mem.i8 = relu_fb;
	buffer_c->el_params.sa.zero_point.mem.i16 = relu_zp;

	if (pad_channels_next_block && !reduce_next_block){
		// for channels padding in next block
		int c = buffer_b->shape[2] + pad_channels_next_block;
		buffer_c->mem_stride[0] = buffer_b->shape[1] * c;
		buffer_c->mem_stride[1] = c;
		buffer_c->mem_stride[2] = 1;
	}
	status = mli_krn_eltwise_add_sa8(buffer_a, buffer_b, buffer_c);
	assert(status == MLI_STATUS_OK);

#ifndef _MSC_VER
	if (detailed_profiling){
		t6 = _lr(T0_COUNT);
		printf("add %.2f\n", (float)(t6 - t4) / (float) 1000000);
	}
#endif	// !_MSC_VER

#ifndef _MSC_VER
	if (profiling){
		printf("blazeblock %.2f\n", (float)(_lr(T0_COUNT) - t0) / (float) 1000000);
	}
#endif	// !_MSC_VER
	
	return result_index;
}

struct Point {
	int x;
	int y;

	explicit Point(): x(0), y(0) {};
	Point(int x_, int y_): x(x_), y(y_) {};

	bool positive(){
		return (x > 0) && (y > 0);
	}

	bool non_negative(){
		return (x >= 0) && (y >= 0);
	}
};

struct Rect {
	Point r1;	// top left corner
	Point r2;	// bottom right corner

	explicit Rect(): r1(), r2() {};
	Rect(Point r1_, Point r2_): r1(r1_), r2(r2_) {};
};

struct Tiling {
	Point pos;
	Point input_size;
	Point num_tiles;
	Point overlap;
	Point tile_size_base;
	Point crop;

	Tiling(Point input_size_, Point num_tiles_, Point overlap_, Point crop_) : 
		pos(), input_size(input_size_), num_tiles(num_tiles_), overlap(overlap_), crop(crop_) {
			assert(input_size_.positive());
			assert(num_tiles_.positive());
			assert(overlap_.non_negative());
			tile_size_base.x = input_size.x / num_tiles.x;
			tile_size_base.y = input_size.y / num_tiles.y;
			assert(tile_size_base.x > overlap.x);
			assert(tile_size_base.y > overlap.y);
		}

	void update() {
		if (pos.x < (num_tiles.x - 1)) pos.x++;
		else {
			pos.x = 0;
			pos.y++;
			if (pos.y == num_tiles.y) pos.y = 0;
		}
	}

	Rect get_tile(){
		int x1 = pos.x * tile_size_base.x;
		if (pos.x && overlap.x) x1 -= overlap.x;
		int y1 = pos.y * tile_size_base.y;
		if (pos.y && overlap.y) y1 -= overlap.y;
		Point r1(x1, y1);

		int x2 = (1 + pos.x) * tile_size_base.x;
		if (pos.x == (num_tiles.x - 1)) x2 = input_size.x;
		else x2 += overlap.x;
		int y2 = (1 + pos.y) * tile_size_base.y;
		if (pos.y == (num_tiles.y - 1)) y2 = input_size.y;
		else y2 += overlap.y;
		Point r2(x2, y2);

		return Rect(r1, r2);
	}

	Rect get_crop() {

		int x1 = 0;
		if (pos.x) x1 += crop.x;

		int y1 = 0;
		if (pos.y) y1 += crop.y;

		int x2 = 0;
		if (pos.x < (num_tiles.x - 1)) x2 += crop.x;

		int y2 = 0;
		if (pos.y < (num_tiles.y - 1)) y2 += crop.y;

		return Rect({ x1, y1 }, {x2, y2});
	}

};

void blazenet(int8_t * input, int8_t * output, DeQuantizeInfo * dequantize_info){
	
	init_tensors();
	mli_tensor input_dmp;
	input_dmp.data.mem.void_p = input;
	input_dmp.shape[0] = IMAGE_SIDE;
	input_dmp.shape[1] = IMAGE_SIDE;
	input_dmp.shape[2] = 3;
	input_dmp.rank = 3;
	input_dmp.el_type = MLI_EL_SA_8;
	DebugWriteResultToFile("_BLOCK_01_IN.idx", &input_dmp);

	mli_status status = MLI_STATUS_OK;
	int result_index;
	mli_tensor* tensors_tiling[3];
	x_tensor.data.mem.void_p = (void*) tensors_buffer;
	y_tensor.data.mem.void_p = (void*) (tensors_buffer + MAX_IR_SIZE_TILING);
	z_tensor.data.mem.void_p = (void*) (tensors_buffer + 2 * MAX_IR_SIZE_TILING);
	w_tensor.data.mem.void_p = (void*) (tensors_buffer + BUFFER_SIZE - 2 * VIRTUAL_BUFFER_SIZE);
	w_tensor.shape[0] = TILE_OUTPUT_HEIGHT;
	w_tensor.shape[1] = TILE_OUTPUT_WIDTH;
	w_tensor.shape[2] = TILE_OUTPUT_CHANNELS;
	w_tensor.mem_stride[0] = (TILE_OUTPUT_CHANNELS + 4) * TILE_OUTPUT_WIDTH;
	w_tensor.mem_stride[1] = TILE_OUTPUT_CHANNELS + 4;  // pre-memstride for channels padding
	w_tensor.mem_stride[2] = 1;
	Tiling input_tiling({ IMAGE_SIDE, IMAGE_SIDE }, { NUM_TILES_X, NUM_TILES_Y }, { TILE_OVERLAP_X, TILE_OVERLAP_Y }, { 0, 0 });
	Tiling output_tiling({ TILE_OUTPUT_WIDTH, TILE_OUTPUT_HEIGHT }, { NUM_TILES_X, NUM_TILES_Y }, { 0, 0 }, { 2, 2 });
	for (int n_tile = 0; n_tile < (NUM_TILES_X * NUM_TILES_Y); n_tile++) {
		Rect input_tile = input_tiling.get_tile();
		input_tiling.update();

		// set input
		int input_tile_width = input_tile.r2.x - input_tile.r1.x;
		for (int i = input_tile.r1.y; i < input_tile.r2.y; i++) {
			for (int j = input_tile.r1.x; j < input_tile.r2.x; j++) {
				for (int k = 0; k < NUM_CHANNELS; k++) {
					int src_ind = i * input_tiling.input_size.x * NUM_CHANNELS + j * NUM_CHANNELS + k;
					int dst_ind = (i - input_tile.r1.y) * input_tile_width * NUM_CHANNELS
						+ (j - input_tile.r1.x) * NUM_CHANNELS + k;
					x_tensor.data.mem.pi8[dst_ind] = input[src_ind];
				}
			}
		}

		// 1st conv
		x_tensor.shape[0] = TILING_INPUT_HEIGHT;
		x_tensor.shape[1] = TILING_INPUT_WIDTH;
		assert(input_tile.r2.x - input_tile.r1.x == TILING_INPUT_WIDTH);
		assert(input_tile.r2.y - input_tile.r1.y == TILING_INPUT_HEIGHT);
		x_tensor.shape[2] = NUM_CHANNELS;
		x_tensor.mem_stride[0] = 0;
		x_tensor.mem_stride[1] = 0;
		x_tensor.mem_stride[2] = 0;
		x_tensor.el_params.sa.scale.mem.i16 = 16448;
		x_tensor.el_params.sa.scale_frac_bits.mem.i8 = 21;
		x_tensor.el_params.sa.zero_point.mem.i16 = -1;
		y_tensor.el_params.sa.scale.mem.i16 = 20736;
		y_tensor.el_params.sa.scale_frac_bits.mem.i8 = 20;
		y_tensor.el_params.sa.zero_point.mem.i16 = -128;
		y_tensor.mem_stride[0] = 0;
		y_tensor.mem_stride[1] = 0;
		y_tensor.mem_stride[2] = 0;

		status = mli_krn_conv2d_hwcn_sa8_sa8_sa32(
			&x_tensor, &conv_weights_tensor, &conv_bias_tensor, &first_conv_cfg, &y_tensor
		);
		assert(status == MLI_STATUS_OK);

		z_tensor.mem_stride[0] = 0;
		z_tensor.mem_stride[1] = 0;
		z_tensor.mem_stride[2] = 0;
		tensors_tiling[0] = &x_tensor;
		tensors_tiling[1] = &y_tensor;
		tensors_tiling[2] = &z_tensor;

		// blazeblock 1
		result_index = blazeblock(
			tensors_tiling, 1,
			&dconv_1_weights_tensor, &dconv_1_bias_tensor,
			&conv_1_weights_tensor, &conv_1_bias_tensor,
			19398, 19, 52,
			16557, 18, 25,
			31386, 20, -128,
			false, false, 0, 4,
			false, false
		);

		// blazeblock 2
		result_index = blazeblock(
			tensors_tiling, result_index,
			&dconv_2_weights_tensor, &dconv_2_bias_tensor,
			&conv_2_weights_tensor, &conv_2_bias_tensor,
			29305, 19, 6,
			28794, 18, 21,
			28336, 19, -128,
			false, true, 4, 4,
			false, false
		);

		// blazeblock 3
		result_index = blazeblock(
			tensors_tiling, result_index,
			&dconv_3_weights_tensor, &dconv_3_bias_tensor,
			&conv_3_weights_tensor, &conv_3_bias_tensor,
			24522, 18, -20,
			23885, 18, 0,
			23818, 19, -128,
			true, false, 4, 4,
			false, false
		);

		// copy tiling output to wTensor
		mli_tensor * tiling_output = tensors_tiling[result_index];
		Rect output_tile = output_tiling.get_tile();
		Rect crop = output_tiling.get_crop();
		output_tiling.update();

		mli_sub_tensor_cfg cfg_sub_tensor;
		cfg_sub_tensor.offset[0] = crop.r1.y;
		cfg_sub_tensor.offset[1] = crop.r1.x;
		cfg_sub_tensor.offset[2] = 0;
		cfg_sub_tensor.size[0] = output_tile.r2.y - output_tile.r1.y;
		cfg_sub_tensor.size[1] = output_tile.r2.x - output_tile.r1.x;
		cfg_sub_tensor.size[2] = TILE_OUTPUT_CHANNELS;
    	cfg_sub_tensor.sub_tensor_rank = 3;
		mli_tensor tiling_output_cropped;
		status = mli_hlp_create_subtensor(tiling_output, &cfg_sub_tensor, &tiling_output_cropped);
		assert(status == MLI_STATUS_OK);

		mli_mov_cfg_t cfg_concat;
		int dst_offsets[] = {output_tile.r1.y, output_tile.r1.x, 0};
		int dst_mem_strides[] = {(int)w_tensor.mem_stride[0], (int)w_tensor.mem_stride[1], (int)w_tensor.mem_stride[2]};
		status = mli_mov_cfg_for_concat(&cfg_concat, dst_offsets, dst_mem_strides);
		assert(status == MLI_STATUS_OK);

		status = mli_mov_tensor_sync(&tiling_output_cropped, &cfg_concat, &w_tensor);
		assert(status == MLI_STATUS_OK);
	}
	
	// re-assign buffers in memory
	w_tensor.el_params.sa.scale.mem.i16 = tensors_tiling[result_index]->el_params.sa.scale.mem.i16;
	w_tensor.el_params.sa.zero_point.mem.i16 = tensors_tiling[result_index]->el_params.sa.zero_point.mem.i16;
	w_tensor.el_params.sa.scale_frac_bits.mem.i8 = tensors_tiling[result_index]->el_params.sa.scale_frac_bits.mem.i8;
	x_tensor.data.mem.void_p = (void*) (tensors_buffer + BUFFER_SIZE - 4 * VIRTUAL_BUFFER_SIZE);
	x_tensor.mem_stride[0] = 0;
	x_tensor.mem_stride[1] = 0;
	x_tensor.mem_stride[2] = 0;
	y_tensor.data.mem.void_p = (void*) (tensors_buffer + BUFFER_SIZE - 6 * VIRTUAL_BUFFER_SIZE);
	y_tensor.mem_stride[0] = 0;
	y_tensor.mem_stride[1] = 0;
	y_tensor.mem_stride[2] = 0;
	mli_tensor* tensors[] = { &w_tensor, &x_tensor, &y_tensor}; 
	
    DebugWriteResultToFile("_BLOCK_03_OUT.idx", tensors[0]);
    
	// blazeblock 4
	result_index = blazeblock(
		tensors, 0,
		&dconv_4_weights_tensor, &dconv_4_bias_tensor,
		&conv_4_weights_tensor, &conv_4_bias_tensor,
		27915, 18, -18,
		17245, 18, 19,
		26600, 19, -128,
		false, false, 4, 6,
		false, false
	);
    DebugWriteResultToFile("_BLOCK_04_OUT.idx", tensors[result_index]);

	// blazeblock 5
	result_index = blazeblock(
		tensors, result_index,
		&dconv_5_weights_tensor, &dconv_5_bias_tensor,
		&conv_5_weights_tensor, &conv_5_bias_tensor,
		17767, 17, -45,
		22183, 18, 58,
		19513, 19, -128,
		false, true, 6, 6,
		false, false
	);
    

	// blazeblock 6
	result_index = blazeblock(
		tensors, result_index,
		&dconv_6_weights_tensor, &dconv_6_bias_tensor,
		&conv_6_weights_tensor, &conv_6_bias_tensor,
		27327, 18, -14,
		25348, 19, -26,
		26915, 19, -128,
		true, false, 6, 8,
		false, false
	);
    DebugWriteResultToFile("_BLOCK_06_OUT.idx", tensors[result_index]);

	// blazeblock 7
	result_index = blazeblock(
		tensors, result_index,
		&dconv_7_weights_tensor, &dconv_7_bias_tensor,
		&conv_7_weights_tensor, &conv_7_bias_tensor,
		23788, 18, 4,
		29227, 19, 7,
		17442, 18, -128,
		false, false, 8, 8,
		false, false
	);

	// blazeblock 8
	result_index = blazeblock(
		tensors, result_index,
		&dconv_8_weights_tensor, &dconv_8_bias_tensor,
		&conv_8_weights_tensor, &conv_8_bias_tensor,
		27917, 18, -25,
		21026, 18, 38,
		23911, 19, -128,
		false, false, 8, 8,
		false, false
	);
    DebugWriteResultToFile("_BLOCK_08_OUT.idx", tensors[result_index]);
    
	// blazeblock 9
	result_index = blazeblock(
		tensors, result_index,
		&dconv_9_weights_tensor, &dconv_9_bias_tensor,
		&conv_9_weights_tensor, &conv_9_bias_tensor,
		17854, 17, 3,
		21003, 18, 24,
		30237, 19, -128,
		false, false, 8, 8,
		false, false
	);

	// blazeblock 10
	result_index = blazeblock(
		tensors, result_index,
		&dconv_10_weights_tensor, &dconv_10_bias_tensor,
		&conv_10_weights_tensor, &conv_10_bias_tensor,
		20181, 17, -8,
		24993, 18, 14,
		16994, 18, -128,
		false, false, 8, 8,
		false, false
	);
    DebugWriteResultToFile("_BLOCK_10_OUT.idx", tensors[result_index]);
    
	// blazeblock 11
	result_index = blazeblock(
		tensors, result_index,
		&dconv_11_weights_tensor, &dconv_11_bias_tensor,
		&conv_11_weights_tensor, &conv_11_bias_tensor,
		21904, 17, 24,
		16630, 17, 35,
		17276, 18, -128,
		false, true, 8, 8,
		false, false
	);

	// save buffer to use later
	q_tensor.data.mem.void_p = (void*) (tensors_buffer + BUFFER_SIZE - VIRTUAL_BUFFER_SIZE);
	q_tensor.el_params.sa.scale.mem.i16 = tensors[result_index]->el_params.sa.scale.mem.i16;
	q_tensor.el_params.sa.scale_frac_bits.mem.i8 = tensors[result_index]->el_params.sa.scale_frac_bits.mem.i8;
	q_tensor.el_params.sa.zero_point.mem.i16 = tensors[result_index]->el_params.sa.zero_point.mem.i16;
	
	mli_mov_cfg_t copy_cfg;
	status = mli_mov_cfg_for_copy(&copy_cfg);
	assert(status == MLI_STATUS_OK);

	status = mli_mov_tensor_sync(tensors[result_index], &copy_cfg, &q_tensor);
	assert(status == MLI_STATUS_OK);

	// tiling for weights
	for (int i = 0; i < (BB_CONV_KX * BB_CONV_KY * BB_12_IC * BB_12_OC); i++) {
		tensors_buffer[i] = conv_12_weights[i];
	}

	// blazeblock 12
	result_index = blazeblock(
		tensors, result_index,
		&dconv_12_weights_tensor, &dconv_12_bias_tensor,
		&conv_12_weights_tensor, &conv_12_bias_tensor,
		18383, 16, 0,
		27597, 18, 49,
		18906, 18, -128,
		true, false, 8, 0,
		false, false
	);
    DebugWriteResultToFile("_BLOCK_12_OUT.idx", tensors[result_index]);

	// tiling for weights
	for (int i = 0; i < (BB_CONV_KX * BB_CONV_KY * BB_13_IC * BB_13_OC); i++) {
		tensors_buffer[i] = conv_13_weights[i];
	}

	// blazeblock 13
	result_index = blazeblock(
		tensors, result_index,
		&dconv_13_weights_tensor, &dconv_13_bias_tensor,
		&conv_13_weights_tensor, &conv_13_bias_tensor,
		24941, 17, 12,
		28884, 18, 47,
		22135, 18, -128,
		false, false, 0, 0,
		false, false
	);

	// tiling for weights
	for (int i = 0; i < (BB_CONV_KX * BB_CONV_KY * BB_14_IC * BB_14_OC); i++) {
		tensors_buffer[i] = conv_14_weights[i];
	}

	// blazeblock 14
	result_index = blazeblock(
		tensors, result_index,
		&dconv_14_weights_tensor, &dconv_14_bias_tensor,
		&conv_14_weights_tensor, &conv_14_bias_tensor,
		28759, 17, -9,
		29735, 18, 29,
		24311, 18, -128,
		false, false, 0, 0,
		false, false
	);
    DebugWriteResultToFile("_BLOCK_14_OUT.idx", tensors[result_index]);
    
	// tiling for weights
	for (int i = 0; i < (BB_CONV_KX * BB_CONV_KY * BB_15_IC * BB_15_OC); i++) {
		tensors_buffer[i] = conv_15_weights[i];
	}

	// blazeblock 15
	result_index = blazeblock(
		tensors, result_index,
		&dconv_15_weights_tensor, &dconv_15_bias_tensor,
		&conv_15_weights_tensor, &conv_15_bias_tensor,
		17052, 16, -13,
		19074, 17, 51,
		24053, 18, -128,
		false, false, 0, 0,
		false, false
	);

	// tiling for weights
	for (int i = 0; i < (BB_CONV_KX * BB_CONV_KY * BB_16_IC * BB_16_OC); i++) {
		tensors_buffer[i] = conv_16_weights[i];
	}

	// blazeblock 16
	result_index = blazeblock(
		tensors, result_index,
		&dconv_16_weights_tensor, &dconv_16_bias_tensor,
		&conv_16_weights_tensor, &conv_16_bias_tensor,
		17413, 16, -17,
		24782, 17, 51,
		24148, 18, -128,
		false, false, 0, 0,
		false, false
	);
    DebugWriteResultToFile("_BLOCK_16_OUT.idx", tensors[result_index]);

	mli_tensor * tensor_a = tensors[result_index];
	mli_tensor * tensor_b = nullptr;
	mli_tensor * tensor_c = nullptr;
	int b_ind = -1;
	int c_ind = -1;
	for (int i = 0; i < 3; i++) {
		if (i == result_index) continue;

		if (tensor_b == nullptr) {
			tensor_b = tensors[i];
			b_ind = i;
			continue;
		}

		if (tensor_c == nullptr) {
			tensor_c = tensors[i];
			c_ind = i;
		}

	}
	v_tensor.data.mem.void_p = (void*) (tensors_buffer + BUFFER_SIZE - 3 * VIRTUAL_BUFFER_SIZE);
	mli_tensor * tensor_d = &v_tensor;
	mli_tensor * tensor_e = &q_tensor;

	// 2nd output conv
	tensor_b->el_params.sa.scale.mem.i16 = 24107;
	tensor_b->el_params.sa.scale_frac_bits.mem.i8 = 17;
	tensor_b->el_params.sa.zero_point.mem.i16 = 77;
	status = mli_krn_conv2d_hwcn_sa8_sa8_sa32(
		tensor_a, &conv_17_weights_tensor, &conv_17_bias_tensor, &conv_cfg, tensor_b
	);
    DebugWriteResultToFile("_FIN_CONV_02_OUT.idx", tensor_b);
	assert(status == MLI_STATUS_OK);

	// 1st output conv
	tensor_c->el_params.sa.scale.mem.i16 = 29098;
	tensor_c->el_params.sa.scale_frac_bits.mem.i8 = 18;
	tensor_c->el_params.sa.zero_point.mem.i16 = 47;
	status = mli_krn_conv2d_hwcn_sa8_sa8_sa32(
		tensor_e, &conv_18_weights_tensor, &conv_18_bias_tensor, &conv_cfg, tensor_c
	);
    DebugWriteResultToFile("_FIN_CONV_01_OUT.idx", tensor_c);
	assert(status == MLI_STATUS_OK);

	// tiling for weights
	for (int i = 0; i < (BB_CONV_KX * BB_CONV_KY * OCONV_3_IC * OCONV_3_OC); i++){
		tensors_buffer[i] = conv_19_weights[i];  
	}

	// 4th output conv
	tensor_d->el_params.sa.scale.mem.i16 = 24796;
	tensor_d->el_params.sa.scale_frac_bits.mem.i8 = 15;
	tensor_d->el_params.sa.zero_point.mem.i16 = -46;
	status = mli_krn_conv2d_hwcn_sa8_sa8_sa32(
		tensor_a, &conv_19_weights_tensor, &conv_19_bias_tensor, &conv_cfg, tensor_d
	);
    DebugWriteResultToFile("_FIN_CONV_04_OUT.idx", tensor_d);
	assert(status == MLI_STATUS_OK);

	// tiling for weights
	for (int i = 0; i < (BB_CONV_KX * BB_CONV_KY * OCONV_4_IC * OCONV_4_OC); i++) {
		tensors_buffer[i] = conv_20_weights[i];
	}

	// 3d output conv
	tensor_a->el_params.sa.scale.mem.i16 = 21595;
	tensor_a->el_params.sa.scale_frac_bits.mem.i8 = 16;
	tensor_a->el_params.sa.zero_point.mem.i16 = -46;
	status = mli_krn_conv2d_hwcn_sa8_sa8_sa32(
		tensor_e, &conv_20_weights_tensor, &conv_20_bias_tensor, &conv_cfg, tensor_a
	);
    DebugWriteResultToFile("_FIN_CONV_03_OUT.idx", tensor_a);
	assert(status == MLI_STATUS_OK);

	// write output data
	dequantize_info[0] = DeQuantizeInfo(tensor_c);
	dequantize_info[1] = DeQuantizeInfo(tensor_b);
	dequantize_info[2] = DeQuantizeInfo(tensor_a);
	dequantize_info[3] = DeQuantizeInfo(tensor_d);
	memcpy(output, tensor_c->data.mem.pi8, dequantize_info[0].size);
	int offset = dequantize_info[0].size;
	memcpy(output + offset, tensor_b->data.mem.pi8, dequantize_info[1].size);
	offset += dequantize_info[1].size;
	memcpy(output + offset, tensor_a->data.mem.pi8, dequantize_info[2].size);
	offset += dequantize_info[2].size;
	memcpy(output + offset, tensor_d->data.mem.pi8, dequantize_info[3].size);
}

void dequantize(int8_t * quantized, const DeQuantizeInfo * dequantize_info,  float * dequantized){
	int offset = 0;
	for (int j = 0; j < NUM_OUTPUT_PARTS; j++){
		const DeQuantizeInfo& info = dequantize_info[j];
		for (int i = 0; i < info.size; i++){
			int pos = i + offset;
			dequantized[pos] = sa2float(quantized[pos], info.zp, info.scale, info.fraq_bits);
		}
		offset += info.size;
	}
}
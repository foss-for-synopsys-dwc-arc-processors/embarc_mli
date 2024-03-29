/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _CIFAR10_CONSTANTS_H_
#define _CIFAR10_CONSTANTS_H_

#include "mli_config.h"

#include "cifar10_model.h"

// Defining weight data type
//===================================
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
#define W_EL_TYPE (MLI_EL_SA_8)
#define B_EL_TYPE (MLI_EL_SA_32)
typedef int8_t w_type;
typedef int32_t b_type;
#define W_FIELD pi8
#define B_FIELD pi32
#elif (MODEL_BIT_DEPTH == MODEL_FX_8W16D)
#define W_EL_TYPE (MLI_EL_FX_8)
#define B_EL_TYPE (MLI_EL_FX_8)
typedef int8_t w_type;
typedef int8_t b_type;
#define W_FIELD pi8
#define B_FIELD pi8
#else // (MODEL_BIT_DEPTH == MODEL_FX_16)
#define W_EL_TYPE (MLI_EL_FX_16)
#define B_EL_TYPE (MLI_EL_FX_16)
typedef int16_t w_type;
typedef int16_t b_type;
#define W_FIELD pi16
#define B_FIELD pi16
#endif


// Defining data sections attributes
//===================================
#if (PLATFORM == V2DSP_XY)
#if defined (__GNUC__) && !defined (__CCAC__)
// ARC GNU tools
// Model Weights attribute
#define _Wdata_attr __attribute__((section(".mli_model")))
#define _W  _Wdata_attr

// Model Weights (part 2) attribute 
#define _W2data_attr __attribute__((section(".mli_model_p2")))
#define _W2  _W2data_attr

// Bank X (XCCM) attribute
#define __Xdata_attr __attribute__((section(".Xdata")))
#define _X  __Xdata_attr

// Bank Y (YCCM) attribute
#define __Ydata_attr __attribute__((section(".Ydata")))
#define _Y  __Ydata_attr

// Bank Z (DCCM) attribute
#define __Zdata_attr __attribute__((section(".Zdata")))
#define _Z  __Zdata_attr

#else	
// Metaware tools
// Model Weights attribute
#define _Wdata_attr __attribute__((section(".mli_model")))
#define _W __xy _Wdata_attr

// Model Weights (part 2) attribute 
#define _W2data_attr __attribute__((section(".mli_model_p2")))
#define _W2 __xy _W2data_attr

// Bank X (XCCM) attribute
#define __Xdata_attr __attribute__((section(".Xdata")))
#define _X __xy __Xdata_attr

// Bank Y (YCCM) attribute
#define __Ydata_attr __attribute__((section(".Ydata")))
#define _Y __xy __Ydata_attr

// Bank Z (DCCM) attribute
#define __Zdata_attr __attribute__((section(".Zdata")))
#define _Z __xy __Zdata_attr
#endif // if defined (__GNUC__) && !defined (__CCAC__)


#elif (PLATFORM == V2DSP_VECTOR)

#define _Wdata_attr __attribute__((section(".vecmem_data")))
#define _W __vccm _Wdata_attr

// Model Weights (part 2) attribute
#define _W2data_attr __attribute__((section(".vecmem_data")))
#define _W2 __vccm _W2data_attr

// Bank X (XCCM) attribute
#define __Xdata_attr __attribute__((section(".vecmem_data")))
#define _X __vccm __Xdata_attr

// Bank Y (YCCM) attribute
#define __Ydata_attr __attribute__((section(".vecmem_data")))
#define _Y __vccm __Ydata_attr

// Bank Z (DCCM) attribute
#define __Zdata_attr __attribute__((section(".vecmem_data")))
#define _Z __vccm __Zdata_attr

#else // PLATFORM != V2DSP_XY && PLATFORM != V2DSP_VECTOR
#define _X __attribute__((section(".mli_ir_buf")))
#define _Y __attribute__((section(".mli_ir_buf")))
#define _Z __attribute__((section(".mli_ir_buf")))
#define _W __attribute__((section(".mli_model")))
#define _W2 __attribute__((section(".mli_model")))
#endif

//======================================================
//
// Common data transform (Qmn) defines
//
//======================================================

#define QMN(type, fraq, val)   (type)(val * (1u << (fraq)) + ((val >= 0)? 0.5f: -0.5f))
#define FRQ_BITS(int_part, el_type) ((sizeof(el_type)*8)-int_part-1)

//======================================================
//
// Common data transform (Qmn) defines
//
//======================================================

extern const w_type  _W  L1_conv_wt_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern const int8_t  conv1_w_fraq_arr[];
extern const int16_t conv1_w_scale_arr[];
extern const int16_t conv1_w_zp_arr[];
#endif

extern const b_type  _W  L1_conv_bias_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern const int8_t  conv1_b_fraq_arr[];
extern const int16_t conv1_b_scale_arr[];
extern const int16_t conv1_b_zp_arr[];
#endif


extern const w_type  _W  L2_conv_wt_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern const int8_t  conv2_w_fraq_arr[];
extern const int16_t conv2_w_scale_arr[];
extern const int16_t conv2_w_zp_arr[];
#endif

extern const b_type  _W  L2_conv_bias_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern const int8_t  conv2_b_fraq_arr[];
extern const int16_t conv2_b_scale_arr[];
extern const int16_t conv2_b_zp_arr[];
#endif


extern const w_type  _W2  L3_conv_wt_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern const int8_t  conv3_w_fraq_arr[];
extern const int16_t conv3_w_scale_arr[];
extern const int16_t conv3_w_zp_arr[];
#endif

extern const b_type  _W2  L3_conv_bias_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern const int8_t  conv3_b_fraq_arr[];
extern const int16_t conv3_b_scale_arr[];
extern const int16_t conv3_b_zp_arr[];
#endif


extern const w_type  _W2  L4_fc_wt_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern const int8_t  conv4_w_fraq_arr[];
extern const int16_t conv4_w_scale_arr[];
extern const int16_t conv4_w_zp_arr[];
#endif

extern const b_type  _W2  L4_fc_bias_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern const int8_t  conv4_b_fraq_arr[];
extern const int16_t conv4_b_scale_arr[];
extern const int16_t conv4_b_zp_arr[];
#endif

//======================================================
//
// Tensor's Integer bits per layer definitions
//
//======================================================
#if (MODEL_BIT_DEPTH == MODEL_FX_16) || (MODEL_BIT_DEPTH == MODEL_FX_8W16D)

#define CONV1_W_INT   (-1)
#define CONV1_B_INT   (0)
#define CONV1_OUT_INT (3)

#define CONV2_W_INT   (-1)
#define CONV2_B_INT   (-1)
#define CONV2_OUT_INT (5)

#define CONV3_W_INT   (-1)
#define CONV3_B_INT   (-2)
#define CONV3_OUT_INT (5)

#define FC4_W_INT   (-1)
#define FC4_B_INT   (-2)
#define FC4_OUT_INT (5)

#endif

//======================================================
//
// Shape and Fractional bits per layer definitions
//
//======================================================

// CONV1
//================================================
#define CONV1_W_SHAPE {5, 5, 3, 32}
#define CONV1_W_MEM_STRIDE {5 * 3 * 32, 3 * 32, 32, 1}
#define CONV1_W_ELEMENTS (5*5*3*32)
#define CONV1_W_RANK (4)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)

#define CONV1_W_FRAQ    conv1_w_fraq_arr
#define CONV1_W_SCALE   conv1_w_scale_arr
#define CONV1_W_ZP      conv1_w_zp_arr
#define CONV1_W_DIM     3
#define CONV1_W_SA_ELEMENTS (32)

#else

#define CONV1_W_FRAQ   (FRQ_BITS(CONV1_W_INT, w_type))
#define L1_WQ(val)   QMN(w_type, CONV1_W_FRAQ, val)

#endif

#define CONV1_B_ELEMENTS (32)
#define CONV1_B_SHAPE {32}
#define CONV1_B_MEM_STRIDE {1}
#define CONV1_B_RANK (1)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)

#define CONV1_B_FRAQ    conv1_b_fraq_arr
#define CONV1_B_SCALE   conv1_b_scale_arr
#define CONV1_B_ZP      conv1_b_zp_arr
#define CONV1_B_DIM     0
#define CONV1_B_SA_ELEMENTS (32)

#else

#define CONV1_B_FRAQ   (FRQ_BITS(CONV1_B_INT, w_type))
#define L1_BQ(val)   QMN(w_type, CONV1_B_FRAQ, val)
#define CONV1_OUT_FRAQ (FRQ_BITS(CONV1_OUT_INT, d_type))

#endif

#define CONV1_OUT_H (32)
#define CONV1_OUT_W (32)
#define CONV1_OUT_C (32)

// CONV2
//================================================
#define CONV2_W_SHAPE {5, 5, 32, 16}
#define CONV2_W_MEM_STRIDE {5 * 32 * 16, 32 * 16, 16, 1}
#define CONV2_W_ELEMENTS (5*5*32*16)
#define CONV2_W_RANK (4)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)

#define CONV2_W_FRAQ    conv2_w_fraq_arr
#define CONV2_W_SCALE   conv2_w_scale_arr
#define CONV2_W_ZP      conv2_w_zp_arr
#define CONV2_W_DIM     3
#define CONV2_W_SA_ELEMENTS (16)

#else
#define CONV2_W_FRAQ   (FRQ_BITS(CONV2_W_INT, w_type))
#define L2_WQ(val)   QMN(w_type, CONV2_W_FRAQ, val)
#endif

#define CONV2_B_SHAPE {16}
#define CONV2_B_MEM_STRIDE {1}
#define CONV2_B_ELEMENTS (16)
#define CONV2_B_RANK (1)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)

#define CONV2_B_FRAQ    conv2_b_fraq_arr
#define CONV2_B_SCALE   conv2_b_scale_arr
#define CONV2_B_ZP      conv2_b_zp_arr
#define CONV2_B_DIM     0
#define CONV2_B_SA_ELEMENTS (16)

#else

#define CONV2_B_FRAQ   (FRQ_BITS(CONV2_B_INT, w_type))
#define L2_BQ(val)   QMN(w_type, CONV2_B_FRAQ, val)
#define CONV2_OUT_FRAQ (FRQ_BITS(CONV2_OUT_INT, d_type))

#endif

#define CONV2_OUT_H (16)
#define CONV2_OUT_W (16)
#define CONV2_OUT_C (16)

// CONV3
//================================================
#define CONV3_W_SHAPE {5, 5, 16, 32}
#define CONV3_W_MEM_STRIDE {5 * 16 * 32, 16 * 32, 32, 1}
#define CONV3_W_ELEMENTS (5*5*16*32)
#define CONV3_W_RANK (4)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)

#define CONV3_W_FRAQ    conv3_w_fraq_arr
#define CONV3_W_SCALE   conv3_w_scale_arr
#define CONV3_W_ZP      conv3_w_zp_arr
#define CONV3_W_DIM     3
#define CONV3_W_SA_ELEMENTS (32)

#else

#define CONV3_W_FRAQ   (FRQ_BITS(CONV3_W_INT, w_type))
#define L3_WQ(val)   QMN(w_type, CONV3_W_FRAQ, val)

#endif

#define CONV3_B_SHAPE {32}
#define CONV3_B_MEM_STRIDE {1}
#define CONV3_B_ELEMENTS (32)
#define CONV3_B_RANK (1)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)

#define CONV3_B_FRAQ    conv3_b_fraq_arr
#define CONV3_B_SCALE   conv3_b_scale_arr
#define CONV3_B_ZP      conv3_b_zp_arr
#define CONV3_B_DIM     0
#define CONV3_B_SA_ELEMENTS (32)

#else

#define CONV3_B_FRAQ   (FRQ_BITS(CONV3_B_INT, w_type))
#define L3_BQ(val)   QMN(w_type, CONV3_B_FRAQ, val)
#define CONV3_OUT_FRAQ (FRQ_BITS(CONV3_OUT_INT, d_type))

#endif

#define CONV3_OUT_H (8)
#define CONV3_OUT_W (8)
#define CONV3_OUT_C (32)

// FC4
//================================================
#define FC4_W_SHAPE {(32*16), 10}
#define FC4_W_MEM_STRIDE {10, 1}
#define FC4_W_ELEMENTS (32*16*10)
#define FC4_W_RANK (2)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)

#define CONV4_W_FRAQ    conv4_w_fraq_arr
#define CONV4_W_SCALE   conv4_w_scale_arr
#define CONV4_W_ZP      conv4_w_zp_arr
#define CONV4_W_DIM     1
#define CONV4_W_SA_ELEMENTS (10)

#else

#define FC4_W_FRAQ   (FRQ_BITS(FC4_W_INT, w_type))
#define L4_WQ(val)   QMN(w_type, FC4_W_FRAQ, val)

#endif

#define FC4_B_SHAPE {10}
#define FC4_B_MEM_STRIDE {1}
#define FC4_B_ELEMENTS (10)
#define FC4_B_RANK (1)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)

#define CONV4_B_FRAQ    conv4_b_fraq_arr
#define CONV4_B_SCALE   conv4_b_scale_arr
#define CONV4_B_ZP      conv4_b_zp_arr
#define CONV4_B_DIM     0
#define CONV4_B_SA_ELEMENTS (10)

#else

#define FC4_B_FRAQ   (FRQ_BITS(FC4_B_INT, w_type))
#define L4_BQ(val)   QMN(w_type, FC4_B_FRAQ, val)
#define FC4_OUT_FRAQ (FRQ_BITS(FC4_OUT_INT, d_type))

#endif

#define FC4_OUT_SIZE (10)
#endif  //_CIFAR10_CONSTANTS_H_

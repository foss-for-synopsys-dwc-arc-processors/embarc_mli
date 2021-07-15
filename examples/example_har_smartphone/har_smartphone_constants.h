/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _HAR_SMARTPHONE_CONSTANTS_H_
#define _HAR_SMARTPHONE_CONSTANTS_H_

#include "mli_config.h"

#include "har_smartphone_model.h"
#include "tests_aux.h"

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
//ARC GNU tools
// Model Weights attribute
#define _Wdata_attr __attribute__((section(".mli_model")))
#define _W  _Wdata_attr

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
//Metaware tools
// Model Weights attribute
#define _Wdata_attr __attribute__((section(".mli_model")))
#define _W __xy _Wdata_attr

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

// Model Weights attribute
#define _Wdata_attr __attribute__((section(".vecmem_data")))
#define _W __vccm _Wdata_attr

// Operand X attribute (VCCM)
#define __Xdata_attr __attribute__((section(".vecmem_data")))
#define _X __vccm __Xdata_attr

// Operand Y attribute (VCCM)
#define __Ydata_attr __attribute__((section(".vecmem_data")))
#define _Y __vccm __Ydata_attr

// Operand Z attribute (VCCM)
#define __Zdata_attr __attribute__((section(".vecmem_data")))
#define _Z __vccm __Zdata_attr

#else // PLATFORM != V2DSP_XY && PLATFORM != V2DSP_VECTOR
#define _X __attribute__((section(".mli_ir_buf")))
#define _Y __attribute__((section(".mli_ir_buf")))
#define _Z __attribute__((section(".mli_ir_buf")))
#define _W __attribute__((section(".mli_model")))
#endif

//======================================================
//
// Common data transform (Qmn) defines (round-to-nearest)
//
//======================================================

#define EL_MAX(type) (type)((1u << (sizeof(type)*8-1))-1)
#define EL_MIN(type) (type)(-(1u << (sizeof(type)*8-1)))
#define SAT(type, val) (MIN(EL_MAX(type), MAX(EL_MIN(type), val)))

#define QMN(type, fraq, val)   (type)SAT(type, ((val) * (1u << (fraq)) + (((val) >= 0)? 0.5f: -0.5f)))
#define FRQ_BITS(int_part, el_type) ((sizeof(el_type)*8)-int_part-1)

//======================================================
//
// Quantized model parameters (statically allocated)
//
//======================================================
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern int16_t zero_zp_arr_shared[];
#endif

extern const w_type  _W  L1_fc_wt_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern int8_t  fc1_w_fraq_arr[];
extern int16_t fc1_w_scale_arr[];
#endif

extern const b_type  _W  L1_fc_bias_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern int8_t  fc1_b_fraq_arr[];
extern int16_t fc1_b_scale_arr[];
#endif

extern const w_type  _W  L2_lstm_wt_in_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern int8_t  lstm2_w_in_fraq_arr[];
extern int16_t lstm2_w_in_scale_arr[];
#endif

extern const w_type  _W  L2_lstm_wt_out_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern int8_t  lstm2_w_out_fraq_arr[];
extern int16_t lstm2_w_out_scale_arr[];
#endif

extern const b_type  _W  L2_lstm_bias_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern int8_t  lstm2_b_fraq_arr[];
extern int16_t lstm2_b_scale_arr[];
#endif

extern const w_type  _W  L3_lstm_wt_in_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern int8_t  lstm3_w_in_fraq_arr[];
extern int16_t lstm3_w_in_scale_arr[];
#endif

extern const w_type  _W  L3_lstm_wt_out_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern int8_t  lstm3_w_out_fraq_arr[];
extern int16_t lstm3_w_out_scale_arr[];
#endif

extern const b_type  _W  L3_lstm_bias_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern int8_t  lstm3_b_fraq_arr[];
extern int16_t lstm3_b_scale_arr[];
#endif

extern const w_type  _W  L4_fc_wt_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern int8_t  fc4_w_fraq_arr[];
extern int16_t fc4_w_scale_arr[];
#endif

extern const b_type  _W  L4_fc_bias_buf[];
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
extern int8_t  fc4_b_fraq_arr[];
extern int16_t fc4_b_scale_arr[];
#endif

//======================================================
//
// Tensor's Integer bits per layer definitions
//
//======================================================
#define FC1_W_INT   (1)
#define FC1_B_INT   (1)
#define FC1_OUT_INT (3)

#define LSTM2_W_INT   (0)
#define LSTM2_B_INT   (1)
#define LSTM2_CELL_INT (4)

#define LSTM3_W_INT   (0)
#define LSTM3_B_INT   (1)
#define LSTM3_CELL_INT (3)

#define FC4_W_INT   (0)
#define FC4_B_INT   (-2)
#define FC4_OUT_INT (3)

//======================================================
//
// Shape and Fractional bits per layer definitions
//
//======================================================

// FC1
//================================================
#define FC1_W_SHAPE {9, 32}
#define FC1_W_MEM_STRIDE {32, 1}
#define FC1_W_ELEMENTS (9*32)
#define FC1_W_RANK (2)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
#define FC1_W_FRAQ    fc1_w_fraq_arr
#define FC1_W_SCALE   fc1_w_scale_arr
#define FC1_W_ZP      zero_zp_arr_shared
#define FC1_W_DIM     1

#else
#define FC1_W_FRAQ   (FRQ_BITS(FC1_W_INT, w_type))
#define QW1(val)   QMN(w_type, FC1_W_FRAQ, val)
#endif

#define FC1_B_ELEMENTS (32)
#define FC1_B_SHAPE {32}
#define FC1_B_MEM_STRIDE {1}
#define FC1_B_RANK (1)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
#define FC1_B_FRAQ    fc1_b_fraq_arr
#define FC1_B_SCALE   fc1_b_scale_arr
#define FC1_B_ZP      zero_zp_arr_shared
#define FC1_B_DIM     0

#else
#define FC1_B_FRAQ   (FRQ_BITS(FC1_B_INT, b_type))
#define QB1(val)   QMN(b_type, FC1_B_FRAQ, val)

#define FC1_OUT_FRAQ (FRQ_BITS(FC1_OUT_INT, d_type))
#endif



// LSTM2
//================================================
#define LSTM2_W_IN_SHAPE {4, 32, 32}
#define LSTM2_W_IN_MEM_STRIDE {32*32, 32, 1}
#define LSTM2_W_IN_ELEMENTS (4*32*32)
#define LSTM2_W_IN_RANK (3)

#define LSTM2_W_OUT_SHAPE {4, 32, 32}
#define LSTM2_W_OUT_MEM_STRIDE {32*32, 32, 1}
#define LSTM2_W_OUT_ELEMENTS (4*32*32)
#define LSTM2_W_OUT_RANK (3)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
#define LSTM2_W_IN_FRAQ    lstm2_w_in_fraq_arr
#define LSTM2_W_IN_SCALE   lstm2_w_in_scale_arr
#define LSTM2_W_IN_ZP      zero_zp_arr_shared
#define LSTM2_W_IN_DIM     0

#define LSTM2_W_OUT_FRAQ    lstm2_w_out_fraq_arr
#define LSTM2_W_OUT_SCALE   lstm2_w_out_scale_arr
#define LSTM2_W_OUT_ZP      zero_zp_arr_shared
#define LSTM2_W_OUT_DIM     0
#else
#define LSTM2_W_FRAQ   (FRQ_BITS(LSTM2_W_INT, w_type))
#define QW2(val)   QMN(w_type, LSTM2_W_FRAQ, val)
#endif

#define LSTM2_B_ELEMENTS (4*32)
#define LSTM2_B_SHAPE {4, 32}
#define LSTM2_B_MEM_STRIDE {32, 1}
#define LSTM2_B_RANK (2)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
#define LSTM2_B_SA_FRAQ    lstm2_b_fraq_arr
#define LSTM2_B_SCALE   lstm2_b_scale_arr
#define LSTM2_B_ZP      zero_zp_arr_shared
#define LSTM2_B_DIM     0
#define LSTM2_SA_PARAMS 4
#else
#define LSTM2_B_FRAQ   (FRQ_BITS(LSTM2_B_INT, b_type))
#define QB2(val)   QMN(b_type, LSTM2_B_FRAQ, val)

#define LSTM2_OUT_FRAQ (FRQ_BITS(0, d_type))
#define LSTM2_CELL_FRAQ (FRQ_BITS(LSTM2_CELL_INT, d_type))
#endif

// LSTM3
//================================================
#define LSTM3_W_IN_SHAPE {4, 32, 32}
#define LSTM3_W_IN_MEM_STRIDE {32*32, 32, 1}
#define LSTM3_W_IN_ELEMENTS (4*32*32)
#define LSTM3_W_IN_RANK (3)

#define LSTM3_W_OUT_SHAPE {4, 32, 32}
#define LSTM3_W_OUT_MEM_STRIDE {32*32, 32, 1}
#define LSTM3_W_OUT_ELEMENTS (4*32*32)
#define LSTM3_W_OUT_RANK (3)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
#define LSTM3_W_IN_FRAQ    lstm3_w_in_fraq_arr
#define LSTM3_W_IN_SCALE   lstm3_w_in_scale_arr
#define LSTM3_W_IN_ZP      zero_zp_arr_shared
#define LSTM3_W_IN_DIM     0

#define LSTM3_W_OUT_FRAQ    lstm3_w_out_fraq_arr
#define LSTM3_W_OUT_SCALE   lstm3_w_out_scale_arr
#define LSTM3_W_OUT_ZP      zero_zp_arr_shared
#define LSTM3_W_OUT_DIM     0
#else
#define LSTM3_W_FRAQ   (FRQ_BITS(LSTM3_W_INT, w_type))
#define QW3(val)   QMN(w_type, LSTM3_W_FRAQ, val)
#endif

#define LSTM3_B_ELEMENTS (4*32)
#define LSTM3_B_SHAPE {4, 32}
#define LSTM3_B_MEM_STRIDE {32, 1}
#define LSTM3_B_RANK (2)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
#define LSTM3_B_FRAQ    lstm3_b_fraq_arr
#define LSTM3_B_SCALE   lstm3_b_scale_arr
#define LSTM3_B_ZP      zero_zp_arr_shared
#define LSTM3_B_DIM     0
#define LSTM3_SA_PARAMS 4
#else
#define LSTM3_B_FRAQ   (FRQ_BITS(LSTM3_B_INT, b_type))
#define QB3(val)   QMN(b_type, LSTM3_B_FRAQ, val)

#define LSTM3_OUT_FRAQ (FRQ_BITS(0, d_type))
#define LSTM3_CELL_FRAQ (FRQ_BITS(LSTM3_CELL_INT, d_type))
#endif

// FC4
//================================================
#define FC4_W_SHAPE {32, 6}
#define FC4_W_MEM_STRIDE {6, 1}
#define FC4_W_ELEMENTS (32*6)
#define FC4_W_RANK (2)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
#define FC4_W_FRAQ    fc4_w_fraq_arr
#define FC4_W_SCALE   fc4_w_scale_arr
#define FC4_W_ZP      zero_zp_arr_shared
#define FC4_W_DIM    1
#else
#define FC4_W_FRAQ   (FRQ_BITS(FC4_W_INT, w_type))
#define QW4(val)   QMN(w_type, FC4_W_FRAQ, val)
#endif

#define FC4_B_ELEMENTS (6)
#define FC4_B_SHAPE {6}
#define FC4_B_MEM_STRIDE {1}
#define FC4_B_RANK (1)

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
#define FC4_B_FRAQ    fc4_b_fraq_arr
#define FC4_B_SCALE   fc4_b_scale_arr
#define FC4_B_ZP      zero_zp_arr_shared
#define FC4_B_DIM    0
#else
#define FC4_B_FRAQ   (FRQ_BITS(FC4_B_INT, b_type))
#define QB4(val)   QMN(b_type, FC4_B_FRAQ, val)

#define FC4_OUT_FRAQ (FRQ_BITS(FC4_OUT_INT, d_type))
#endif

#endif // _HAR_SMARTPHONE_CONSTANTS_H_

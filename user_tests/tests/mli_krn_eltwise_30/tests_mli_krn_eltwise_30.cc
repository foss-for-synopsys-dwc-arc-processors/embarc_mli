/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <vector>
#include <variant>
#include <algorithm>

#include "mli_types.h"
#include "mli_ref_compiler_api.hpp"
#include "mli_ref_runtime_api.hpp"
#include "mli_kernels_factory_ref.hpp"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"

#include "vectors_mli_krn_eltwise.inc"

#define MAX_MIN_UPPER_LIMIT_SHIFT 23
#define MUL_MAX_SHIFT 31

using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

typedef mli_status(*eltwise_func_ptr)(
  const mli_tensor* /*in1*/,
  const mli_tensor* /*in2*/,
  mli_tensor* /*out*/);

enum class EltwiseTy {
    ADD,
    SUB,
    MUL,
    MAX,
    MIN,
};

struct eltwise_test_operands {
  const char* descr;
  const eltwise_func_ptr mli_krn_eltwise;
  tensor_quantizer in1;
  tensor_quantizer in2;
  tensor_quantizer out;
  const quality_metrics threshold;
  const crc32_calc check_sum;
  EltwiseTy ty;
};

// TODO Checksums of test tensors for various mli calculations mode.
// When developer finished implementation of kernel and consider it as ok, He need to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)

// Shared CRC Results
const crc32_calc  test_1_chksum_sa8{ 0x8BF4D950 }, test_2_chksum_sa8{ 0x2A1351FD },
                  test_3_chksum_sa8{ 0x46D90B34 }, test_4_chksum_sa8{ 0xF22D7321 },
                  test_5_chksum_sa8{ 0xC69DE0A9 }, test_6_chksum_fx16{ 0xfc026def },
                  test_6_chksum_sa8{ 0x3a54561 }, test_7_chksum_fx16{ 0x488ed527 },
                  test_7_chksum_sa8{ 0xD4B7515B }, test_8_chksum_fx16{ 0x68889D84 },
                  test_8_chksum_sa8{ 0x2D86F301 }, test_9_chksum_fx16{ 0x9417F3D7 },
                  test_9_chksum_sa8{ 0x351016DF }, test_10_chksum_fx16{ 0xD728E430 },
                  test_10_chksum_sa8{ 0xDC1A832D }, test_11_chksum_fx16{ 0xBF03F2E0 },
                  test_11_chksum_sa8{ 0xD36B7E94 };

// Platform Specific CRC Results
#if defined(CRC_RM_UP)
const crc32_calc test_1_chksum_fx16{ 0xAC3BE4B7 }, test_2_chksum_fx16{ 0x170065BD },
                 test_3_chksum_fx16{ 0x1E1FA5DD }, test_4_chksum_fx16{ 0xE27C401E },
                 test_5_chksum_fx16{ 0x1a678d57 };
#else
const crc32_calc test_1_chksum_fx16{ 0x5C7970C5 }, test_2_chksum_fx16{ 0x10D03580 },
                 test_3_chksum_fx16{ 0x6DD8F3E6 }, test_4_chksum_fx16{ 0xE27C401E },
                 test_5_chksum_fx16{ 0x1DB7DD6A };
#endif

#else  // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_sa8,
                  test_6_chksum_fx16, test_6_chksum_sa8,
                  test_7_chksum_fx16, test_7_chksum_sa8,
                  test_8_chksum_fx16, test_8_chksum_sa8,
                  test_9_chksum_fx16, test_9_chksum_sa8,
                  test_10_chksum_fx16, test_10_chksum_sa8,
                  test_11_chksum_fx16, test_11_chksum_sa8;

#endif

const quality_metrics thresholds_fx16_general {quality_metrics::kPassValueMaxAbsErr,
                                               quality_metrics::kPassValueSnr,
                                               /* SNR DB = */ 60.f,
                                               quality_metrics::kPassValueQuantErrPerc};

const quality_metrics thresholds_sa8_general {quality_metrics::kPassValueMaxAbsErr,
                                              quality_metrics::kPassValueSnr,
                                              /* SNR DB = */ 30.f,
                                              quality_metrics::kPassValueQuantErrPerc };

static const eltwise_test_operands tests_list[] = {
  // Eltwise add of two vectors
  {"Test 1 FX16 Add two vectors",  mli_krn_eltwise_add_fx16,
                                   input_1_fx16, input_2_fx16, test_1_out_fx16,
                                   thresholds_fx16_general, test_1_chksum_fx16, EltwiseTy::ADD},
  {"Test 1 SA8 Add two vectors",  mli_krn_eltwise_add_sa8,
                                  input_1_sa8, input_2_sa8, test_1_out_sa8,
                                  thresholds_sa8_general, test_1_chksum_sa8, EltwiseTy::ADD},
    
  // Eltwise add of vector and scalar
  {"Test 2 FX16 Add vec & scalar",  mli_krn_eltwise_add_fx16,
                                    input_2_fx16, input_3_fx16, test_2_out_fx16,
                                    thresholds_fx16_general, test_2_chksum_fx16, EltwiseTy::ADD},
  {"Test 2 SA8 Add vec & scalar",  mli_krn_eltwise_add_sa8,
                                   input_2_sa8, input_3_sa8, test_2_out_sa8,
                                   thresholds_sa8_general, test_2_chksum_sa8, EltwiseTy::ADD},

  // Eltwise sub of two vectors
  {"Test 3 FX16 Sub two vectors",  mli_krn_eltwise_sub_fx16,
                                   input_1_fx16, input_2_fx16, test_3_out_fx16,
                                   thresholds_fx16_general, test_3_chksum_fx16, EltwiseTy::SUB},
  {"Test 3 SA8 Sub two vectors",  mli_krn_eltwise_sub_sa8,
                                  input_1_sa8, input_2_sa8, test_3_out_sa8,
                                  thresholds_sa8_general, test_3_chksum_sa8, EltwiseTy::SUB},

  // Eltwise sub scalar from vec
  {"Test 4 FX16 Sub scalar - vec",  mli_krn_eltwise_sub_fx16,
                                    input_1_fx16, input_3_fx16, test_4_out_fx16,
                                    thresholds_fx16_general, test_4_chksum_fx16, EltwiseTy::SUB},
  {"Test 4 SA8 Sub scalar - vec",  mli_krn_eltwise_sub_sa8,
                                   input_1_sa8, input_3_sa8, test_4_out_sa8,
                                   thresholds_sa8_general, test_4_chksum_sa8, EltwiseTy::SUB},

  // Eltwise sub vec from scalar
  {"Test 5 FX16 Sub vec - scalar",  mli_krn_eltwise_sub_fx16,
                                    input_3_fx16, input_2_fx16, test_5_out_fx16,
                                    thresholds_fx16_general, test_5_chksum_fx16, EltwiseTy::SUB},
  {"Test 5 SA8 Sub vec - scalar",  mli_krn_eltwise_sub_sa8,
                                   input_3_sa8, input_2_sa8, test_5_out_sa8,
                                   thresholds_sa8_general, test_5_chksum_sa8, EltwiseTy::SUB},
  // Eltwise Mul of two vectors
  {"Test 6 FX16 Mul two vectors",  mli_krn_eltwise_mul_fx16,
                                   input_1_fx16, input_2_fx16, test_6_out_fx16,
                                   thresholds_fx16_general, test_6_chksum_fx16, EltwiseTy::MUL},
  {"Test 6 SA8 Mul two vectors",  mli_krn_eltwise_mul_sa8,
                                  input_1_sa8, input_2_sa8, test_6_out_sa8,
                                  thresholds_sa8_general, test_6_chksum_sa8, EltwiseTy::MUL},

  // Eltwise Mul vector & scalar
  {"Test 7 FX16 Mul vec & scalar",  mli_krn_eltwise_mul_fx16,
                                    input_1_fx16, input_3_fx16, test_7_out_fx16,
                                    thresholds_fx16_general, test_7_chksum_fx16, EltwiseTy::MUL},
  {"Test 7 SA8 Mul vec & scalar",  mli_krn_eltwise_mul_sa8,
                                   input_1_sa8, input_3_sa8, test_7_out_sa8,
                                   thresholds_sa8_general, test_7_chksum_sa8, EltwiseTy::MUL},

  // Eltwise Max two vectors
  {"Test 8 FX16 Max two vectors",  mli_krn_eltwise_max_fx16,
                                 input_1_fx16_12, input_2_fx16_12, test_8_out_fx16,
                                 thresholds_fx16_general, test_8_chksum_fx16, EltwiseTy::MAX},
  // {"Test 8 SA8 Max two vectors",  mli_krn_eltwise_max_sa8,
  //                               input_1_sa8_12, input_2_sa8_12, test_8_out_sa8,
  //                               thresholds_sa8_general, test_8_chksum_sa8, EltwiseTy::MAX},

  // Eltwise Max vector & scalar
  {"Test 9 FX16 Max vec & scalar",  mli_krn_eltwise_max_fx16,
                                  input_1_fx16_13, input_3_fx16_13, test_9_out_fx16,
                                  thresholds_fx16_general, test_9_chksum_fx16, EltwiseTy::MAX},
  // {"Test 9 SA8 Max vec & scalar",  mli_krn_eltwise_max_sa8,
  //                                input_1_sa8_13, input_3_sa8_13, test_9_out_sa8,
  //                                thresholds_sa8_general, test_9_chksum_sa8, EltwiseTy::MAX},

  // Eltwise Min two vectors
  
  // {"Test 10 SA8 Min two vectors",  mli_krn_eltwise_min_sa8,
  //                                input_1_sa8_12, input_2_sa8_12, test_10_out_sa8,
  //                                thresholds_sa8_general, test_10_chksum_sa8, EltwiseTy::MIN},
  
  {"Test 10 FX16 Min two vectors",  mli_krn_eltwise_min_fx16,
                                  input_1_fx16_12, input_2_fx16_12, test_10_out_fx16,
                                  thresholds_fx16_general, test_10_chksum_fx16, EltwiseTy::MIN},
  

  // Eltwise Min vector & scalar
  {"Test 11 FX16 Min vec & scalar",  mli_krn_eltwise_min_fx16,
                                   input_1_fx16_13, input_3_fx16_13, test_11_out_fx16,
                                   thresholds_fx16_general, test_11_chksum_fx16, EltwiseTy::MIN},
  // {"Test 11 SA8 Min vec & scalar",  mli_krn_eltwise_min_sa8,
  //                                 input_1_sa8_13, input_3_sa8_13, test_11_out_sa8,
  //                                 thresholds_sa8_general, test_11_chksum_sa8, EltwiseTy::MIN},
};

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

constexpr int kMemSize = 2048*sizeof(int32_t); // increase for int32 input data
static int8_t g_scratch_mem_in1[kMemSize] = { 0 };
static int8_t g_scratch_mem_in2[kMemSize] = { 0 };
static int8_t g_scratch_mem_out[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t g_mem_pool[kMemSize] = { 0 };

typedef struct _convert_param {
  int pre_op_shift1;
  int pre_op_shift2;
  int post_op_shift;
  int32_t scalar1;
  int32_t scalar2;
  int16_t in_offset1;
  int16_t in_offset2;
  int16_t out_offset;
  int16_t scale16_1;
  int16_t scale16_2;
  bool scalar_op1;
  bool scalar_op2;
} convert_param;

struct EltwiseOp {
  // All supported kernels
  using KernelTy = std::variant<lib_mli::Add_CS*,
                                lib_mli::Sub_CS*,
                                lib_mli::Mul_CS*,
                                lib_mli::Max_CS*,
                                lib_mli::Min_CS*>;

  // Element wise Kernel
  EltwiseOp(const eltwise_test_operands* cur_test) {
    mem_in1_keeper = memory_manager((int8_t*)(g_scratch_mem_in1), sizeof(g_scratch_mem_in1));
    mem_in2_keeper = memory_manager((int8_t*)(g_scratch_mem_in2), sizeof(g_scratch_mem_in2));
    mem_out_keeper = memory_manager((int8_t*)(g_scratch_mem_out), sizeof(g_scratch_mem_out));

    in1 = cur_test->in1.get_quantized_tensor(mem_in1_keeper.allocate_memory(cur_test->in1));
    in2 = cur_test->in2.get_quantized_tensor(mem_in2_keeper.allocate_memory(cur_test->in2));
    out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
    original_out = out;

    ty = cur_test->ty;
  }

  void ByteCopy(int8_t* src, uint32_t src_offset, int8_t* dst, uint32_t dst_offset, uint32_t length) {
    for (uint32_t i = 0; i < length; ++i) {
      dst[i + dst_offset] = src[i + src_offset];
    }
  };

  void CreateKernel(const lib_mli::Tensor<lib_mli::NoBuffer, 4>& in1_tensor,
                    const lib_mli::Tensor<lib_mli::NoBuffer, 4>& in2_tensor,
                    const lib_mli::Tensor<lib_mli::NoBuffer, 4>& out_tensor) {
    lib_mli::PlatformDescription pd;
    lib_ref::KernelsFactory kernel_factory(pd);
    switch (ty) {
    case EltwiseTy::ADD: {
      uint32_t eltwise_size = kernel_factory.Add_CS_GetSize();
      void* eltwise_buffer = malloc(eltwise_size);
      kernel = kernel_factory.Add_CS(eltwise_buffer, in1_tensor, in2_tensor, out_tensor);
      break;
    }
    case EltwiseTy::SUB: {
      uint32_t eltwise_size = kernel_factory.Sub_CS_GetSize();
      void* eltwise_buffer = malloc(eltwise_size);
      kernel = kernel_factory.Sub_CS(eltwise_buffer, in1_tensor, in2_tensor, out_tensor);
      break;
    }
    case EltwiseTy::MUL: {
      uint32_t eltwise_size = kernel_factory.Mul_CS_GetSize();
      void* eltwise_buffer = malloc(eltwise_size);
      kernel = kernel_factory.Mul_CS(eltwise_buffer, in1_tensor, in2_tensor, out_tensor);
      break;
    }
    case EltwiseTy::MAX: {
      uint32_t eltwise_size = kernel_factory.Max_CS_GetSize();
      void* eltwise_buffer = malloc(eltwise_size);
      kernel = kernel_factory.Max_CS(eltwise_buffer, in1_tensor, in2_tensor, out_tensor);
      break;
    }
    case EltwiseTy::MIN: {
      uint32_t eltwise_size = kernel_factory.Min_CS_GetSize();
      void* eltwise_buffer = malloc(eltwise_size);
      kernel = kernel_factory.Min_CS(eltwise_buffer, in1_tensor, in2_tensor, out_tensor);
      break;
    }
    default:
      assert(false);
      break;
    }
  }

  // memory memagement for ins & outs tensors
  memory_manager mem_in1_keeper;
  memory_manager mem_in2_keeper;
  memory_manager mem_out_keeper;

  // ins & outs tensors
  mli_tensor in1;
  mli_tensor in2;
  mli_tensor out;
  mli_tensor original_out;

  // kernel related
  EltwiseTy ty;
  KernelTy kernel;

  // restore the final results
  bool convert{false};
  // NOTE: In general, input and ouput should have the same dtype execpt mul kernel
  // input element size
  uint32_t in_elem_size;
  // output element size
  uint32_t out_elem_size;
  // inputs element size ratio
  uint32_t in_elem_size_ratio;
  // output element size ratio
  uint32_t out_elem_size_ratio;
  convert_param param;

  // the size and offset of output buffer
  uint32_t out_size;
  uint32_t out_mem_offset{0};

  // eltwise runtime instnace
  void* eltwise_instance{nullptr};
  uint32_t eltwise_instance_size{0};

  // eltwise private data
  lib_mli::PrivateData* eltwise_conf_private{nullptr};
  uint32_t eltwise_conf_private_size{0};
};

template <typename T>
T rshift(T in_val, int shift_right) {
  using unsigned_T = typename std::make_unsigned<T>::type;
  unsigned_T one = 1u;
  int nbits_max = sizeof(T) * 8 - 1;
  int nbits_min = 0;

  if (shift_right < nbits_min)
      return T(in_val) << (-shift_right);
  if (shift_right == nbits_min)
      return in_val;

  if (shift_right > nbits_max)
      return 0;

#ifdef CRC_RM_UP
  T round = (one << shift_right >> 1);
  T tmp = (in_val + round) >> shift_right;
#else
  T tmp = in_val >> shift_right;
  T last_deleted_mask = (one << shift_right >> 1);
  if (((in_val & last_deleted_mask) != (T)0) &&
          ((((T)tmp & (T)1) != (T)0) ||  ((in_val & (last_deleted_mask-(T)1))!= (T)0))) {
    return tmp + 1;
  }
#endif
  return tmp;
}

template <typename i_T, typename o_T>
o_T sat_fx(i_T in_val) {
  if (sizeof(i_T) <= sizeof(o_T)) {
    return (o_T)(in_val);
  }
  auto max_limit = std::numeric_limits<o_T>::max();
  auto min_limit = std::numeric_limits<o_T>::min();
  return std::min(std::max((i_T)min_limit, in_val), (i_T)max_limit);
}

template <typename T, typename o_T>
o_T norm_fx(T x)
{
    o_T inp_size = sizeof(T) * 8;
    T hi = x < (T)0 ? (T)-1 : (T)0;
    o_T r = 0;

    if (x == (T)0)
        return inp_size - 1;

    while ((x >> r) != hi)
        r++;
    return (inp_size - 1) - r;
}

template <typename i_T, typename o_T>
o_T norm_cast(i_T val , int32_t *norm_shift) {
    int32_t cast_shift = (sizeof(i_T) - sizeof(o_T)) * 8;
    int32_t norm = norm_fx<i_T, o_T>(val);
    *norm_shift = cast_shift - norm;
    return sat_fx<i_T, o_T>(rshift(val, *norm_shift));
}

template <typename io_T>
void convert_parameters(EltwiseTy ty,
                        const mli_tensor* in1,
                        const mli_tensor* in2,
                        mli_tensor* out,
                        bool convert,
                        convert_param* param) {
  int32_t scale_factor1 = 0, scale_factor2 = 0;
  int16_t scale16_1 = 1, scale_1 = 1, scale16_2 = 1, scale_2 = 1, scale_out = 1,
  shift1 = 0, shift2 = 0, shift_out = 0;
  int16_t in_offset1 = 0, in_offset2 = 0, out_offset = 0;
  int pre_op_shift1 = 0, pre_op_shift2 = 0, post_op_shift = 0;

  if (convert) {  //if in1.type == SA8
  //offset
    in_offset1 = in1->el_params.sa.zero_point.mem.i16;
    in_offset2 = in2->el_params.sa.zero_point.mem.i16;
    out_offset = out->el_params.sa.zero_point.mem.i16;
    //scale
    scale_1 = in1->el_params.sa.scale.mem.i16;
    scale_2 = in2->el_params.sa.scale.mem.i16;
    scale_out = out->el_params.sa.scale.mem.i16;
    //shift
    shift1 = in1->el_params.sa.scale_frac_bits.mem.i8;
    shift2 = in2->el_params.sa.scale_frac_bits.mem.i8;
    shift_out = out->el_params.sa.scale_frac_bits.mem.i8;
    
    if (ty == EltwiseTy::MAX || ty == EltwiseTy::MIN) {
      int shift;
      int32_t scale_factor = norm_cast<int32_t, int32_t>((int32_t)scale_1, &shift);
      scale_factor = scale_factor / scale_out;
      post_op_shift = shift1 - shift_out - shift;
      scale16_1 = norm_cast<int32_t, int16_t>(scale_factor, &shift);
      post_op_shift -= shift;
      shift = MAX(post_op_shift - MAX_MIN_UPPER_LIMIT_SHIFT, 0) + MIN(MUL_MAX_SHIFT + post_op_shift, 0);
      scale16_1 = scale16_1 >> shift;
      post_op_shift -= shift;
      scale16_2 = scale16_1;
    }else if (ty == EltwiseTy::MUL) {
      int shift;
      scale_factor1 = scale_1 * scale_2;
      scale_factor1 = norm_cast<int32_t, int32_t>(scale_factor1, &shift);
      scale_factor1 = (scale_factor1 / scale_out);
      post_op_shift = shift1 + shift2 - shift_out - shift;
      scale16_1 = norm_cast<int32_t, int16_t>(scale_factor1, &shift);
      post_op_shift -= shift;
      shift = MAX(post_op_shift - MUL_MAX_SHIFT, 0) + MIN(MUL_MAX_SHIFT + post_op_shift, 0);
      scale16_1 = scale16_1 >> shift;
      post_op_shift -= shift;
    } 
    else if(ty == EltwiseTy::ADD ||ty == EltwiseTy::SUB )
    {
      int norm_shift1, norm_shift2;
      scale_factor1 = norm_cast<int32_t, int32_t>((int32_t)scale_1, &norm_shift1);
      scale_factor2 = norm_cast<int32_t, int32_t>((int32_t)scale_2, &norm_shift2);
      scale_factor1 /= scale_out;
      scale_factor2 /= scale_out;
      pre_op_shift1 = -norm_shift1 + shift1 - shift_out;
      pre_op_shift2 = -norm_shift2 + shift2 - shift_out;
      scale16_1 = norm_cast<int32_t, int16_t>(scale_factor1, &norm_shift1);
      scale16_2 = norm_cast<int32_t, int16_t>(scale_factor2, &norm_shift2);
      pre_op_shift1 -= norm_shift1;
      pre_op_shift2 -= norm_shift2;
      shift1 = MAX(pre_op_shift1 - MAX_MIN_UPPER_LIMIT_SHIFT, 0) + MIN(MUL_MAX_SHIFT + pre_op_shift1, 0);
      shift2 = MAX(pre_op_shift2 - MAX_MIN_UPPER_LIMIT_SHIFT, 0) + MIN(MUL_MAX_SHIFT + pre_op_shift2, 0);
      scale16_1 = scale16_1 >> shift1;
      scale16_2 = scale16_2 >> shift2;
      pre_op_shift1 -= shift1;
      pre_op_shift2 -= shift2;
    }
  }
   else {
    constexpr int byte_size = 8;
    /*
     * max_shift will be determined according to the size of the out register to avoid
     * overflow in the rounding value.
     */
    int max_shift = sizeof(io_T) * byte_size;
    if (ty == EltwiseTy::MUL) {
      max_shift = 2 * max_shift - 1;
      if (in1->el_type == MLI_EL_SA_8) {
        post_op_shift = (in1->el_params.sa.scale_frac_bits.mem.i8 +
            in2->el_params.sa.scale_frac_bits.mem.i8) - out->el_params.sa.scale_frac_bits.mem.i8;
      } else {
        post_op_shift = (in1->el_params.fx.frac_bits +
            in2->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;
      }
    } else if (ty == EltwiseTy::MIN || ty == EltwiseTy::MAX) {
      max_shift = max_shift - 1;
      post_op_shift = in1->el_params.fx.frac_bits - out->el_params.fx.frac_bits;
    } else {
      max_shift = 2 * max_shift - 1;
      pre_op_shift1 = MIN(in1->el_params.fx.frac_bits -  in2->el_params.fx.frac_bits, 0);
      pre_op_shift2 = MIN(in2->el_params.fx.frac_bits -  in1->el_params.fx.frac_bits, 0);
      post_op_shift = MAX(in1->el_params.fx.frac_bits, in2->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;
      MLI_EXTRA_ASSERT(pre_op_shift1 > -max_shift);
      MLI_EXTRA_ASSERT(pre_op_shift2 > -max_shift);
    }
    post_op_shift = MIN(post_op_shift, max_shift);
  }

  bool scalar_op1 = 0;
  bool scalar_op2 = 0;
  uint32_t in1_sz = 0;
  uint32_t in2_sz = 0;
  in1_sz = in1->rank == 0;
  in2_sz = in2->rank == 0;
  scalar_op1 = (in1_sz == 1);
  scalar_op2 = (in2_sz == 1);
  param->scalar_op1 = scalar_op1;
  param->scalar_op2 = scalar_op2;

  param->scalar1 = 0;
  /* Extract ins as scalar values */
  if (scalar_op1) {
    if (in1->el_type == MLI_EL_FX_16) {
      param->scalar1 = in1->data.mem.i16;
    } else if (in1->el_type == MLI_EL_FX_8 || in1->el_type == MLI_EL_SA_8) {
      param->scalar1 = in1->data.mem.i8;
    } else {
      assert(false);
    }
  }
  param->scalar2 = 0;
  if (scalar_op2) {
    if (in2->el_type == MLI_EL_FX_16) {
      param->scalar2 = in2->data.mem.i16;
    } else if (in2->el_type == MLI_EL_FX_8 || in2->el_type == MLI_EL_SA_8) {
      param->scalar2 = in2->data.mem.i8;
    } else {
      assert(false);
    }
  }

  param->pre_op_shift1 = pre_op_shift1;
  param->pre_op_shift2 = pre_op_shift2;
  param->post_op_shift = post_op_shift;
  param->in_offset1 = in_offset1;
  param->in_offset2 = in_offset2;
  param->out_offset = out_offset;
  param->scale16_1 = scale16_1;
  param->scale16_2 = scale16_2;
}

void convert_and_copy_input(EltwiseOp& op,
                           uint32_t in1_size, uint32_t in1_mem_offset,
                           uint32_t in2_size, uint32_t in2_mem_offset) {
 assert(in1_size == op.in1.data.capacity * op.in_elem_size_ratio);
 assert(in2_size == op.in2.data.capacity * op.in_elem_size_ratio);

 if (op.in_elem_size_ratio == 1) {
    // no conversion and copy directly
    int8_t* in1_src = op.in1.data.mem.pi8;
    if (op.param.scalar_op1) {
      in1_src = reinterpret_cast<int8_t*>(&op.param.scalar1);
    }
    op.ByteCopy(in1_src, 0, (int8_t*)g_mem_pool, in1_mem_offset, in1_size);

    int8_t* in2_src = op.in2.data.mem.pi8;
    if (op.param.scalar_op2) {
      in2_src = reinterpret_cast<int8_t*>(&op.param.scalar2);
    }
    op.ByteCopy(in2_src, 0, (int8_t*)g_mem_pool, in2_mem_offset, in2_size);
  } else if (op.in_elem_size_ratio == 2) {
    // convert int16_t to int32_t (with shifting)
    auto converter = [&op] (int16_t* src, uint32_t length, uint32_t offset, uint32_t byte_size, int pre_op_shift) {
      std::vector<int32_t> in_temp;
      for (uint32_t i = 0; i < length; ++i) {
        int32_t temp_value = static_cast<int32_t>(src[i]);
        if (pre_op_shift < 0) {
          temp_value <<= -pre_op_shift;
        } else {
          temp_value >>= pre_op_shift;
        }
        in_temp.push_back(temp_value);
      }
      assert(in_temp.size() * sizeof(int32_t) == byte_size);
      op.ByteCopy(reinterpret_cast<int8_t*>(in_temp.data()), 0, (int8_t*)g_mem_pool, offset, byte_size);
    };

    if (!op.param.scalar_op1) {
      converter(op.in1.data.mem.pi16, op.in1.data.capacity / sizeof(int16_t), in1_mem_offset, in1_size, op.param.pre_op_shift1);
    } else {
      op.ByteCopy(reinterpret_cast<int8_t*>(&op.param.scalar1), 0, (int8_t*)g_mem_pool, in1_mem_offset, in1_size);
    }

    if (!op.param.scalar_op2) {
      converter(op.in2.data.mem.pi16, op.in2.data.capacity / sizeof(int16_t), in2_mem_offset, in2_size, op.param.pre_op_shift2);
    } else {
      op.ByteCopy(reinterpret_cast<int8_t*>(&op.param.scalar2), 0, (int8_t*)g_mem_pool, in2_mem_offset, in2_size);
    }

  } else if (op.in_elem_size_ratio == 4) {
    // convert int8_t to int32_t
    auto converter = [&op] (int8_t* src, uint32_t length, uint32_t offset, uint32_t byte_size,
                            int in_offset, int scale16, int pre_op_shift) {
      std::vector<int32_t> in_temp;
      for (uint32_t i = 0; i < length; ++i) {
        int32_t temp_value = static_cast<int32_t>(src[i]);
        temp_value = temp_value - in_offset;
        temp_value = temp_value * scale16;
        if (op.ty == EltwiseTy::ADD || op.ty == EltwiseTy::SUB) {
          temp_value = rshift(temp_value, pre_op_shift);
        }
        in_temp.push_back(temp_value);
      }
      assert(in_temp.size() * sizeof(int32_t) == byte_size);
      op.ByteCopy(reinterpret_cast<int8_t*>(in_temp.data()), 0, (int8_t*)g_mem_pool, offset, byte_size);
    };

    int8_t* in1_src = op.in1.data.mem.pi8;
    if (op.param.scalar_op1) {
      in1_src = reinterpret_cast<int8_t*>(&op.param.scalar1);
    }
    converter(in1_src, op.in1.data.capacity, in1_mem_offset, in1_size,
              op.param.in_offset1, op.param.scale16_1, op.param.pre_op_shift1);

    int8_t* in2_src = op.in2.data.mem.pi8;
    if (op.param.scalar_op2) {
      in2_src = reinterpret_cast<int8_t*>(&op.param.scalar2);
    }
    converter(in2_src, op.in2.data.capacity, in2_mem_offset, in2_size,
              op.param.in_offset2, op.param.scale16_2, op.param.pre_op_shift2);

  } else {
    // not support yet
    assert(false);
  }
}

template <typename T>
T get_input_val (const mli_tensor& in, const mli_tensor& out, uint32_t nth_of_output) {
  assert(in.el_type == MLI_EL_SA_8 || in.el_type == MLI_EL_FX_16);

  if (in.rank == 0) {
    if (sizeof(T) == sizeof(int8_t)) {
      return in.data.mem.i8;
    } else if (sizeof(T) == sizeof(int16_t)) {
      return in.data.mem.i16;
    }
  }

  assert(in.rank == out.rank);
  uint32_t idx = out.rank - 1;
  uint32_t in_offset = 1;
  uint32_t out_offset = 1;
  for (uint32_t elem_num = 1; idx >= 0; idx--) {
    elem_num *= out.shape[idx];
    if (nth_of_output < elem_num) {
      break;
    }
    in_offset *= in.shape[idx];
    out_offset *= out.shape[idx];
  }

  in_offset = nth_of_output / out_offset * in_offset + (in.shape[idx] != 1 ? nth_of_output % out_offset : 0);

  if (sizeof(T) == sizeof(int8_t)) {
    return *(in.data.mem.pi8 + in_offset);
  } else if (sizeof(T) == sizeof(int16_t)) {
    return *(in.data.mem.pi16 + in_offset);
  }
  return 0;
}

void convert_and_copy_output(EltwiseOp& op) {
  // Copy back the results for quality metrics
  assert(op.out_size == op.out.data.capacity * op.out_elem_size_ratio);
  uint32_t out_elem_size = mli_hlp_tensor_element_size(&op.out);
  uint32_t num_elem = op.out.data.capacity / out_elem_size;
  //assert(op.out_size / num_elem == sizeof(int32_t) );
  int32_t* out_ptr = reinterpret_cast<int32_t*>((int8_t*)g_mem_pool + op.out_mem_offset);

  // TODO: refactor to use Rescale kernel
  if (op.out_elem_size_ratio == 1) {
    // no conversion and copy directly
    op.ByteCopy((int8_t*)g_mem_pool, op.out_mem_offset, op.out.data.mem.pi8, 0, op.out_size);
  } else if (op.out_elem_size_ratio == 2) {
    // convert int32 to int16
    std::vector<int16_t> out_temp;
    for (uint32_t i = 0; i < num_elem; ++i) {
      int64_t val = static_cast<int64_t>(out_ptr[i]);
      int16_t tmp16 = rshift(val, op.param.post_op_shift);
      out_temp.push_back(tmp16);
    }
    op.ByteCopy(reinterpret_cast<int8_t*>(out_temp.data()), 0, op.out.data.mem.pi8, 0, op.out_size);
  } else if (op.out_elem_size_ratio == 4) {
    // convert int32 to int8
    std::vector<int8_t> out_temp;
    for (uint32_t i = 0; i < num_elem; ++i) {
      int64_t acc = out_ptr[i];
      if (op.ty == EltwiseTy::MUL) {
        // The output of acc is Xa * Xb and we will calc other parts
        //   out_val = Sa*Sb * (Xa-Oa) * (Xb-Ob)
        //   out_val = Sa * Sb * (Xa * Xb - Xa * Ob - Xb * Oa + Oa * Ob)
        //   out_val = Sa * Sb * (acc - Xa * Ob - Xb * Oa + Oa * Ob)

        // acc - Xa * Ob - Xb * Oa
        if (op.in_elem_size == sizeof(int8_t)) {
          acc = acc - op.param.in_offset2 * get_input_val<int8_t>(op.in1, op.out, i);
          acc = acc - op.param.in_offset1 * get_input_val<int8_t>(op.in2, op.out, i);
        } else if (op.in_elem_size == sizeof(int16_t)) {
          acc = acc - op.param.in_offset2 * get_input_val<int16_t>(op.in1, op.out, i);
          acc = acc - op.param.in_offset1 * get_input_val<int16_t>(op.in2, op.out, i);
        }
        // Oa * Ob
        acc = acc + op.param.in_offset1 * op.param.in_offset2;
        // Sa * Sb
        acc = acc * op.param.scale16_1 * op.param.scale16_2;
      }
      int16_t tmp16 = rshift(acc, op.param.post_op_shift);
      tmp16 = tmp16 + op.param.out_offset;
      tmp16 = sat_fx<int32_t, int8_t>(rshift(tmp16, 0));
      out_temp.push_back(tmp16);
    }
    op.ByteCopy(out_temp.data(), 0, op.out.data.mem.pi8, 0, op.out_size);
  } else {
    // not support yet
    assert(false);
  }
}

bool preprocess_phase(const reporter_full& reporter,
                      const eltwise_test_operands* cur_test,
                      const EltwiseOp& op) {
  bool is_test_passed = true;

  if (!(cur_test->in1.is_valid() && cur_test->in2.is_valid() && cur_test->out.is_valid())) {
      reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
      is_test_passed = false;
  }
  if (is_test_passed &&
      (tensor_quantizer::validate_tensor(op.in1) != tensor_quantizer::kOk ||
      tensor_quantizer::validate_tensor(op.in2) != tensor_quantizer::kOk ||
      tensor_quantizer::validate_tensor(op.out) != tensor_quantizer::kOk)) {
    reporter.report_message(cur_test->descr,
                            "FAILED at quantization step: more memory for one of tensors might be required");
    is_test_passed = false;
  }

  if (is_test_passed &&
      (op.mem_in1_keeper.is_memory_corrupted() || op.mem_in2_keeper.is_memory_corrupted() ||
      op.mem_out_keeper.is_memory_corrupted())) {
    reporter.report_message(cur_test->descr,
      "FAILED at quantization step: memory beside one of operands is corrupted");
    is_test_passed = false;
  }

  return is_test_passed;
}

template<typename EltwiseOpTy>
void plan_memory(EltwiseOp& op,
                 uint32_t input1_shape[4],
                 uint32_t input2_shape[4], uint32_t output_shape[4]) {
  // STEP 1.2.1: Memory management (Up to user on how to deal with it)
  //==================================================================
  EltwiseOpTy eltwise_op = std::get<EltwiseOpTy>(op.kernel);

  // We have single buffer for everything.
  uint32_t offsets[1] = {0};
  uint32_t in1_mem_offset;
  uint32_t in2_mem_offset;

  // Descriptors
  uint32_t* offset = &offsets[0];
  offset = &offsets[0];
  uint32_t descr_size = eltwise_op->GetRuntimeObjectSize();
  *offset += descr_size;

  // Leave space for private data buffer
  offset = &offsets[0];
  uint32_t private_buffer_size = eltwise_op->GetKernelPrivateDataSize();
  *offset += private_buffer_size;

  // Define buffers for in\out tensors
  uint32_t in1_size = eltwise_op->GetInputLeftBufferSize() * op.in_elem_size;
  lib_mli::OffsetBuffer add_in1_buf{*offset, 0, in1_size, op.in_elem_size};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> add_in1_tensor(add_in1_buf, input1_shape);
  in1_mem_offset = *offset;
  *offset += in1_size;

  uint32_t in2_size = eltwise_op->GetInputRightBufferSize() * op.in_elem_size;
  lib_mli::OffsetBuffer add_in2_buf{*offset, 0, in2_size, op.in_elem_size};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> add_in2_tensor(add_in2_buf, input2_shape);
  in2_mem_offset = *offset;
  *offset += in2_size;

  offset = &offsets[0];
  // NOTE: The output should be aligned, otherwise, it will cause `vvst` crash.
  //       For example, offset is 4 bytes aligned if output is int32_t.
  *offset = CEIL_RND(*offset, op.out_elem_size);
  uint32_t out_size = eltwise_op->GetOutputBufferSize() * op.out_elem_size;
  lib_mli::OffsetBuffer add_out_buf{*offset, 0, out_size, op.out_elem_size};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> add_out_tensor(add_out_buf, output_shape);
  op.out_mem_offset = *offset;
  op.out_size = out_size;
  *offset += out_size;

  // MLI tensor structures and eltwise configuration
  offset = &offsets[0];
  uint32_t ctrl_buffer_size = eltwise_op->GetCtrlBufferSize();
  lib_mli::OffsetBuffer eltwise_ctrl_buf{*offset, 0, ctrl_buffer_size, sizeof(char)};
  *offset += ctrl_buffer_size;

  assert(ctrl_buffer_size == 0);
  assert(*offset < kMemSize);

  mli_status status = MLI_STATUS_OK;

  // Attaching buffer (descriptors) to the operation
  status = eltwise_op->AttachBufferOffsets(add_in1_tensor, add_in2_tensor, add_out_tensor, eltwise_ctrl_buf);

  op.eltwise_instance = (int8_t*)g_mem_pool;
  op.eltwise_instance_size = eltwise_op->GetRuntimeObjectSize();

  status = eltwise_op->GetKernelPrivateData((int8_t*)g_mem_pool + op.eltwise_instance_size);
  assert(status == MLI_STATUS_OK);
  op.eltwise_conf_private =
      reinterpret_cast<lib_mli::PrivateData*>((int8_t*)g_mem_pool + op.eltwise_instance_size);
  op.eltwise_conf_private_size = eltwise_op->GetKernelPrivateDataSize();

  // STEP 1.2.2: Copy dataset from scratch buffer to the global shared memory pool
  //==================================================================
  // Copy input data from scratch buffer to the shared memory pool
  convert_and_copy_input(op, in1_size, in1_mem_offset, in2_size, in2_mem_offset);
}

void prepare_phase(const eltwise_test_operands* cur_test, EltwiseOp& op) {
  // STEP 1.1: Construct EltwiseOp as a specific ExecutionInterface successor
  //==================================================================
  uint32_t input1_shape[4] = {1, 1, 1, 1};
  int32_t input1_stride[4] = {1, 1, 1, 1};
  if (op.in1.rank > 0) {
    assert(op.in1.rank == 2);
    input1_shape[2] = op.in1.shape[0];
    input1_shape[3] =  op.in1.shape[1];
    input1_stride[2] = op.in1.mem_stride[0];
    input1_stride[3] = op.in1.mem_stride[1];
  }

  uint32_t input2_shape[4] = {1, 1, 1, 1};
  int32_t input2_stride[4] = {1, 1, 1, 1};
  if (op.in2.rank > 0) {
    assert(op.in2.rank == 2);
    input2_shape[2] = op.in2.shape[0];
    input2_shape[3] =  op.in2.shape[1];
    input2_stride[2] = op.in2.mem_stride[0];
    input2_stride[3] = op.in2.mem_stride[1];
  }

  assert(op.out.rank == 2);
  uint32_t output_shape[4] = {1, 1, op.out.shape[0], op.out.shape[1]};
  int32_t output_stride[4] = {1, 1, op.out.mem_stride[0], op.out.mem_stride[1]};

  const lib_mli::Tensor<lib_mli::NoBuffer, 4> in1_tensor(input1_shape, input1_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, 4> in2_tensor(input2_shape, input2_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, 4> out_tensor(output_shape, output_stride);

  // the size of accumulator in bytes
  uint32_t acc_size = sizeof(int32_t);


  // Currently, eltwise kernel only support 32bits. So, convert input to 32bits for only add and sub 
  uint32_t elem_size = mli_hlp_tensor_element_size(&op.in1);
  assert(elem_size <= acc_size);
  assert(elem_size == mli_hlp_tensor_element_size(&op.in2));
  assert(elem_size == mli_hlp_tensor_element_size(&op.out));
  if ((op.in1.el_type != MLI_EL_FX_16) &&
      (op.in1.el_type != MLI_EL_SA_8)) {
    // not tested yet
    assert(false);
  }

  if (op.in1.el_type == MLI_EL_SA_8) 
  { 
    op.convert = true;
  }
  convert_parameters<int32_t>(op.ty, &op.in1, &op.in2, &op.out, op.convert, &op.param);

  uint32_t elem_size_ratio = 1;
  if (elem_size != acc_size) {
    elem_size_ratio = acc_size / elem_size;
  }

  //i8 + i8 -> (i32) -> i8 or i16 + i16 -> (i32) -> i16
  if (op.ty == EltwiseTy::MUL ) {
    // For mul, we pass i8 or i16 inputs directly and get i32 output. Then change it back.
    op.in_elem_size = elem_size;
    op.in_elem_size_ratio = 1;
    
    op.out_elem_size = acc_size;
    op.out_elem_size_ratio = elem_size_ratio;
    // NOTE: The complete equation of mul is Sa*Sb * (Xa-Oa) * (Xb-Ob)
    //         Sa is the scale of first input, Xa is the value and Oa is offset.
    //         Sb is the scale of second input, Xb is the value and Ob is offset.
    // In the current implementation, we only do Xa*Xb in mul kernel and leave other
    // parts in testing (convert and copy functions). We have to extend mul to support
    // encode zp methods in order to compute other parts in mli internal.
   } 
   else if(op.ty == EltwiseTy::MIN || op.ty == EltwiseTy::MAX)
  {
    // For min and max, we pass i8 or i16 inputs directly and get output directly
    op.in_elem_size = elem_size;
    op.in_elem_size_ratio = 1;
    
    op.out_elem_size = elem_size;
    op.out_elem_size_ratio = 1;
  }
   else {
    // Except mul,max and min operations, we first change inputs to i32 and feed into eltwise kernel
    // to get the i32 output. Then we will change it back to the origial data type i8 or i16.
    op.in_elem_size_ratio = elem_size_ratio;
    op.in_elem_size = acc_size;
    op.out_elem_size = acc_size;
    op.out_elem_size_ratio = elem_size_ratio;
   }
   

  op.CreateKernel(in1_tensor, in2_tensor, out_tensor);

  // STEP 1.2: Memory management And Copy Inputs
  //==================================================================
  if (op.ty == EltwiseTy::ADD) {
    plan_memory<lib_mli::Add_CS*>(op, input1_shape, input2_shape, output_shape);
  } else if (op.ty == EltwiseTy::SUB) {
    plan_memory<lib_mli::Sub_CS*>(op, input1_shape, input2_shape, output_shape);
  } else if (op.ty == EltwiseTy::MUL) {
    plan_memory<lib_mli::Mul_CS*>(op, input1_shape, input2_shape, output_shape);
  } else if (op.ty == EltwiseTy::MAX) {
    plan_memory<lib_mli::Max_CS*>(op, input1_shape, input2_shape, output_shape);
  } else if (op.ty == EltwiseTy::MIN) {
    plan_memory<lib_mli::Min_CS*>(op, input1_shape, input2_shape, output_shape);
  } else {
    assert(false);
  }
}

void execution_phase(const EltwiseOp& op) {
  // STEP 3: Execution phase
  //==================================================================
  uint32_t tiles_num = 1;

  uint64_t membasis[] = { reinterpret_cast<uint64_t>(g_mem_pool) };

  auto eltwise = lib_mli::ExecutionInterface::Create(
    static_cast<void*>(op.eltwise_instance), op.eltwise_instance_size,
    op.eltwise_conf_private, op.eltwise_conf_private_size,
    membasis, sizeof(membasis) / sizeof(membasis[0]));

  assert(eltwise != nullptr);

  auto status = MLI_STATUS_OK;
  for (int i = 0; i < tiles_num; ++i) {
    status = eltwise->Prefetch();
    assert(status == MLI_STATUS_OK);

    status = eltwise->Issue();
    assert(status == MLI_STATUS_OK);

    status = eltwise->Update();
    assert(status == MLI_STATUS_OK);
  }
}

bool postprocess_phase(const reporter_full& reporter,
                       const eltwise_test_operands* cur_test,
                       EltwiseOp& op) {

  convert_and_copy_output(op);

  auto& out = op.out;
  auto& source_out_tensor = op.original_out;
  bool is_per_tensor_quant = true;
  bool is_test_passed = true;
  quality_metrics test_metics;

  if (is_test_passed &&
      (op.mem_in1_keeper.is_memory_corrupted() ||
       op.mem_in2_keeper.is_memory_corrupted()
       // the above copy step may impact the memory of out tensor if it is strided memory
       // op.mem_out_keeper.is_memory_corrupted()
       )) {
    reporter.report_message(cur_test->descr,
      "FAILED after kernel run: memory beside one of operands is corrupted");
    is_test_passed = false;
  }

  if (is_test_passed &&
      test_metics.calculate_metrics(out, cur_test->out) == false) {
    reporter.report_message(cur_test->descr, "FAILED at comparison output with reference");
    is_test_passed = false;
  }

  if (out.el_type == MLI_EL_FX_8 || out.el_type == MLI_EL_FX_16) {
    is_test_passed &= out.el_params.fx.frac_bits == source_out_tensor.el_params.fx.frac_bits;
  } else if (out.el_type == MLI_EL_SA_8 || out.el_type == MLI_EL_SA_32) {
    if (out.el_params.sa.dim < 0 || source_out_tensor.el_params.sa.dim < 0) {
      is_test_passed &=
        (out.el_params.sa.scale.mem.i16 == source_out_tensor.el_params.sa.scale.mem.i16) &&
        (out.el_params.sa.zero_point.mem.i16 == source_out_tensor.el_params.sa.zero_point.mem.i16) &&
        (out.el_params.sa.scale_frac_bits.mem.i8 ==
         source_out_tensor.el_params.sa.scale_frac_bits.mem.i8);
    } else {
      is_per_tensor_quant = false;
      is_test_passed = false;
  
    }
  }

  if (!is_test_passed) {
    reporter.report_message(cur_test->descr,
      is_per_tensor_quant ? "FAILED as element params of output tensor was modified"
                          : "FAILED as per-axis quantization of output tensor isn't supported");
  }

  if (is_test_passed) {
    crc32_calc data_crc;
    data_crc(op.in1);
    data_crc(op.in2);
    data_crc(op.out);

    is_test_passed = reporter.evaluate_and_report_case(cur_test->descr,
        test_metics, cur_test->threshold, data_crc, cur_test->check_sum);
  }

  return is_test_passed;
}

int main() {
  const reporter_full reporter;
  bool final_status = true;

  reporter.report_header("MLI3.0|Kernels|Basic Eltwise Functions Tests");

  for (int i = 0; i < kTestsNum; ++i) {
    bool is_test_passed = true;
    const eltwise_test_operands* cur_test = &tests_list[i];

    // STEP 0: Preprocessing phase
    //==================================================================
    EltwiseOp op = EltwiseOp(cur_test);
    is_test_passed = preprocess_phase(reporter, cur_test, op);

    // STEP 1: Preparing phase
    //==================================================================
    prepare_phase(cur_test, op);

    // STEP 2: Executing phase
    //==================================================================
    // Run conv2d MLI3.0 kernel
    execution_phase(op);

    // Run eltwise MLI2.0 kernel, for debug purpose
    // if (is_test_passed &&
    //     cur_test->mli_krn_eltwise(&op.in1, &op.in2, &op.out) != MLI_STATUS_OK) {
    //   reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
    //   is_test_passed = false;
    // }

    // STEP 3: Postprocessing phase
    //==================================================================
    is_test_passed &= postprocess_phase(reporter, cur_test, op);

    final_status &= is_test_passed;
  }

  reporter.report_outline("[AUTO] Group: mli_krn_eltwise_30", final_status);

  return (final_status) ? 0 : 1;
}

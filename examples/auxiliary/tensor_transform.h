/*
 *  Copyright (c) 2019, Synopsys, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1) Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2)  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3) Neither the name of the <ORGANIZATION> nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _TENSOR_TRANSFORM_H_
#define _TENSOR_TRANSFORM_H_
/**
 * @file Tensor Conversion function set
 * @brief Various conversions of MLI TEnsor beside MLI_Helpers functionality
 */

#ifdef __cplusplus
extern "C"
{
#endif

#include "mli_types.h"

/**
 * @brief Transform float array to MLI FX tensor
 *
 * @detail Perform float-to-fx transformation of data according to *dst* tensor element type parameters.
 *          Function won't touch any fields of *dst* structurefilling only memory it provides as *data*
 *
 * @param[in] src - Pointer to input array of float values
 * @param[in] src_size - Size of input array.
 * @param[out] dst - Pointer to output FX tensor structure. Must provide valid data pointer to array of sufficient size,
 *                  valid element type and number of fractional bytes.
 *
 * @return Operation status code (MLI_Types.h)
  */
mli_status mli_hlp_float_to_fx_tensor(
        const float *src,
        uint32_t  src_size,
        mli_tensor *dst);

/**
 * @brief Transform data of MLI FX tensor to float array
 *
 * @detail Perform fx-to-float transformation of data according to *src* tensor element type parameters.
 *          dst array must be sufficient to hold all transformed data (dst_size == total number of elements inside tensor)
 *
 * @param[in] src - Pointer to valid input FX tensor structure
 * @param[out] dst - Size of output float array.
 * @param[in] dst_size - number of elements in output array.
 *
 * @return Operation status code (MLI_Types.h)
  */
mli_status mli_hlp_fx_tensor_to_float(
        const mli_tensor *src,
        float *dst,
        uint32_t  dst_size);

#ifdef __cplusplus
}
#endif

#endif //_TENSOR_TRANSFORM_H_

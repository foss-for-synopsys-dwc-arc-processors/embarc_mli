/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_SERVICE_FUNCTIONS_HPP_
#define _MLI_SERVICE_FUNCTIONS_HPP_

#include <assert.h>
#include <stdint.h>

namespace snps_arc::metaware::mli::service {

inline const unsigned GetBufferSize(int rank, const uint32_t* shape,
                                    const int32_t* stride) {
  unsigned ret_val = 0;
  for (int dim = rank - 1; dim >= 0; --dim) {
    ret_val += stride[dim] * (shape[dim] - 1);
  }
  ret_val += 1;
  return ret_val;
}

}  // namespace snps_arc::metaware::mli::service

#endif /* _MLI_SERVICE_FUNCTIONS_HPP_ */
/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_CONV_PRIVATE_TYPES_H_
#define _MLI_KRN_CONV_PRIVATE_TYPES_H_

#include "mli_types.h"

namespace snps_arc::metaware::mli{

class Conv2dPrivateData : public PrivateData {
  public:
    Conv2dPrivateData() : PrivateData(kConv2dId){}

};

} // namespace mli

#endif // _MLI_KRN_CONV_PRIVATE_TYPES_H_
/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_MOV_PRIVATE_TYPES_HPP_
#define _MLI_MOV_PRIVATE_TYPES_HPP_

#include "mli_types.hpp"

namespace snps_arc::metaware::mli{

class MovePrivateData : public PrivateData {
  public:
    MovePrivateData() : PrivateData(kMoveId){}

};

} // namespace mli

#endif // _MLI_MOV_PRIVATE_TYPES_HPP_

/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_REF_SYNC_INTERFACE_HPP_
#define _MLI_REF_SYNC_INTERFACE_HPP_

#include "mli_sync_interface.hpp"

namespace lib_mli = ::snps_arc::metaware::mli;

namespace snps_arc::metaware::mli::ref {

class SynchronizationInterface : public lib_mli::SynchronizationInterface {
  public:
    mli_status WaitEvent(int32_t mask){
      return MLI_STATUS_OK;
    }

};

} //namespace ref


#endif
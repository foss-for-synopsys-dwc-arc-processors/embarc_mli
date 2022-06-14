/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_SYNC_INTERFACE_HPP_
#define _MLI_SYNC_INTERFACE_HPP_

namespace snps_arc::metaware::mli {

class SynchronizationInterface {

  public:
    virtual mli_status WaitEvent(int32_t mask) = 0;
    
};


}  //namespace mli
#endif
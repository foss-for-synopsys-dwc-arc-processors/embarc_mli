/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _KWS_FACTORY_H_
#define _KWS_FACTORY_H_

#include <stdint.h>

#include "kws_module.h"

//
// Base interface of KWS modules factory and factories for all implemented modules.
//
// Generative abstraction for building KWS module, that used here to completely hide
// particular implementation of class from user code. User sees only factory interface which
// not exposes anything specific to the module implementation it generates.
//
// Reason: We need to make KWS modules completely encapsulated to not expose implementation of particular
// module to user code, because it can contain things that user code must not know about.
// For example, a KWS class declaration may contains DSP library primitives or XY types specifier in
// function signatures, which isn't supported by ARC GNU Toolchain. To Deal with it we can compile KWS Module
// separately and link it up to the application. But for this purpose, user code must not see declaration of
// specific KWS - only it's abstraction.
//
// This file contains declaration of base kws_factory class, and factories of all implemented kws modules.
// It means, it should be the only entry point for user code. Implementation of derived factories should be
// placed by the implementation of KWS module it related for.



namespace mli_kws {

//==============================================================
// KWS Factory base class.
//
// Usage example:
//      my_kws_factory  builder; //instance of class derived from kws_factory
//      kws_info kws_cfg = builder.info();
//      char* state_fast_mem = ...; // Allocate fast memory for state (see kws_info type)
//
//      kws_module* kws = builder.create_module(state_fastmem_a);
//      ... // work with module
//      delete kws; // Module is allocated on heap, so it should be managed
//==============================================================
class kws_factory {
 public:

    // Virtual destructor to prevent memory leakage
    virtual ~kws_factory() = default;

    // Information about related KWS module
    //
    // return: kws_info structure with with all required info on related KWS module
    virtual kws_info info() = 0;

    // Create KWS module and return to the caller
    //
    // param:
    // state_fastmem_a      - pointer to fast memory A to keep module state.
    //                        Size of memory must be the kws_info::state_fastmem_a_sz.
    //
    // return: Pointer to allocated and initialized KWS module. Caller is responsible to manage module afterward.
    virtual kws_module* create_module(void* state_fastmem_a) = 0;
};


//==============================================================
// DS CLSTM Module factory
//
//==============================================================
class dsconv_lstm_factory : public kws_factory {
 public:

    // Information about related KWS module
    kws_info info() override;

    // Create KWS module and return to the caller
    kws_module* create_module(void* state_fastmem_a) override;
};

} // namespace mli_kws {


#endif // #ifndef _KWS_FACTORY_H_

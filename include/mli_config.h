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

/**
 * @file MLI Library Configuration header
 *
 * @brief This header defines MLI Library configuration 
 */

#ifndef _MLI_CONFIG_H_
#define _MLI_CONFIG_H_
/**
* Define Library build configuration options
*/

/**
* Concatenate primitive: Maximum number of tensor might be concatenated.
*/
#define MLI_CONCAT_MAX_TENSORS (8)

/**
* Library Debug mode
*/
#define     DBG_MODE_RELEASE   (0) /*< No debug. Messages:OFF; Assertions:OFF; ReturnCodes: Always OK */
#define     DBG_MODE_RET_CODES (1) /*< Return codes mode. Messages:OFF; Assertions:OFF; ReturnCodes: Valid Return*/
#define     DBG_MODE_ASSERT    (3) /*< Assert. Messages:OFF; Assertions:ON; Extra Assertions:OFF; ReturnCodes: Valid Return */
#define     DBG_MODE_DEBUG     (3) /*< Debug. Messages:ON; Assertions:ON; Extra Assertions:OFF; ReturnCodes: Valid Return */
#define     DBG_MODE_FULL      (4) /*< Full Debug. Messages:ON; Assertions:ON; Extra Assertions:ON; ReturnCodes: Valid Return */

#ifndef MLI_DEBUG_MODE
#define MLI_DEBUG_MODE (DBG_MODE_RELEASE)
#endif

/**
* Define platform specific data
*/
#include <stdint.h>

#include <arc/xy.h>

#ifdef __FXAPI__
#include <stdfix.h>
#else
#error "ARC FX Library (FXAPI) is required dependency"
#endif

/*
* Define the platform (according to pre-defined macro or according to HW config)
* 1 - ARCV2DSP ISA
* 2 - ARCV2DSP ISA with XY Memory
* 3 - ARCV2DSP ISA with 64bit operands (HS Only)
*/

#if defined(V2DSP_XY) || ((defined __Xxy) && !(defined(V2DSP) || defined(V2DSP_WIDE)))
/* Platform with XY memory (EM9D or EM11D) */
#undef V2DSP_XY
#define ARC_PLATFORM (2)
#define ARC_PLATFORM_STR  "ARCv2DSP XY"

#elif defined(V2DSP_WIDE) || ((defined __Xdsp_wide) && !(defined(V2DSP) || defined(V2DSP_XY)))
/* Platform with wide DSP ISA (HS45D or HS47D) */
#undef V2DSP_WIDE
#define ARC_PLATFORM (3)
#define ARC_PLATFORM_STR  "ARCv2DSP Wide"

#elif defined(V2DSP) || ((defined(__Xdsp2) || defined(__Xdsp_complex)) && !(defined(V2DSP_XY) || defined(V2DSP_WIDE)))
/* Platform with DSP ISA (EM5D or EM7D) */
#undef V2DSP
#define ARC_PLATFORM (1)
#define ARC_PLATFORM_STR  "ARCv2DSP"

#else
#error "Target platform is undefined or defined incorrectly"
#endif

#define     V2DSP      (1)
#define     V2DSP_XY   (2)
#define     V2DSP_WIDE (3)

/*
* Re-define ML pointers for XY specific platform
*/
#if (ARC_PLATFORM == V2DSP_XY)
#define MLI_PTR(p) __xy p *
#else
#define MLI_PTR(p) p *
#endif

#endif // _MLI_CONFIG_H_

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

#ifndef _ARC

#include <stdint.h>

unsigned int __JLI_TABLE__[1];

typedef struct {
    void (*target)();
} vect_entry_type;

#define INT_VECTOR_BASE 0x25
#define VECT_START _lr(INT_VECTOR_BASE)
#define IRQ_INTERRUPT 0x40b
#define IRQ_PRIORITY  0x206
#define IDENTITY      0x0004
#define STATUS32      0x000a
#define ISA_CONFIG    0x00c1

void _setvecti(int vector, _Interrupt void(*target)()) {
    volatile vect_entry_type *vtab = (_Uncached vect_entry_type *)VECT_START;
    vtab[vector].target = (void (*)())target;
    return;
}


void _sleep(int n) {
    return;
}

void _init_ad(void) {
    uint32_t stat32 = _lr(STATUS32) | 0x80000;
    __builtin_arc_flag(stat32);
}

int start_init(void) {
    uint32_t status = 0; //OK
    uint32_t identity_rg = _lr(IDENTITY);
    if ((identity_rg & 0xff) > 0x40 ) {
        //ARCv2EM
        uint32_t isa= _lr(ISA_CONFIG);
        if ( isa & 0x400000 ) { //check processor support unaligned accesses
            _init_ad(); //allows unaligned accesses
        } else {
           status = 1; //Error
        }
    } else {
        status = 1; //Error
    }

    return status;
}

#endif
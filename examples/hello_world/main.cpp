/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include "mli_api.h"

#include <assert.h>
#include <stdio.h>

#include "mli_config.h"

#if (PLATFORM == V2DSP_XY)
#define DATA_ATTR __xy __attribute__((section(".Xdata")))

#elif (PLATFORM == V2DSP_VECTOR)
#define DATA_ATTR __vccm __attribute__((section(".vecmem_data")))

#else
#define DATA_ATTR

#endif

#define NUM_ELEMS 8

int16_t DATA_ATTR data_in1[NUM_ELEMS] = { 1, 2, 3, 4, 5, 6, 7, 8 };
int16_t DATA_ATTR data_in2[NUM_ELEMS] = { 10, 20, 30, 40, 50, 60, 70, 80 };
int16_t DATA_ATTR data_out[NUM_ELEMS] = { 0, 0, 0, 0, 0, 0, 0, 0 };

int main() {
    mli_tensor in1 = { 0 };
    in1.el_type = MLI_EL_FX_16;
    in1.rank = 1;
    in1.shape[0] = NUM_ELEMS;
    in1.mem_stride[0] = 1;
    in1.data.capacity = sizeof(data_in1);
    in1.data.mem.pi16 = (int16_t *)data_in1;

    mli_tensor in2 = { 0 };
    in2.el_type = MLI_EL_FX_16;
    in2.rank = 1;
    in2.shape[0] = NUM_ELEMS;
    in2.mem_stride[0] = 1;
    in2.data.capacity = sizeof(data_in2);
    in2.data.mem.pi16 = (int16_t *)data_in2;

    mli_tensor out = { 0 };
    out.el_type = MLI_EL_FX_16;
    out.rank = 1;
    out.shape[0] = NUM_ELEMS;
    out.mem_stride[0] = 1;
    out.data.capacity = sizeof(data_out);
    out.data.mem.pi16 = (int16_t *)data_out;

    printf("in1:\n");
    for (int i = 0; i < NUM_ELEMS; i++) {
        printf("%d ", in1.data.mem.pi16[i]);
    }
    printf("\nin2:\n");
    for (int i = 0; i < NUM_ELEMS; i++) {
        printf("%d ", in2.data.mem.pi16[i]);
    }

    mli_status status;
    status = mli_krn_eltwise_add_fx16(&in1, &in2, &out);
    assert(status == MLI_STATUS_OK);

    printf("\nmli_krn_eltwise_add_fx16 output:\n");
    for (int i = 0; i < NUM_ELEMS; i++) {
        printf("%d ", out.data.mem.pi16[i]);
    }

    status = mli_krn_eltwise_sub_fx16(&in1, &in2, &out);
    assert(status == MLI_STATUS_OK);

    printf("\nmli_krn_eltwise_sub_fx16 output:\n");
    for (int i = 0; i < NUM_ELEMS; i++) {
        printf("%d ", out.data.mem.pi16[i]);
    }
    printf("\n");

    return 0;
}

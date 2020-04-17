/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_MATH_MACROS_H_
#define _MLI_MATH_MACROS_H_

#define MAX(A,B) (((A) > (B))? (A): (B))
#define MIN(A,B) (((A) > (B))? (B): (A))

#define CEIL_DIV(num,den) (((num) + (den) - 1)/(den))

#define MLI_MAT_MUL_Q31_SHIFT 31

#endif // _MLI_MATH_MACROS_H_

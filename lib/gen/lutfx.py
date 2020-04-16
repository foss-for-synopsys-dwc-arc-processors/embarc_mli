# Copyright 2019-2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#
"""
embarc Machine Learning Inference Library (embarc MLI):
Lookup Table generator for mli_prv_activation_lut function. 

For usage examples see mli_prv_lut.cc file (<embarc_mli_repo>/lib/src/private/src/mli_prv_lut.cc)
"""

from __future__ import print_function
import os
import sys
import argparse
import math

parser = argparse.ArgumentParser(description='Creates lookup table(LUT) data C code',
    epilog="'expr' can be any valid python expression containig math functions. "
    "Example to generate tanh table: lutfx.py -t FX16 -o size/2 -s 178 -qi 4 -f tanh(x)")
parser.add_argument('-v', '--verbose', action='count', help='Sets verbosity level')
parser.add_argument('-f', '--function', type=str, default='x', help='Sets LUT function', metavar='expr')
parser.add_argument('-s', '--lut_size', type=str, default='16', help='Sets LUT element count', metavar='expr')
parser.add_argument('-o', '--lut_offset', type=str, default='lut_size/2', help='Sets LUT index for f(0)', metavar='expr')
parser.add_argument('-qi', '--lut_in_fract', type=int, default=4, help='Sets LUT input fractional bits', metavar='value')
parser.add_argument('-qo', '--lut_out_fract', type=int, default=-1, help='Sets LUT output fractional bits', metavar='value')
parser.add_argument('-t', '--lut_format', type=str, default='FX16', help='Sets LUT output format', metavar='option',
    choices=['FX8', 'FX16', 'FX32'])

def fxp2float(fxp_value, fract_bits):
    return float(fxp_value) / (1 << fract_bits)

def float2fxp(float_value, fract_bits):
    x = float_value * (1 << fract_bits)
    return int(x + 0.5 if x > 0 else x - 0.5)

# get string description of Q format (sign bit is not included)
def qformat(total_bits, fract_bits):
    intbits = str(total_bits - fract_bits - 1) + '.' if total_bits > fract_bits + 1 else ''
    return "Q" + intbits + str(fract_bits)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def expneg(x):
    return math.exp(x) if x <= 0 else 1.0

def main():
    args = parser.parse_args()

    # parse lut_size argument
    lut_size = eval('lambda : int(' + args.lut_size + ')', {})()
    # build dict of helper symbols
    evalsymbols = {name: getattr(math, name) for name in dir(math) if name[0] != '_'}
    evalsymbols.update({'size': lut_size, 'sigm': sigmoid, 'expneg': expneg})
    # parse lut_in_offset and lut_function arguments
    lut_in_offset = eval('lambda : int(' + args.lut_offset + ')', evalsymbols)()
    lut_function = eval('lambda x: float(' + args.function + ')', evalsymbols)

    lut_in_delta = fxp2float(1, args.lut_in_fract)
    if lut_size < 256:
        lut_in_bits = 8
    elif lut_size < 65536:
        lut_in_bits = 16
    else:
        lut_in_bits = 32

    if args.lut_format == 'FX8':
        number_width = 4
        lut_out_bits = 8
        lut_out_min = -2**7
        lut_out_max = +2**7-1
        lut_out_fract = args.lut_out_fract if args.lut_out_fract >= 0 else 7
    elif args.lut_format == 'FX16':
        number_width = 6
        lut_out_bits = 16
        lut_out_min = -2**15
        lut_out_max = +2**15-1
        lut_out_fract = args.lut_out_fract if args.lut_out_fract >= 0 else 15
    elif args.lut_format == 'FX32':
        number_width = 11
        lut_out_bits = 32
        lut_out_min = -2**31
        lut_out_max = +2**31-1
        lut_out_fract = args.lut_out_fract if args.lut_out_fract >= 0 else 31

    lut_out_delta = fxp2float(1, lut_out_fract)

    err_max = 0
    lut = []
    for lut_index in range(lut_size):
        x = fxp2float(lut_index - lut_in_offset, args.lut_in_fract)
        y = lut_function(x)

        lut_out = float2fxp(y, lut_out_fract)
        lut_out = lut_out_min if lut_out < lut_out_min else lut_out
        lut_out = lut_out_max if lut_out > lut_out_max else lut_out

        lut.append(lut_out)

        if lut_index == 0:
            lut_out_prev = lut_out

        # determine max error between current and previous points
        # (starting from lut_index == 1)
        err_max_local = 0.0
        if lut_index > 0 and lut_index < lut_size:
            for test_step in range(1, 256):
                # calculate reference value
                ref_y = lut_function(x - (lut_in_delta * test_step / 256.0))
                # calculate linear-interpolated LUT value
                lin_y = fxp2float((lut_out + (lut_out_prev - lut_out) * (test_step / 256.0)), lut_out_fract)
                # calculate max error above lut_out_delta
                delta = abs(lin_y - ref_y)
                err_max_local = delta if err_max_local < delta else err_max_local

        lut_out_prev = lut_out

        flut = fxp2float(lut_out, lut_out_fract)
        err_max = err_max_local if err_max < err_max_local else err_max
        if args.verbose:
            print("lut[%4d]=%6d: x=%+f fref=%+f flut=%+f err=%f (%d lsb)" %
                (lut_index - lut_in_offset, lut_out, x, y, flut, 
                    err_max_local, float2fxp(err_max_local, lut_out_fract)))

    if args.verbose:
        print("lut_size  = %d" % lut_size)
        print("in_offset = %d" % lut_in_offset)
        print("in_delta  = %f" % lut_in_delta)
        print("out_delta = %f" % lut_out_delta)
        print("out_error = %f (%d lsb)" % (err_max, float2fxp(err_max, lut_out_fract)))

    print(
"""
/*
    *** Generated by %s ***
    arguments  = %s
    lut_size   = %d
    in_offset  = %d
    in_format  = %s
    out_format = %s
    out_error  = %f (%d lsb) (linear interpolation)
*/
"""
        % (os.path.basename(__file__),
        ' '.join(["'"+arg+"'" if "(" in arg else arg for arg in sys.argv[1:]]),
        lut_size,
        lut_in_offset,
        qformat(lut_in_bits, args.lut_in_fract),
        qformat(lut_out_bits, lut_out_fract),
        err_max, float2fxp(err_max, lut_out_fract)), end=''
    )

    print("static const int%d_t lut_data[] = {" % lut_out_bits)
    length = len(lut)
    for i in range(length):
        print('    ' if (i % 8) == 0 else '', end='')
        print(('%%+%dd' % number_width) % lut[i], end='')
        print('' if i == length-1 else ', ',
            end=('\n' if (i+1) % 8 == 0 and i != length-1 else ''))
    print("\n};")

if __name__ == '__main__':
    main()

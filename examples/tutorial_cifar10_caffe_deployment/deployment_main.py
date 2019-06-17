# Copyright 2019, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#
"""
embarc Machine Learning Inference Library (embarc MLI):
Model Deployment tutorial for Caffe and CIFAR-10: Launching script
"""

import argparse
from deployment_steps import *


def main(argn):
    # Go through the model deployment tutorial for Caffe and CIFAR-10 step-by-step
    print('MLI Deployment tutorial script.')
    print('Used parameters:')
    for param, val in vars(argn).items():
        print('\t{}: {}'.format(param, val))
    print("\n")

    print("Step 1: Instrument the input model.")
    classifier, layers, ir_tensors = instrument_model_step(argn.prototext, argn.model)
    print("Done\n")

    print("Step 2: Collect data statistic for each layer.")
    collect_inference_statistic_step(classifier, argn.lmdb_data_dir, ir_tensors, 128.0, 1.0 / 128.0)
    print("Done\n")

    print("Step 3: Define Qm.n data format.")
    define_qmn_step(layers, argn.kernel_type)
    print("Done\n")

    print("Step 4: Quantize and output weights.")
    quantize_and_output_step(layers, argn.kernel_type, argn.output_prefix, argn.wrap_coefficients)
    print("Done\n")

    # Next step should be done manually using generated files
    # write_code_for_me(...)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CIFAR-10 for Caffe Deployment Tutorial Script ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--lmdb_data_dir',
        type=str,
        default='cifar10_train_lmdb',
        help='Path to CIFAR-10 subset in LMDB format (created by caffe cifar-10 example specific tool)')
    parser.add_argument(
        '--model',
        type=str,
        default='cifar10_small.caffemodel.h5',
        help='CIFAR-10 caffemodl file')
    parser.add_argument(
        '--prototext',
        type=str,
        default='cifar10_small.prototxt',
        help='CIFAR-10 test prototext file')
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='cifar10_small',
        help='Prefix for output files including *.inc(coefficients) *.c (tensors) *.h (declarations)')
    parser.add_argument(
        '--kernel_type',
        type=str,
        default='fx8',
        choices=['fx8', 'fx16', 'fx8w16d'],
        help='')
    parser.add_argument(
        '--wrap_coefficients',
        action='store_true',
        help='Wrap float values into macro for compile-time quantization purpose')

    flags, unparsed = parser.parse_known_args()
    main(flags)

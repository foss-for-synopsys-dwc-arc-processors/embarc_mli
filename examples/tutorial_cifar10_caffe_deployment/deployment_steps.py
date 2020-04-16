# Copyright 2019-2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#
#
"""
embarc Machine Learning Inference Library (embarc MLI):
Tutorial deployment steps file. Each function reflects one step from data deployment subset
"""

import caffe
import lmdb
import math
import numpy as np
import os

from mli_fxtools import TensorRepresentation, MacBasedKernelRepresentation


QUANT_DEFINES_TEXT = """
//========================================================  
// 
// Common Qmn Data Transformation define. Round-to-Nearest
//
//========================================================
// Quick quantization
#define QMN(type, fraq, val)   (type)(val * (1u << (fraq)) + ((val >= 0)? 0.5f: -0.5f))
#define FRQ_BITS(int_part, el_type) ((sizeof(el_type)*8)-int_part-1)

// Quantization with saturation (slower and higher compile-time memory consumption)
// #ifndef MAX
// #define MAX(A,B) (((A) > (B))? (A): (B))
// #endif
// #ifndef MIN
// #define MIN(A,B) (((A) > (B))? (B): (A))
// #endif
// 
// #define EL_MAX(type) (type)((1u << (sizeof(type)*8-1))-1)
// #define EL_MIN(type) (type)(-(1u << (sizeof(type)*8-1)))
// #define SAT(type, val) (MIN(EL_MAX(type), MAX(EL_MIN(type), val)))
// 
// #define QMN(type, fraq, val)   (type)SAT(type, ((val) * (1u << (fraq)) + (((val) >= 0)? 0.5f: -0.5f)))
// #define FRQ_BITS(int_part, el_type) ((sizeof(el_type)*8)-int_part-1)"""


TENSOR_DESCR_TEMPLATE = """
const mli_tensor {name} = {{
    .data = (void *){data},
    .capacity = {capacity},
    .shape = {{{shape}}},
    .rank = {rank},
    .el_type = {el_type},
    .el_params.fx.frac_bits = {frac_bits}
}};
"""


CONST_VAL_TEMPLATE = "const uint8_t {name} = {val};"


def instrument_model_step(prototxt_path, model_path):
    """
    Initialize the caffe model and identify  lists with layers / tensors for the following analysis.

    Parameters
    ----------
    model_path : str
        Path to *.caffemodel file
    prototxt_path : str
        Path to caffemodel description (prototext) file

    Raises
    ------
    RuntimeError
        If number of identefied layer or tensors is 0 (probably bad input aruments).

    Returns
    -------
    outs : (classifier object, identified layers list, identified ir tensors list) tuple.
    """
    caffe.init_log()
    caffe.set_mode_cpu()
    classifier = caffe.Net(prototxt_path, model_path, caffe.TEST)

    # Construct list and dictionary objects for the following filling
    layers_to_quantization = []
    ir_tensors = {}

    # Parse Caffe model (very basic) to fill layers and tensors structures and tune relationships
    for layer_name, param_blobs in classifier.params.items():
        if classifier.layer_dict[layer_name].type in ('InnerProduct', 'Convolution'):
            new_layer_repr = MacBasedKernelRepresentation(param_blobs[0].data, param_blobs[1].data)
            input_name = classifier.bottom_names[layer_name][0]
            output_name = classifier.top_names[layer_name][0]
            if input_name not in ir_tensors:
                ir_tensors[input_name] = TensorRepresentation()
            if output_name not in ir_tensors:
                ir_tensors[output_name] = TensorRepresentation()

            new_layer_repr.input = ir_tensors[input_name]
            new_layer_repr.output = ir_tensors[output_name]
            layers_to_quantization.append((layer_name, new_layer_repr))

    # Print side info and output (constructed object tuple)
    if len(layers_to_quantization) == 0 or len(ir_tensors) == 0:
        raise RuntimeError("Layers/blobs wasn't defined properly from model files.")
    else:
        print('Layers with params: {}'.format([name for name, _ in layers_to_quantization]))
        print('Input/output tensors: {}'.format(ir_tensors.keys()))
    return classifier, layers_to_quantization, ir_tensors


def collect_inference_statistic_step(classifier, lmdb_data_dir, ir_tensors, input_mean=0.0, input_scale=1.0):
    """
    Initialize the caffe model and identify  lists with layers / tensors for the following analysis.

    Parameters
    ----------
    classifier : caffe Net object
        initialize classifyer object with inference (forward_all) function.
    lmdb_data_dir : str
        path to dataset (lmdb format) with input vectors for IR tensors analisys
    ir_tensors: dict {tensor_name: TensorRepresentation()}
        dictionary with identified tensors for inference-time analisis
    input_mean: float
        mean value for input normalization by the following formula: in[i] = (in[i] - input_mean) * input_scale
    input_scale: float
        scale value for input normalization by the following formula: in[i] = (in[i] - input_mean) * input_scale


    Returns
    -------
    outs : {tensor_name: TensorRepresentation()} dict
        tensors with accumulated statistic for input dataset.
    """

    # Open dataset and initialize helper variables
    lmdb_env = lmdb.open(lmdb_data_dir)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    out_key = classifier.outputs[0]
    correct = 0
    total = 0

    # Iterate dataset with updating data statistic
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        data_raw = caffe.io.datum_to_array(datum)

        data = np.asarray([(data_raw - input_mean) * input_scale])
        label = datum.label
        pred = classifier.forward_all(data=data)[out_key]
        pred_class = np.argmax(pred)
        for name in ir_tensors.keys():
            ir_tensors[name].account(classifier.blobs[name].data)

        # Accumulate classification accuracy for the model
        correct += 1 if pred_class == label else 0
        total += 1
        if total % 1000 == 0 and total != 0:
            print("Processed {} vectors ({} are correct {:.3f}%)".format(total, correct,
                                                                         (float(correct) / total) * 100.0))

    # Print info and return list with updated tensors
    if total == 0:
        print("WARNING: Inference statistic wasn't gathared. Set default intermediate tensors range (-8 : 8)")
        for name in ir_tensors.keys():
            ir_tensors[name].account([-7.99, 7.99])
    else:
        print("Processed {} vectors ({} are correct {:.3f}%)".format(total, correct, float(correct) / total * 100.0))
    return ir_tensors


def define_qmn_step(layers, kernel_type):
    """
    Print tensors parameters for each layer according to kernel_type.

    Parameters
    ----------
    layers : list
        Identified layers with accumulated statistic
    kernel_type : str
        One of  MLI supported kernel types ('fx8', 'fx16', 'fx8w16d') or 'float'
    """
    str_template = '\t{}: q{}.{} (range [{:.3f} : {:.3f}])'
    for name, layer in layers:
        layer_params = layer.get_layer_data(kernel_type)
        print('{}:'.format(name))
        for tensor_name in layer_params.keys():
            print(str_template.format(tensor_name,
                                      layer_params[tensor_name]['int_bits'], layer_params[tensor_name]['frac_bits'],
                                      layer_params[tensor_name]['min'], layer_params[tensor_name]['max']))


def quantize_and_output_step(layers, kernel_type, out_files_prefix, to_wrap_coefficients):
    """
    Generate C source files with quantized model parameters and descriptive tensor structures
    that might be used by MLI based application. The following files  will be generated:
         <out_files_prefix>_tensors.h - header file with declarations of entities (tensors and constants)
         <out_files_prefix>_tensors.c - source file with definitions of entities (tensors and constants)
          <out_files_prefix>_coeff.inc - include file with quantized (or wrapped) coefficients

    Parameters
    ----------
    layers : list
        Identified layers with accumulated statistic
    kernel_type : str
        One of  MLI supported kernel types ('fx8', 'fx16', 'fx8w16d') or 'float'
    out_files_prefix : str
        Common prefix for output source files
    to_wrap_coefficients: bool
        Whether coefficients must be wrapped into transformation macro for compile-time quantization.
        Also implies generation of some helper macro functions.
    """

    def print_nparr_to_file(outfile, np_data, coeffs_in_line=-1, cast_prefix=None):
        """
        Output values from numpy array into source file.

        Parameters
        ----------
        outfile : file descriptor object
            opened text file to output
        np_data : numpy array
            data to output
        coeffs_in_line : int
            Number of values in output line. if <=0 outputs all in one line
        cast_prefix: str
            wrapping string for each value: "cast_prefix(val[i]), ..."
        """
        flat_np_data = np_data.flatten()
        vals_per_line = flat_np_data.size if not (0 < coeffs_in_line < flat_np_data.size) else coeffs_in_line

        if np.issubdtype(flat_np_data.dtype, np.floating):
            val_template = '{: > 11.9f}'
        elif np.issubdtype(flat_np_data.dtype, np.signedinteger):
            max_chars = int(math.log10(np.iinfo(flat_np_data.dtype).max) + 2)
            val_template = ''.join(('{: > ', str(max_chars), 'd}'))
        else:
            raise RuntimeError("Unsupported data type ({})".format(str(flat_np_data.dtype)))

        if type(cast_prefix) is str:
            val_template = ''.join((cast_prefix, '(', val_template, ')'))

        for idx in range(flat_np_data.size - 1):
            if idx != 0 and (idx % vals_per_line) == 0:
                outfile.write('\\\n\t')
            outfile.write(val_template.format(flat_np_data[idx]) + ',')
        outfile.write(val_template.format(flat_np_data[-1]))

    # Test Files IO
    coefficients_fpath = '_'.join((out_files_prefix, 'coeff.inc'))
    tensors_fpath = '_'.join((out_files_prefix, 'tensors.c'))
    headet_fpath = '_'.join((out_files_prefix, 'tensors.h'))
    print("Generate next files:\n\t{}\n\t{}\n\t{}".format(headet_fpath, tensors_fpath, coefficients_fpath))
    with open(coefficients_fpath, 'w') as coeff_file:
        with open(tensors_fpath, 'w') as tensr_file:
            with open(headet_fpath, 'w') as header_file:
                # Write headers / initial information for each output file
                tensr_file.write("// AUTO GENERATED FILE\n\n")
                tensr_file.write("#include <mli_api.h>\n")
                tensr_file.write('#include "{}"\n\n'.format(os.path.split(coefficients_fpath)[1]))
                coeff_file.write("// AUTO GENERATED FILE\n\n")
                header_file.write("// AUTO GENERATED FILE\n\n")
                header_file.write("#include <mli_api.h>\n\n")

                # In case we need to output wrapped float coefficients, we must put additional macro code and
                # declare additional definitions and replace parameter
                params_type = 'int8_t' if 'fx8' in kernel_type else 'int16_t'
                el_type_val = 'MLI_EL_FX_8' if 'fx8' in kernel_type else 'MLI_EL_FX_16'
                if to_wrap_coefficients:
                    coeff_file.write("{}\n\n".format(QUANT_DEFINES_TEXT))
                    tensr_file.write('#define W_EL_TYPE ({})\n'.format(el_type_val))
                    el_type_val = 'W_EL_TYPE'
                    tensr_file.write('typedef {} w_type;\n'.format(params_type))
                    params_type = 'w_type'

                for layer_idx in range(len(layers)):
                    name, layer = layers[layer_idx]
                    name_cap = name.upper()
                    prefix = ''.join(('L', str(layer_idx + 1), '_', name))
                    layer_params = layer.get_layer_data(kernel_type)
                    wt_def = name_cap + "_W"
                    b_def = name_cap + "_B"

                    # Filling header file with declarations of new entities
                    header_file.write("extern const uint8_t {};\n".format(prefix + "_in_frac_bits"))
                    header_file.write("extern const uint8_t {};\n".format(prefix + "_out_frac_bits"))
                    header_file.write("extern const mli_tensor {};\n".format(prefix + "_wt"))
                    header_file.write("extern const mli_tensor {};\n\n".format(prefix + "_b"))

                    # Filling tensors file with definitions of new entities (including tensors structures)
                    tensr_file.write("\n//   {} params\n//{}\n".format(name, '=' * 54))
                    tensr_file.write("const {} {} = {};\n".format(params_type, prefix + '_wt_buf[]', wt_def))
                    tensr_file.write("const {} {} = {};\n".format(params_type, prefix + '_b_buf[]', b_def))
                    tensr_file.write(CONST_VAL_TEMPLATE.format(name=prefix + "_in_frac_bits",
                                                               val=layer_params['input']['frac_bits']) + '\n')
                    tensr_file.write(CONST_VAL_TEMPLATE.format(name=prefix + "_out_frac_bits",
                                                               val=layer_params['output']['frac_bits']) + '\n')
                    tensr_file.write(TENSOR_DESCR_TEMPLATE.format(
                        name=prefix + "_wt", data=prefix + '_wt_buf',
                        capacity="sizeof({})".format(prefix + "_wt"),
                        shape=','.join(map(str, layer_params['weights']['data'].shape)),
                        rank=len(layer_params['weights']['data'].shape),
                        el_type=el_type_val,
                        frac_bits=layer_params['weights']['frac_bits'] if not to_wrap_coefficients else wt_def + 'FRAQ'
                    ))
                    tensr_file.write(TENSOR_DESCR_TEMPLATE.format(
                        name=prefix + "_b", data=prefix + '_b_buf',
                        capacity="sizeof({})".format(prefix + "_b"),
                        shape=','.join(map(str, layer_params['bias']['data'].shape)),
                        rank=len(layer_params['bias']['data'].shape),
                        el_type=el_type_val,
                        frac_bits=layer_params['bias']['frac_bits'] if not to_wrap_coefficients else b_def + 'FRAQ'
                    ))

                    # Filling coefficients files with quantized (or wrapped) coefficients
                    if to_wrap_coefficients:
                        float_params = layer.get_layer_data('float')
                        weights = float_params['weights']['data']
                        bias = float_params['bias']['data']
                        wcast_prefix = 'L' + str(layer_idx + 1) + '_QMN_W'
                        bcast_prefix = 'L' + str(layer_idx + 1) + '_QMN_B'
                        coeffs_per_line = 10
                    else:
                        weights = layer_params['weights']['data']
                        bias = layer_params['bias']['data']
                        wcast_prefix = None
                        bcast_prefix = None
                        coeffs_per_line = 32 if 'fx8' in kernel_type else 16

                    # weights
                    coeff_file.write("//   {} constants\n//{}\n".format("_".join((name_cap, 'W')), '=' * 54))
                    if to_wrap_coefficients:
                        coeff_file.write("#define {} (FRQ_BITS({}, {}))\n".format(wt_def + 'FRAQ',
                                                                                  layer_params['weights']['int_bits'],
                                                                                  params_type))
                        coeff_file.write("#define {}(val) QMN({},{},val)\n".format(wcast_prefix,
                                                                                   params_type,
                                                                                   wt_def + 'FRAQ'))
                    coeff_file.write("#define {} {{ \\\n\t".format(wt_def))
                    print_nparr_to_file(coeff_file, weights, coeffs_per_line, wcast_prefix)
                    coeff_file.write("}\n\n")

                    # Biases in the same way as weights
                    coeff_file.write("//   {} constants\n//{}\n".format("_".join((name_cap, 'B')), '=' * 54))
                    if to_wrap_coefficients:
                        coeff_file.write("#define {} (FRQ_BITS({}, {}))\n".format(b_def + 'FRAQ',
                                                                                  layer_params['bias']['int_bits'],
                                                                                  params_type))
                        coeff_file.write("#define {}(val) QMN({},{},val)\n".format(bcast_prefix,
                                                                                   params_type,
                                                                                   b_def + 'FRAQ'))
                    coeff_file.write("#define {} {{ \\\n\t".format(b_def))
                    print_nparr_to_file(coeff_file, bias, coeffs_per_line, bcast_prefix)
                    coeff_file.write("}\n\n")

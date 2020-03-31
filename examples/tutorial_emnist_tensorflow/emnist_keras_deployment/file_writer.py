import os
from pathlib import Path
from math import ceil

import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Activation, Flatten
from tensorflow.keras.models import Model
import idx2numpy

from . import constants

@constants.execution_log_decorator
def write_weights(model, weights, biases, to_wrap_coefficients):
    '''Create file, containing weights and biases'''
    path = Path(constants.MODEL_FOLDER) / constants.WEIGHTS_FILE_NAME
    path = Path(constants.MODEL_FOLDER) / constants.WEIGHTS_FILE_NAME
    with path.open('w') as weights_file:
        weights_file.write(constants.WEIGHTS_FILE_INCLUDES.format(
            constants_file_name=constants.CONSTANTS_FILE_NAME
        ))
        weights_file.write('\n')

        conv_dense_idx = 0
        for layer in model.layers:
            if (isinstance(layer, Conv2D) or isinstance(layer, Dense)):
                conv_dense_idx += 1
                weights_type = "" if conv_dense_idx < len(weights) / 2 else "2"

                # Write weights to file
                layer_weights = weights[layer.name]['data']
                if(isinstance(layer, Conv2D)):
                    weights_file.write(constants.CONVOLUTION_WEIGHTS_ARRAY_NAME.format(
                        type=weights_type, layer_idx=conv_dense_idx
                    ) + ' ={\\\n    ')

                    layer_weights = layer_weights.transpose(3,2,0,1)

                if(isinstance(layer, Dense)):
                    weights_file.write(constants.DENSE_WEIGHTS_ARRAY_NAME.format(
                        type=weights_type, layer_idx=conv_dense_idx
                    ) + ' ={\\\n    ')

                    if isinstance(prev_layer, Flatten):
                        layer_weights = layer_weights.transpose(1,0)
                        for out_idx in range(layer_weights.shape[0]):
                            row = layer_weights[out_idx]
                            row = row.reshape(prev_layer.input_shape[1],prev_layer.input_shape[2],
                                              prev_layer.input_shape[3]).transpose(2,0,1)
                            layer_weights[out_idx] = row.flatten()
                    else:
                        layer_weights = layer_weights.transpose(1, 0)

                layer_weights = layer_weights.flatten()
                for i in range(len(layer_weights)):
                    if to_wrap_coefficients:
                        weights_file.write(constants.WEIGHTS_QUANT_WRAPPER.format(
                            conv_dense_idx,
                            layer_weights[i]
                        ))
                    else:
                        weights_file.write(constants.WEIGHTS_BASIC_WRAPPER.format(
                            layer_weights[i]
                        ))

                    if i < len(layer_weights) - 1:
                        weights_file.write(',')
                        if (i + 1) % 10 == 0:
                            weights_file.write("\\\n    ")
                    else:
                        weights_file.write("\\\n};\n\n")


                # Write biases to file
                layer_biases = biases[layer.name]['data']
                if (isinstance(layer, Conv2D)):
                    weights_file.write(constants.CONVOLUTION_BIASES_ARRAY_NAME.format(
                        type=weights_type, layer_idx=conv_dense_idx
                    ) + ' ={\\\n    ')

                if (isinstance(layer, Dense)):
                    weights_file.write(constants.DENSE_BIASES_ARRAY_NAME.format(
                        type=weights_type, layer_idx=conv_dense_idx
                    ) + ' ={\\\n    ')

                layer_biases = layer_biases.flatten()
                for i in range(len(layer_biases)):
                    if to_wrap_coefficients:
                        weights_file.write(constants.BIASES_QUANT_WRAPPER.format(
                            conv_dense_idx, layer_biases[i]
                        ))
                    else:
                        weights_file.write(constants.BIASES_BASIC_WRAPPER.format(
                            layer_biases[i]
                        ))
                    if i < len(layer_biases) - 1:
                        weights_file.write(',')
                        if (i + 1) % 10 == 0:
                            weights_file.write("\\\n    ")
                    else:
                        weights_file.write("\\\n};\n\n")

            prev_layer = layer


@constants.execution_log_decorator
def write_constants_h(model, weights, biases, outputs):
    '''Create file, containing model layer parameters'''
    path = Path(constants.MODEL_FOLDER) / constants.CONSTANTS_FILE_NAME
    with path.open('w') as constants_file:
        constants_file.write(constants.CONSTANTS_FILE_GUARD_BEG.format(
            project_name_caps=constants.PROJECT_NAME_CAPS
        ))
        constants_file.write('\n')
        constants_file.write(constants.CONSTANTS_FILE_INCLUDES.format(
            project_name=constants.PROJECT_NAME
        ))
        constants_file.write('\n')

        if constants.KERNEL_TYPE != 'fx16':
            constants_file.write(constants.CONSTANTS_W_EL_TYPE_8)
        else:
            constants_file.write(constants.CONSTANTS_W_EL_TYPE_16)
        constants_file.write('\n')

        constants_file.write(constants.CONSTANTS_FILE_DEFINES)
        constants_file.write('\n')

        # Write weights arrays declaration
        conv_dense_idx = 0
        for layer in model.layers:
            if(isinstance(layer, Conv2D) or isinstance(layer, Dense)):
                conv_dense_idx += 1
                weights_type = "" if conv_dense_idx < len(weights) / 2 else "2"

                if(isinstance(layer, Conv2D)):
                    constants_file.write(constants.CONSTANTS_CONVOLUTION_WEIGHTS
                                         .format(
                        type=weights_type,
                        layer_idx=conv_dense_idx
                    ))
                    constants_file.write('\n')
                    constants_file.write(constants.CONSTANTS_CONVOLUTION_BIASES
                                         .format(
                        type=weights_type,
                        layer_idx=conv_dense_idx
                    ))
                    constants_file.write('\n')

                if(isinstance(layer, Dense)):
                    constants_file.write(constants.CONSTANTS_DENSE_WEIGHTS
                                         .format(
                        type=weights_type,
                        layer_idx=conv_dense_idx
                    ))
                    constants_file.write('\n')
                    constants_file.write(constants.CONSTANTS_DENSE_BIASES
                                         .format(
                        type=weights_type,
                        layer_idx=conv_dense_idx
                    ))
                    constants_file.write('\n')

        constants_file.write('\n\n')

        # Write int_bits values
        conv_dense_idx = 0
        for layer in model.layers:
            if(isinstance(layer, Conv2D)):
                conv_dense_idx += 1
                constants_file.write(constants.CONSTANTS_CONV_DEF.format(
                    layer_idx=conv_dense_idx,
                    w_bit=weights[layer.name]['int_bits'],
                    b_bit=biases[layer.name]['int_bits'],
                    o_bit=outputs[layer.name]['int_bits']
                ))

            if(isinstance(layer, Dense)):
                conv_dense_idx += 1
                constants_file.write(constants.CONSTANTS_DENSE_DEF.format(
                    layer_idx=conv_dense_idx,
                    w_bit=weights[layer.name]['int_bits'],
                    b_bit=biases[layer.name]['int_bits'],
                    o_bit=outputs[layer.name]['int_bits']
                ))

            constants_file.write('\n\n')

        conv_dense_idx = 0
        for layer in model.layers:
            if(isinstance(layer, Conv2D)):
                conv_dense_idx += 1
                layer_weights = weights[layer.name]['data']
                constants_file.write(constants.CONSTANTS_CONV_SHAPES.format(
                    layer_idx=conv_dense_idx,
                    s1=layer_weights.shape[3],
                    s2=layer_weights.shape[2],
                    s3=layer_weights.shape[0],
                    s4=layer_weights.shape[1],
                ))

            if(isinstance(layer, Dense)):
                conv_dense_idx += 1
                layer_weights = weights[layer.name]['data']
                constants_file.write(constants.CONSTANTS_DENSE_SHAPES.format(
                    layer_idx=conv_dense_idx,
                    s1=layer_weights.shape[1],
                    s2=layer_weights.shape[0]
                ))

            constants_file.write('\n\n')

        constants_file.write(constants.CONSTANTS_FILE_GUARD_END.format(
            project_name_caps=constants.PROJECT_NAME_CAPS
        ))


@constants.execution_log_decorator
def write_model_h(test_data_shape):
    '''Create file, containing model declaration'''
    path = Path(constants.MODEL_FOLDER) / constants.MODEL_H_FILE_NAME
    with path.open('w') as model_h_file:
        model_h_file.write(constants.MODEL_H_GUARD_BEG.format(
            project_name_caps=constants.PROJECT_NAME_CAPS
        ))
        model_h_file.write('\n\n')
        model_h_file.write(constants.MODEL_H_FILE_INCLUDES)
        model_h_file.write('\n')

        if constants.DEBUG_VERSION:
            model_h_file.write(constants.MODEL_H_FILE_DEFINES_DBG.format(
                project_name=constants.PROJECT_NAME,
                is1=test_data_shape[1],
                is2=test_data_shape[2],
                is3=test_data_shape[3],
                os1=constants.NUM_CLASSES,
            ))
        else:
            model_h_file.write(constants.MODEL_H_FILE_DEFINES.format(
                project_name=constants.PROJECT_NAME,
                is1=test_data_shape[1],
                is2=test_data_shape[2],
                is3=test_data_shape[3],
                os1=constants.NUM_CLASSES,
            ))
            
        model_h_file.write('\n')

        if constants.KERNEL_TYPE == 'fx8':
            model_h_file.write(constants.MODEL_H_D_TYPE_8)
        else:
            model_h_file.write(constants.MODEL_H_D_TYPE_16)

        model_h_file.write(constants.MODEL_H_GUARD_END.format(
            project_name_caps=constants.PROJECT_NAME_CAPS
        ))


def get_convolution_params(model):
    conv_kernel = 0
    conv_stride = 0
    conv_padding = 0

    for layer in model.layers:
        if(isinstance(layer, Conv2D)):
            conv_kernel = layer.kernel_size[0]
            conv_stride = layer.strides[0]
            conv_padding = int(conv_kernel / 2)
            break;

    return conv_kernel, conv_stride, conv_padding


def get_pooling_params(model):
    pool_kernel = 0
    pool_stride = 0
    pool_padding = 0

    for layer in model.layers:
        if(isinstance(layer, MaxPooling2D)):
            pool_kernel = layer.pool_size[0]
            pool_stride = layer.strides[0]
            break;

    return pool_kernel, pool_stride, pool_padding

def get_first_input_shape(model):
    return model.layers[0].input_shape[1], model.layers[0].input_shape[2], \
           model.layers[0].input_shape[3]

def get_first_output_shape(model):
    return model.layers[0].output_shape[3], model.layers[0].output_shape[1], \
           model.layers[0].output_shape[2]


@constants.execution_log_decorator
def write_model_c(model):
    '''Create file, containing model definition'''
    conv_kernel, conv_stride, conv_padding = get_convolution_params(model)
    pool_kernel, pool_stride, pool_padding = get_pooling_params(model)

    output_shape0, output_shape1, output_shape2 = get_first_output_shape(model)
    last_output_shape0 = model.layers[-1].output_shape[1]
    input_shape0, input_shape1, input_shape2 = get_first_input_shape(model)

    path = Path(constants.MODEL_FOLDER) / constants.MODEL_C_FILE_NAME
    with path.open('w') as model_c_file:
        if constants.DEBUG_VERSION:
            model_c_file.write(constants.MODEL_C_FILE_INCLUDES_DBG.format(
                project_name=constants.PROJECT_NAME
            ))
        else:
            model_c_file.write(constants.MODEL_C_FILE_INCLUDES.format(
                project_name=constants.PROJECT_NAME
            ))
        model_c_file.write('\n')

        if constants.KERNEL_TYPE == 'fx8':
            model_c_file.write(constants.MODEL_C_D_EL_TYPE_8)
        else:
            model_c_file.write(constants.MODEL_C_D_EL_TYPE_16)
        model_c_file.write('\n')

        #if not constants.DEBUG_VERSION:
        #    model_c_file.write(constants.MODEL_C_CONVERT_FLOAT_FUNC)
        #    model_c_file.write('\n')

        #    model_c_file.write(constants.MODEL_C_ALL_PRED_FUNC.format(
        #        project_name=constants.PROJECT_NAME
        #    ))
        #    model_c_file.write('\n')

        #    model_c_file.write(constants.MODEL_C_TOP_N_FUNC.format(
        #        project_name=constants.PROJECT_NAME
        #    ))
        #    model_c_file.write('\n')

        model_c_file.write(constants.MODEL_C_FILE_DEFINES.format(
            most_buf_s0=output_shape0,
            most_buf_s1=output_shape1,
            most_buf_s2=output_shape2,
            next_buf_s0=ceil(output_shape0 / 2),
            next_buf_s1=output_shape1,
            next_buf_s2=output_shape2
        ))
        model_c_file.write('\n')
        model_c_file.write(constants.MODEL_C_IO_TENSORS.format(
            project_name=constants.PROJECT_NAME,
            is0=input_shape0,
            is1=input_shape1,
            is2=input_shape2,
            os0=last_output_shape0,
            in_frac_bits=constants.FRAC_BITS,
            out_frac_bits=0
        ))

        model_c_file.write('\n\n')
        model_c_file.write(constants.MODEL_C_CONV_CFG.format(
            stride_height=conv_stride, stride_width=conv_stride,
            padding_bottom=conv_padding, padding_top=conv_padding,
            padding_left=conv_padding, padding_right=conv_padding
        ))
        model_c_file.write('\n\n')
        model_c_file.write(constants.MODEL_C_POOL_CFG.format(
            kernel_height=pool_kernel, kernel_width=pool_kernel,
            stride_height=pool_stride, stride_width=pool_stride,
            padding_bottom=pool_padding, padding_top=pool_padding,
            padding_left=pool_padding, padding_right=pool_padding
        ))
        model_c_file.write('\n\n')

        conv_dense_idx = 0
        for layer in model.layers:
            if(isinstance(layer, Conv2D)):
                conv_dense_idx += 1
                model_c_file.write(constants.MODEL_C_CONV_TENSOR.format(
                    layer_idx=conv_dense_idx
                ))
            if(isinstance(layer, Dense)):
                conv_dense_idx += 1
                model_c_file.write(constants.MODEL_C_DENSE_TENSOR.format(
                    layer_idx=conv_dense_idx
                ))

            model_c_file.write('\n')

        model_c_file.write(constants.MODEL_C_IR_TENSORS)
        model_c_file.write('\n\n')

        model_c_file.write(constants.FUNC_PREPROCESS_INNER.format(
            project_name=constants.PROJECT_NAME,
            frac_bits=constants.FRAC_BITS,
            norm_value=int(constants.NORM_VALUE)
        ))
        model_c_file.write('\n')
        if constants.DEBUG_VERSION:
            model_c_file.write(constants.MODEL_C_FUNC_CHECK_RESULT)
            model_c_file.write('\n')

        #model_c_file.write(constants.MODEL_C_FUNC_RUN_INFERENCE.format(
        #    project_name=constants.PROJECT_NAME
        #))
        #model_c_file.write('\n')

        model_c_file.write(constants.MODEL_C_AUX_FUNC.format(
            project_name=constants.PROJECT_NAME
        ))
        model_c_file.write('\n')

        if constants.KERNEL_TYPE == 'fx8':
            model_c_file.write(constants.MODEL_C_FUNC_DEFS_8_GEN.format(
                #pool_dim1=pool_kernel,
                #pool_dim2=pool_kernel,
                #kernel_dim1=conv_kernel,
                #kernel_dim2=conv_kernel,
                #kernel_stride=conv_stride
            ))

        if constants.KERNEL_TYPE == 'fx16':
            model_c_file.write(constants.MODEL_C_FUNC_DEFS_16_GEN.format(
                #pool_dim1=pool_kernel,
                #pool_dim2=pool_kernel,
                #kernel_dim1=conv_kernel,
                #kernel_dim2=conv_kernel,
                #kernel_stride=conv_stride
            ))


        if constants.KERNEL_TYPE == 'fx8w16d':
            model_c_file.write(constants.MODEL_C_FUNC_DEFS_816_GEN.format(
                #pool_dim1=pool_kernel,
                #pool_dim2=pool_kernel,
                #kernel_dim1=conv_kernel,
                #kernel_dim2=conv_kernel,
                #kernel_stride=conv_stride
            ))

        model_c_file.write('\n\n')

        in_tensor = 'Y'
        out_tensor = 'X'
        conv_dense_idx = 0
        total_idx = 0

        if constants.DEBUG_VERSION:
            model_c_file.write(constants.MODEL_C_NETWORK_DBG.format(
                project_name=constants.PROJECT_NAME,
            ))
            model_c_file.write('\n')
            model_c_file.write(constants.MODEL_C_NETWORK_CYCLES_DBG)
            model_c_file.write('\n')
            model_c_file.write(constants.MODEL_C_NETWORK_PREPROCESS)
            model_c_file.write('\n')
            model_c_file.write(constants.MODEL_C_PERMUTE_DBG)
        else:
            model_c_file.write(constants.MODEL_C_NETWORK.format(
                project_name=constants.PROJECT_NAME,
            ))
            model_c_file.write(constants.MODEL_C_NETWORK_PREPROCESS)
            model_c_file.write('\n')
            model_c_file.write(constants.MODEL_C_PERMUTE)


        for layer in model.layers:
            total_idx += 1
            if(isinstance(layer, Conv2D)):
                conv_dense_idx += 1
                if constants.DEBUG_VERSION:
                    model_c_file.write(constants.MODEL_C_NETWORK_CONV_DBG.format(
                        out_tensor=out_tensor, in_tensor=in_tensor,
                        layer_idx=conv_dense_idx
                    ))
                else:
                    model_c_file.write(constants.MODEL_C_NETWORK_CONV.format(
                        out_tensor=out_tensor, in_tensor=in_tensor,
                        layer_idx=conv_dense_idx
                    ))
                model_c_file.write('\n')
                in_tensor, out_tensor = out_tensor, in_tensor

            if(isinstance(layer, Dense)):
                conv_dense_idx += 1
                if constants.DEBUG_VERSION:
                    model_c_file.write(constants.MODEL_C_NETWORK_DENSE_DBG.format(
                        out_tensor=out_tensor, in_tensor=in_tensor,
                        layer_idx=conv_dense_idx
                    ))
                else:
                    model_c_file.write(constants.MODEL_C_NETWORK_DENSE.format(
                        out_tensor=out_tensor, in_tensor=in_tensor,
                        layer_idx=conv_dense_idx
                    ))
                model_c_file.write('\n')
                in_tensor, out_tensor = out_tensor, in_tensor

            if(isinstance(layer, MaxPooling2D)):
                if constants.DEBUG_VERSION:
                    model_c_file.write(constants.MODEL_C_NETWORK_MAXPOOL_DBG.format(
                        out_tensor=out_tensor, in_tensor=in_tensor,
                        layer_idx=conv_dense_idx
                    ))
                else:
                    model_c_file.write(constants.MODEL_C_NETWORK_MAXPOOL.format(
                        out_tensor=out_tensor, in_tensor=in_tensor,
                        layer_idx=conv_dense_idx
                    ))
                model_c_file.write('\n')
                in_tensor, out_tensor = out_tensor, in_tensor

            if(isinstance(layer, Activation) and len(model.layers) - total_idx > 1
                    and isinstance(model.layers[total_idx], Dense)):
                if constants.DEBUG_VERSION:
                    model_c_file.write(constants.MODEL_C_NETWORK_RELU_DBG.format(
                        out_tensor=out_tensor, in_tensor=in_tensor,
                        layer_idx=conv_dense_idx
                    ))
                else:
                    model_c_file.write(constants.MODEL_C_NETWORK_RELU.format(
                        out_tensor=out_tensor, in_tensor=in_tensor,
                        layer_idx=conv_dense_idx
                    ))
                model_c_file.write('\n')
                in_tensor, out_tensor = out_tensor, in_tensor

            if(isinstance(layer, Activation) and len(model.layers) - total_idx <= 1):
                if constants.DEBUG_VERSION:
                    model_c_file.write(constants.MODEL_C_NETWORK_SOFTMAX_DBG.format(
                        in_tensor=in_tensor, layer_idx=conv_dense_idx
                    ))
                else:
                    model_c_file.write(constants.MODEL_C_NETWORK_SOFTMAX.format(
                        in_tensor=in_tensor, layer_idx=conv_dense_idx
                    ))
                model_c_file.write('\n')
                in_tensor, out_tensor = out_tensor, in_tensor

            if total_idx == len(model.layers):
                model_c_file.write(constants.MODEL_C_NETWORK_END)
                model_c_file.write('\n\n')


        model_c_file.write('\n')


@constants.execution_log_decorator
def write_ml_api_c():
    '''Create file with main function'''
    # path = Path(constants.PROJECT_NAME) / constants.ML_API_FILE_NAME
    path = Path('.') / constants.ML_API_FILE_NAME
    with path.open('w') as ml_api_file:
        ml_api_file.write(constants.ML_API_FILE_MAIN.format(
            project_name=constants.PROJECT_NAME,
            test_sample=constants.TEST_SAMPLE,
            frac_bits=constants.FRAC_BITS,
            norm_value=constants.NORM_VALUE
        ))

@constants.execution_log_decorator
def write_makefile():
    '''Create makefile for debug'''
    # path = Path(constants.PROJECT_NAME) / 'Makefile'
    path = Path('.') / 'Makefile'
    with path.open('w') as makefile:
        makefile.write(constants.MAKE_FILE_MAIN.replace('    ', '\t'))

@constants.execution_log_decorator
def write_test_input(model, test_data):
    '''Create file, containing single sample and its' class'''
    data = test_data[constants.TEST_SAMPLE]
    data_internal = (test_data[constants.TEST_SAMPLE] - constants.NORM_VALUE) / 2 \
                    ** constants.FRAC_BITS
    data_internal = np.expand_dims(data_internal, axis=0)
    pred = model.predict(data_internal)

    path = Path(constants.MODEL_FOLDER) / constants.TEST_FILE_NAME
    with path.open('w') as test_file:
        test_file.write(constants.TEST_FILE_DEFINES.format(
            project_name_caps=constants.PROJECT_NAME_CAPS,
            project_name=constants.PROJECT_NAME
        ))
        test_file.write("\n")
        test_file.write(constants.TEST_FILE_DEFINES_INPUT.format(
            test_sample=constants.TEST_SAMPLE,
            shape0=data.shape[0],
            shape1=data.shape[1],
            shape2=data.shape[2],
            rank=1
        ))
        test_file.write("\n")

        data = data.flatten()
        for i in range(len(data)):
            test_file.write(constants.TEST_FILE_INPUT_QUANT_WRAPPER.format(data[i]))
            if (i < len(data) - 1):
                test_file.write(",")
                if ((i + 1) % 10 == 0):
                    test_file.write("\\\n    ")
            else:
                test_file.write("\\\n};\n\n")

        test_file.write(constants.TEST_FILE_DEFINES_OUTPUT.format(
            test_sample=constants.TEST_SAMPLE,
            shape0=constants.NUM_CLASSES,
            rank=1
        ))
        test_file.write("\n")

        label = pred.flatten()
        for i in range(len(label)):
            test_file.write(constants.TEST_FILE_OUTPUT_QUANT_WRAPPER.format(label[i]))
            if (i < len(label) - 1):
                test_file.write(",")
                if ((i + 1) % 10 == 0):
                    test_file.write("\\\n    ")
            else:
                test_file.write("\\\n};\n\n")

        test_file.write(constants.TEST_FILE_FOOTER.format(constants.PROJECT_NAME_CAPS))


@constants.execution_log_decorator
def write_ir_output_tests(model, test_data):
    '''Create a set of output tensors of main layers for debug'''
    path = Path(constants.MODEL_FOLDER) / constants.IR_TEST_FOLDER
    if not path.exists():
        path.mkdir()

    ex_num = constants.TEST_SAMPLE
    test_data = (test_data - constants.NORM_VALUE) / 2 ** constants.FRAC_BITS

    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=[model.get_layer(layer.name).output for layer in model.layers])
    intermediate_output = intermediate_layer_model.predict(test_data[ex_num:ex_num+1])

    layer_num = 0
    next_layer = model.layers[1]
    for idx in range(len(intermediate_output)):
        layer = model.layers[idx]
        if idx != len(intermediate_output) - 1:
            next_layer = model.layers[idx + 1]
        else:
            next_layer = model.layers[idx]

        if isinstance(layer, Dense):
            layer_num += 1
            if isinstance(next_layer, BatchNormalization):
                idx2numpy.convert_to_file(str(path / ('ir_dense' + str(layer_num) + '.idx')), intermediate_output[idx + 1][0])
            else:
                idx2numpy.convert_to_file(str(path / ('ir_dense' + str(layer_num) + '.idx')), intermediate_output[idx][0])

        if isinstance(layer, MaxPooling2D):
            idx2numpy.convert_to_file(str(path / ('ir_pool' + str(layer_num) + '.idx')), intermediate_output[idx][0].transpose(2,0,1))

        if isinstance(layer, Activation):
            if len(intermediate_output[idx].shape) == 4:
                layer_num += 1
                idx2numpy.convert_to_file(str(path / ('ir_acti' + str(layer_num) + '.idx')), intermediate_output[idx][0].transpose(2,0,1))
            if len(intermediate_output[idx].shape) == 2: 
                idx2numpy.convert_to_file(str(path / ('ir_acti' + str(layer_num) + '.idx')), intermediate_output[idx][0])
    

@constants.execution_log_decorator
def write_test_dataset(test_data, test_labels, test_count):
    '''Create a subset of training data for testing with MLI'''
    path = Path(constants.MODEL_FOLDER) / constants.TEST_DATASET_FOLDER
    if not path.exists():
        path.mkdir()

    idx = np.random.permutation(len(test_labels))
    X_test_check, y_test_check = test_data[idx], test_labels[idx]

    idx2numpy.convert_to_file(str(path / 'tests.idx'), X_test_check[:test_count])
    idx2numpy.convert_to_file(str(path / 'labels.idx'), y_test_check[:test_count])

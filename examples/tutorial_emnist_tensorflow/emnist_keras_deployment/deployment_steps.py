from .file_writer import *
from . import constants
from pathlib import Path

def set_constants(prefix, num_classes, kernel_type, debug, norm_value, frac_bits, test_sample,
                  ir_test_folder, test_dataset_folder):
    """Set constants used in further deployment"""
    constants.create(prefix, num_classes, kernel_type, debug, norm_value, frac_bits,
                     test_sample, ir_test_folder, test_dataset_folder)

    print('Constants set to: ')
    print('\n')
    print('Prefix: ', prefix)
    print('Num classes: ', num_classes)
    print('Kernel type: ', kernel_type)
    print('Debug: ', debug)
    print('Offset: ', norm_value)
    print('Frac bits: ', frac_bits)
    print('Test sample: ', test_sample)
    print('IR folder: ', ir_test_folder)
    print('Test dataset folder: ', test_dataset_folder)
    print('\n')


def generate_model_files(model, model_params, model_params_float, to_wrap_coefficients, test_data):
    """Create a complete set of files, defining model"""
    path = Path('model')
    if not path.exists():
        path.mkdir()

    model_q_weights = {layer:params['weights'] for layer, params in model_params.items()}
    model_q_bias = {layer:params['bias'] for layer, params in model_params.items()}
    model_q_outputs = {layer:params['output'] for layer, params in model_params.items()}
    model_f_weights = {layer:params['weights'] for layer, params in model_params_float.items()}
    model_f_bias = {layer:params['bias'] for layer, params in model_params_float.items()}

    if to_wrap_coefficients:
        write_weights(model, model_f_weights,
                      model_f_bias, to_wrap_coefficients)
    else:
        write_weights(model, model_q_weights,
                      model_q_bias, to_wrap_coefficients)

    write_constants_h(model, model_q_weights,
                      model_q_bias, model_q_outputs)

    write_model_h(test_data.shape)
    write_model_c(model)

    write_ml_api_c()
    write_makefile()


def define_qmn(layers_repr):
    """Print table with Qm.n parameters of each layer"""
    str_template = '\t{}: Q{}.{} (range [{:.3f} : {:.3f}])'
    print('Kernel type: ', constants.KERNEL_TYPE)
    for name, layer in layers_repr:
        layer_params = layer.get_layer_data(constants.KERNEL_TYPE)
        print(name + ": ")
        for tensor_name in layer_params.keys():
            print(str_template.format(
                tensor_name,
                layer_params[tensor_name]['int_bits'],
                layer_params[tensor_name]['frac_bits'],
                layer_params[tensor_name]['min'],
                layer_params[tensor_name]['max']
            ))

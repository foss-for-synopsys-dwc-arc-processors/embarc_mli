# Copyright 2019-2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#
#
"""
Fixed point helper tools for embarc Machine Learning Inference Library (embarc MLI)
"""

import numpy as np


class TensorRepresentation:
    """
    Tensor representation class for quantization and statistic accumulation

    """
    def __init__(self):
        self.__data = None
        self.__max = np.finfo(np.float32).min
        self.__min = np.finfo(np.float32).max

    def set_data(self, data):
        """
        Assign particular data vector to current TensorRepresentation instance
        Accounts input data statistics in the instance.

        Parameters
        ----------
        data : iterable object which can be represented as numpy array.

        Returns
        -------
        self : TensorRepresentation object.
        """
        self.__data = np.asarray(data, dtype=np.float32)
        if data is not None:
            self.account(data)
        return self

    def account(self, data):
        """
        Update required instance statistics by input data vector

        Parameters
        ----------
        data :
            iterable object which can be represented as numpy array.

        Returns
        -------
        self : TensorRepresentation object.
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)
        self.__max = max(data.max(), self.__max)
        self.__min = min(data.min(), self.__min)
        return self

    def int_bits(self):
        """
        Return required number of integer bits in FX container according to accumulated statistics.

        Returns
        -------
        int_bits : integer
        """
        max_abs_val = max(abs(self.__max), abs(self.__min))
        return int(np.ceil(np.log2(max_abs_val)))

    def frac_bits(self, container_type):
        """
        Return required number of fractional bits in FX container of 'container_type'
        according to accumulated statistics.

        Parameters
        ----------
        container_type : numpy integer type, dtype, or instance
            Tensor container data type

        Returns
        -------
        frac_bits : integer
        """
        container_size = np.iinfo(container_type).bits - 1
        frac_bits = container_size - self.int_bits()
        if frac_bits < 0:
            print("WARNING: negative number({}) of frac_bits isn't supported by embarc_MLI."
                  "Saturate to 0".format(frac_bits))
        return max(0, frac_bits)

    def min(self):
        """
        Getter function for tensor's minimum value

        Returns
        -------
        min_val : integer
        """
        return self.__min

    def max(self):
        """
        Getter function for tensor's maximum value

        Returns
        -------
        min_val : integer
        """
        return self.__max

    def get_shape(self):
        """
        Getter function for tensor's shape

        Returns
        -------
        shape : tuple
            If tensor is asigned with particular data, returns it's shape. Else returns None
        """
        if self.__data is None:
            return None
        return self.__data.shape

    def get_data_float(self):
        """
        Returns copy of assigned tensor in source (float) form

        Returns
        -------
        data : numpy array
            If tensor is assigned with particular data, returns it's copy. Else returns None
        """
        if self.__data is None:
            return None
        return self.__data.copy()

    def get_data_quantazed(self, container_type, frac_bits=None):
        """
        Returns data of assigned tensor in quantized form for required container and number of fractiona bits

        Parameters
        ----------
        container_type : numpy integer type, dtype, or instance
            Tensor container data type
        frac_bits : integer or None
            Number of fractional bits of out FX data. If None is passed defines it according to self.frac_bits()

        Returns
        -------
        data : numpy array
            If tensor is assigned with particular data, returns it's quantized copy. Else returns None
        """
        if self.__data is None:
            return None
        if frac_bits is None:
            frac_bits = self.frac_bits(container_type)

        one = 2 ** frac_bits
        cont_info = np.iinfo(container_type)
        quantized_data = self.__data * one
        quantized_data = np.minimum(cont_info.max, np.maximum(cont_info.min, quantized_data))
        return quantized_data.round().astype(container_type)


class MacBasedKernelRepresentation:
    """
    MAC based MLI kernel representation (Conv2d or Fully connected)

    Next basic operation is implied: out = dot_product(in, weights) + bias.
    Instance keeps tensor representation of all operands (input, output, weights, bias)
    for further dependent data quantization.

    Attributes
    ----------
    input : TensorRepresentation
        Layer's input tensor representation. Is available as attribute for easy statistic accumulation.
    output : TensorRepresentation
        Layer's output tensor representation. Is available as attribute for easy statistic accumulation.

    Parameters
    ----------
    weights : list, tuple, numpy array
        Iterable object which can be represented as numpy array.
        Layer coefficients (weights) has been tuned on model training phase. Is a constant during inference time.
    bias : list, tuple, numpy array
        Iterable object which can be represented as numpy array.
        Constant offsets (bias) has been tuned on model training phase. Is a constant during inference time.
    """
    def __init__(self, weights, bias):
        self.__weights = TensorRepresentation().set_data(weights)
        self.__bias = TensorRepresentation().set_data(bias)
        self.input = TensorRepresentation()
        self.output = TensorRepresentation()

    def __quantize_layer_operands(self, params_type, var_type, accum_bits):
        """
        Constructor for dictionary of quantized  tensors and it's parameters.

        Defines Qmn data format for all operands taking into account accumulator limitations and containers size.
        Construct dictionary with quantized data and number of integer and fractional bits.

        Parameters
        ----------
        params_type : numpy integer type, dtype, or instance
            Data type of Weights and Bias tensors
        var_type : numpy integer type, dtype, or instance
            Data type of Input and Output tensors
        accum_bits : integer
            Accumulator size in bits (including sign bit)

        Returns
        -------
        quint_data_dict : dict
            Dictionary with quantized data and it's Qmn format and max/min range
        """
        # define how much bits should we keep unused in operands
        w_shape = self.__weights.get_shape()
        macs_per_out = int(np.prod(w_shape[1:])) + 1 if w_shape is not None else 1
        extra_bits_required = int(np.ceil(np.log2(macs_per_out)))
        acc_extra_bits = accum_bits - 1 - (np.iinfo(params_type).bits - 1 + np.iinfo(var_type).bits - 1)
        not_enough_bits = max(0, extra_bits_required - acc_extra_bits)
        weights_extra_bits = not_enough_bits // 2 if not_enough_bits != 0 else 0
        input_extra_bits = not_enough_bits - weights_extra_bits

        # Info preparation: Defining Qmn format of operands
        return_dict = {key: None for key in ['weights', 'bias', 'input', 'output']}
        return_dict['weights'] = {'int_bits': self.__weights.int_bits() + weights_extra_bits,
                                  'frac_bits': self.__weights.frac_bits(params_type) - weights_extra_bits,
                                  'min': self.__weights.min(), 'max': self.__weights.max()}
        return_dict['input'] = {'int_bits': self.input.int_bits() + input_extra_bits,
                                'frac_bits': self.input.frac_bits(var_type) - input_extra_bits,
                                'min': self.input.min(), 'max': self.input.max()}
        return_dict['bias'] = {'int_bits': self.__bias.int_bits(), 'frac_bits': self.__bias.frac_bits(params_type),
                               'min': self.__bias.min(), 'max': self.__bias.max()}
        return_dict['output'] = {'int_bits': self.output.int_bits(), 'frac_bits': self.output.frac_bits(var_type),
                                 'min': self.output.min(), 'max': self.output.max()}

        # bias frac_bits must be less or equal to sum of weights and input frac_bits
        mul_frac_bits = return_dict['weights']['frac_bits'] + return_dict['input']['frac_bits']
        return_dict['bias']['frac_bits'] = min(mul_frac_bits, return_dict['bias']['frac_bits'])

        # Quantize data according to defined Qmn format
        return_dict['bias']['data'] = self.__bias.get_data_quantazed(params_type, return_dict['bias']['frac_bits'])
        return_dict['input']['data'] = self.input.get_data_quantazed(var_type, return_dict['input']['frac_bits'])
        return_dict['output']['data'] = self.output.get_data_quantazed(var_type, return_dict['output']['frac_bits'])
        return_dict['weights']['data'] = self.__weights.get_data_quantazed(params_type,
                                                                           return_dict['weights']['frac_bits'])
        return return_dict

    def get_layer_data(self, kernel_type='float'):
        """
        Layer data quantization method.

        Returns dictionary with data and parameters for required kernel type.
        If kernel type is 'float', it returns ndarray for each assigned tensor in source (float) representation
        and the rest parameters EXCLUDING number of fractional bits.
        Dictionary structure:
        {
            <tensor_name>:
                {'data': <ndarray with quantized according to kernel_type data,
                            or original data if kernel_type == 'float'>,
                 'int_bits': <number of integer bits according to range>,
                 'frac_bits': <number of fractional bits according to range, or Null if kernel_type == 'float'>,
                 'max': <minimum value in tenor>,
                 'max': <maximum value in tenor>},
        }
        <tensor_name> is on of next entities: ('weights', 'bias', 'input', 'output')

        Parameters
        ----------
        kernel_type : str
            One of  MLI supported kernel types ('fx8', 'fx16', 'fx8w16d') or 'float'

        Raises
        ------
        KeyError
            If `kernel_type` is not in the set of supported kernels.

        Returns
        -------
        quint_data_dict : dict
            Dictionary with quantized data, it's Qmn format and max/min range.

        """
        if kernel_type not in ('float', 'fx8', 'fx16', 'fx8w16d'):
            raise KeyError("Unknown kernel type ({})".format(kernel_type))

        if kernel_type == 'fx8':
            return self.__quantize_layer_operands(np.int8, np.int8, 32)

        elif kernel_type == 'fx16':
            return self.__quantize_layer_operands(np.int16, np.int16, 40)

        elif kernel_type == 'fx8w16d':
            return self.__quantize_layer_operands(np.int8, np.int16, 32)

        elif kernel_type == 'float':
            return_dict = {
                'weights': {'data': self.__weights.get_data_float(), 'int_bits': self.__weights.int_bits(),
                            'min': self.__weights.min(), 'max': self.__weights.max()},
                'bias':    {'data': self.__bias.get_data_float(), 'int_bits': self.__bias.int_bits(),
                            'min': self.__bias.min(), 'max': self.__bias.max()},
                'input':   {'data': self.input.get_data_float(), 'int_bits': self.input.int_bits(),
                            'min': self.input.min(), 'max': self.input.max()},
                'output':   {'data': self.output.get_data_float(), 'int_bits': self.output.int_bits(),
                             'min': self.output.min(), 'max': self.output.max()}
            }
            for key in return_dict.keys():
                return_dict[key]['frac_bits'] = None
            return return_dict

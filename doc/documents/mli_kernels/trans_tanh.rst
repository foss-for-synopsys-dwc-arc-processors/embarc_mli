.. _tanh_prot:

TanH Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel performs hyperbolic tangent activation function on input tensor elementwise 
and stores the result to the output tensor.

.. math:: y_{i} = \frac{e^{x_{i}} - e^{{- x}_{i}}}{e^{x_{i}} + e^{{- x}_{i}}}

Where:

   :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input tensor*

   :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output tensor*

This kernel outputs a tensor of the same shape and type as the input. This kernel supports 
in-place computation: output and input can point to exactly the same memory (the same 
starting address and memory strides). 

If the starting address and memory stride of the input and output tensors are set in such 
a way that memory regions are overlapped, the behavior is undefined.

Kernels which implement TanH functions have the following prototype:

.. code::

   mli_status mli_krn_tanh_<data_format>(
      const mli_tensor *in,
      mli_tensor *out);
	  
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function 
parameters are shown in the following table:

.. table:: TanH Function Parameters
   :align: center
   :widths: auto
   
   +----------------+--------------------+--------------------------------------------+
   | **Parameter**  | **Type**           | **Description**                            |
   +================+====================+============================================+
   | ``in``         | ``mli_tensor *``   | [IN] Pointer to constant input tensor.     |
   +----------------+--------------------+--------------------------------------------+
   | ``out``        | ``mli_tensor *``   | [OUT] Pointer to output tensor.            |
   |                |                    | Result is stored here.                     |
   +----------------+--------------------+--------------------------------------------+
..

.. table:: List of Available TanH Functions
   :align: center
   :widths: auto
   
   +------------------------+------------------------------------+
   | **Function Name**      | **Details**                        |
   +========================+====================================+
   | ``mli_krn_tanh_sa8``   | All tensors data format: **sa8**   |
   +------------------------+------------------------------------+
   | ``mli_krn_tanh_fx16``  | All tensors data format: **fx16**  |
   +------------------------+------------------------------------+
..

Ensure that you satisfy the following conditions before calling the function:

 - ``in`` tensor must be valid.
 
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity 
   (that is, the total amount of elements in input tensor). Other fields are filled 
   by kernel (shape, rank and element specific parameters).

For **sa8** versions of kernel, in addition to the preceding conditions, ensure that you 
satisfy the following conditions before calling the function: 

 - ``in`` tensor must be quantized on the tensor level. It implies that the tensor 
   contains a single scale factor and a single zero offset.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

The range of this function is (-1, 1).  Depending on the data type, quantization parameters of the output 
tensor are configured in the following way:

 - **fx16**

    - ``out.el_params.fx.frac_bits`` is set to 15. Hence, the maximum representable value of sigmoid is
      equivalent to 0.999969482421875 (not 1.0).

 - **sa8**

    - ``out.el_params.sa.zero_point.mem.i16`` is set to 0

    - ``out.el_params.sa.scale.mem.i16`` is set to 1

    - ``out.el_params.sa.scale_frac_bits.mem.i8`` is set to 8
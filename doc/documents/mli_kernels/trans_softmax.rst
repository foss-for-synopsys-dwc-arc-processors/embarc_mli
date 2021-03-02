.. _softmax_prot:

Softmax Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel performs Softmax activation function that is a generalization of the 
logistic function that transforms input vector according to the following formula:

.. math:: y_{i} = \frac{e^{x_{i}}}{\sum_{j}^{}e^{x_{j}}}

Where:

   :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input data subset*

   :math:`x_{j}` *–* :math:`j_{\text{th}}` *value in the same input data
   subset*

   :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output data subset*
	
Softmax function might be applied to the whole tensor, or along a specific axis. 
In the first case, all the input values are involved in the calculation of each output value. 
If an axis is specified, then the softmax function is applied to each slice along the 
specific axis independently. 

This kernel outputs tensor of the same shape and type as input. This kernel supports
in-place computation: output and input can point to exactly the same memory (the same 
starting address and memory strides). If the starting address and memory stride of the 
input and output tensors are set in such a way that memory regions are overlapped, 
the behavior is undefined.
 
Kernels which implement SoftMax functions have the following prototype:

.. code::

   mli_status mli_krn_softmax_<data_format>(
      const mli_tensor *in,
      const mli_softmax_cfg *cfg,
     mli_tensor *out);
..
	 
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function 
parameters are shown in the following table:

.. table:: Softmax ReLU Function Parameters
   :align: center
   :widths: auto
   
   +----------------+-------------------------+-----------------------------------------------+
   | **Parameter**  | **Type**                | **Description**                               |
   +================+=========================+===============================================+
   | ``in``         | ``mli_tensor *``        | [IN] Pointer to constant input tensor.        |
   +----------------+-------------------------+-----------------------------------------------+
   | ``cfg``        | ``mli_softmax_cfg *``   | [IN] Pointer to softmax parameters structure. |
   +----------------+-------------------------+-----------------------------------------------+
   | ``out``        | ``mli_tensor *``        | [OUT] Pointer to output tensor.               |
   |                |                         | Result is stored here                         |
   +----------------+-------------------------+-----------------------------------------------+
..

``mli_softmax_cfg`` is defined as:

.. code::

   typedef mli_prelu_cfg mli_softmax_cfg;
..
  
See Table :ref:`t_mli_prelu_cfg_desc` for more details.

.. table:: List of Available Softmax Functions
   :align: center
   :widths: auto
   
   +---------------------------+------------------------------------+
   | **Function Name**         | **Details**                        |
   +===========================+====================================+
   | ``mli_krn_softmax_sa8``   | All tensors data format: **sa8**   |
   +---------------------------+------------------------------------+
   | ``mli_krn_softmax_fx16``  | All tensors data format: **fx16**  |
   +---------------------------+------------------------------------+
..

Ensure that you satisfy the following conditions before calling the function:

 - ``in`` tensor must be valid.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity 
   (that is, the total amount of elements in input tensor). Other fields are filled 
   by kernel (shape, rank and element specific parameters).
   
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.
 
 - axis parameter might be negative and must be less than in tensor rank.
 

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

The range of this function is (0, 1).  Depending on the data type, quantization parameters of the output 
tensor are configured in the following way:

 - **fx16**

    - ``out.el_params.fx.frac_bits`` is set to 15. Hence, the maximum representable value of sigmoid is
      equivalent to 0.999969482421875 (not 1.0).

 - **sa8**

    - ``out.el_params.sa.zero_point.mem.i16`` is set to -128

    - ``out.el_params.sa.scale.mem.i16`` is set to 1

    - ``out.el_params.sa.scale_frac_bits.mem.i8`` is set to 8
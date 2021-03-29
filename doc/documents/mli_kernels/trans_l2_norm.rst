.. _l2_norm_prot:

L2 Normalization Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel normalizes data across the specified dimension using L2 norm according to the following 
formula:

.. math:: y_{i} = \frac{x_{i}}{\sqrt{epsilon + \sum_{j}{x_{j}}^{2}}}

Where:

   :math:`x_{i}-i_{th}` *-* value in input data subset

   :math:`x_{j}-j_{th}` *-* value in the same input data subset

   :math:`y_{i}-i_{th}` *-* value in output data subset

   :math:`epsilon` *-* lower bound to prevent division on zero

L2 normalization function might be applied to the whole tensor, or along a specific axis. In the 
first case all input values are involved in the calculation of each output value. If the axis is 
specified, then the function is applied to each slice along the specific axis independently. 

This kernel outputs a tensor of the same shape and type as the input. This kernel supports in-place 
computation: output and input can point to exactly the same memory (the same starting address
and memory strides). 

.. note::

   Only an exact overlap of starting address and memory stride of the input and output 
   tensors is acceptable. Partial overlaps result in undefined behavior.
..

Kernels which implement L2 normalization functions have the following prototype:

.. code:: c

   mli_status mli_krn_L2_normalize_<data_format>(
      const mli_tensor *in,
      const mli_tensor *epsilon,
      const mli_softmax_cfg *cfg,
      mli_tensor *out);
	  
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function 
parameters are shown in the following table:

.. table:: L2 Normalization Function Parameters
   :align: center
   :widths: auto
   
   +----------------+------------------------------+--------------------------------------------------------+
   | **Parameter**  | **Type**                     | **Description**                                        |
   +================+==============================+========================================================+
   | ``in``         | ``mli_tensor *``             | [IN] Pointer to constant input tensor.                 |
   +----------------+------------------------------+--------------------------------------------------------+
   | ``epsilon``    | ``mli_tensor *``             | [IN] Pointer to tensor with epsilon value.             |
   +----------------+------------------------------+--------------------------------------------------------+
   | ``cfg``        | ``mli_L2_normalize_cfg *``   | [IN] Pointer to L2 Normalize parameters structure.     |
   +----------------+------------------------------+--------------------------------------------------------+
   | ``out``        | ``mli_tensor *``             | [OUT] Pointer to output tensor. Result is stored here. |
   +----------------+------------------------------+--------------------------------------------------------+
..

``mli_L2_normalize_cfg`` is defined as:

.. code:: c

   typedef mli_prelu_cfg mli_L2_normalize_cfg;
..

See Table :ref:`t_mli_prelu_cfg_desc` for more details.

.. table:: List of Available L2 Normalization Functions
   :align: center
   :widths: auto
   
   +--------------------------+-----------------------------------+
   | **Function Name**        | **Details**                       |
   +==========================+===================================+
   | ``mli_krn_L2_norm_sa8``  | All tensors data format: **sa8**  |
   +--------------------------+-----------------------------------+
   | ``mli_krn_L2_norm_fx16`` | All tensors data format: **fx16** |
   +--------------------------+-----------------------------------+
..

Ensure that you satisfy the following conditions before calling the function:

 - ``in`` and ``epsilon`` tensors must be valid.
 
 - ``epsilon`` tensor must be a valid tensor-scalar (see data field 
   description in the Table :ref:`mli_tnsr_struc`).
   
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient 
   capacity (that is, the total amount of elements in input tensor). Other 
   fields are filled by kernel (shape, rank and element specific parameters).

 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the 
   tensors.

 - ``axis`` parameter might be negative and must be less than ``in`` tensor rank.

For **sa8** versions of the kernel, in addition to the preceding conditions, ensure that you 
satisfy the following conditions before calling the function: 

 - ``in`` and ``epsilon`` tensors must be quantized on the tensor level. This 
   implies that the tensor contains a single scale factor and a single zero offset.

The range of this function is (-1, 1).  Depending on the data type, quantization parameters of the output 
tensor are configured in the following way:

 - **fx16**

    - ``out.el_params.fx.frac_bits`` is set to 15. Hence, the maximum representable value of sigmoid is
      equivalent to 0.999969482421875 (not 1.0).

 - **sa8**

    - ``out.el_params.sa.zero_point.mem.i16`` is set to 0

    - ``out.el_params.sa.scale.mem.i16`` is set to 1

    - ``out.el_params.sa.scale_frac_bits.mem.i8`` is set to 7

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.	
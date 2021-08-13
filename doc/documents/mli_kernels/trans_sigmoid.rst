.. _sigmoid_prot:

Sigmoid Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^

This kernel performs sigmoid (also called as logistic) activation function on input tensor 
element-wise and stores the result to the output tensor.

.. math:: y_{i} = \frac{1}{1 + e^{{- x}_{i}}}

Where:

   :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input tensor*

   :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output tensor*


This kernel uses a look-up table (LUTs) to perform data transformation. 
See :ref:`lut_prot` section and the pseudo-code sample for more details on LUT structure preparation.
Use the following functions for the purpose:

 - :code:`mli_krn_sigm_get_lut_size`
 - :code:`mli_krn_sigm_create_lut`

Functions
^^^^^^^^^

Kernels which implement Sigmoid functions have the following prototype:

.. code:: c

   mli_status mli_krn_sigm_<data_format>(
      const mli_tensor  *in,
      const mli_lut *lut,
      mli_tensor  *out);
..
  
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function 
parameters are shown in the following table:

.. table:: Sigmoid Function Parameters
   :align: center
   :widths: auto
   
   +----------------+----------------------+----------------------------------------------+
   | **Parameter**  | **Type**             | **Description**                              |
   +================+======================+==============================================+
   | ``in``         | ``mli_tensor *``     | [IN] Pointer to constant input tensor.       |
   +----------------+----------------------+----------------------------------------------+
   | ``lut``        | ``mli_lut *``        | [IN] Pointer to a valid LUT table            |
   |                |                      | structure prepared for sigmoid  activation.  |
   +----------------+----------------------+----------------------------------------------+
   | ``out``        | ``mli_tensor *``     | [IN | OUT] Pointer to output tensor.         |
   |                |                      | Result is stored here                        |
   +----------------+----------------------+----------------------------------------------+
..

.. table:: List of Available Sigmoid Functions
   :align: center
   :widths: auto
   
   +------------------------+------------------------------------+
   | **Function Name**      | **Details**                        |
   +========================+====================================+
   | ``mli_krn_sigm_sa8``   | All tensors data format: **sa8**   |
   +------------------------+------------------------------------+
   | ``mli_krn_sigm_fx16``  | All tensors data format: **fx16**  |
   +------------------------+------------------------------------+
..

Conditions
^^^^^^^^^^

Ensure that you satisfy the following general conditions before calling the function:

 - ``in`` and ``out`` tensors must be valid (see :ref:`mli_tnsr_struc`)
   and satisfy data requirements of the selected version of the kernel.

 - ``in`` and ``out`` tensors must be of the same shape

 - ``lut`` structure must be valid and prepared for the sigmoid activation function (see :ref:`lut_prot`).

 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.

For **sa8** versions of kernel, in addition to general conditions, ensure that you satisfy 
the following quantization conditions before calling the function:

 - ``in`` tensor must be quantized on the tensor level. This implies that the tensor contains 
   a single scale factor and a single zero offset.

 - Zero offset of ``in`` tensor must be within [-128, 127] range.
   

Result
^^^^^^

These functions modify:

 - Memory pointed by ``out.data.mem`` field.  
 - ``el_params`` field of ``out`` tensor. 

It is assumed that all the other fields and structures are properly populated 
to be used in calculations and are not modified by the kernel.

The range of this function is (0, 1).  Depending on the data type, quantization parameters of the output 
tensor are configured in the following way:

 - **fx16**

    - ``out.el_params.fx.frac_bits`` is set to 15. Hence, the maximum representable value of sigmoid is
      equivalent to 0.999969482421875 (not 1.0).

 - **sa8**

    - ``out.el_params.sa.zero_point.mem.i16`` is set to -128

    - ``out.el_params.sa.scale.mem.i16`` is set to 1

    - ``out.el_params.sa.scale_frac_bits.mem.i8`` is set to 8

The kernel supports in-place computation. It means that ``out`` and ``in`` tensor structures 
can point to the same memory with the same memory strides but without shift.
It can affect performance for some platforms.

.. warning::

  Only an exact overlap of starting address and memory stride of the ``in`` and ``out`` 
  tensors is acceptable. Partial overlaps result in undefined behavior.
..

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

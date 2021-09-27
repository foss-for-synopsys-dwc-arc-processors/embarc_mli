.. _leaky_relu_prot:

Leaky ReLU Prototype and Function List
--------------------------------------

Description
^^^^^^^^^^^

This kernel performs Rectified Linear Unit (ReLU) with a negative slope activation function. 
It transforms each element of input tensor according to the following formula:

.. math::

   y_{i} =  \Big\{ {\begin{matrix}
   x_{i}\text{ if }x_{i} \geq 0 \\
   {\alpha}*x_{i}\text{ if }x_{i} < 0 \\
   \end{matrix}} 
..

Where:

   :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input tensor*

   :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output tensor*

   :math:`\alpha` - coefficient of the negative slope


Functions
^^^^^^^^^

Kernels which implement Leaky ReLU functions have the following prototype:

.. code:: c

   mli_status mli_krn_leaky_relu_<data_format>(
      const mli_tensor *in,
      const mli_tensor *slope_coeff,
      mli_tensor *out);
..
   
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the 
function parameters are shown in the following table:

.. _t_tfm_data_conv:
.. table:: Leaky ReLU Function Parameters
   :align: center
   :widths: auto
   
   +------------------+----------------------+----------------------------------------------+
   | **Parameter**    | **Type**             | **Description**                              |
   +==================+======================+==============================================+
   | ``in``           | ``mli_tensor *``     | [IN] Pointer to constant input tensor.       |
   +------------------+----------------------+----------------------------------------------+
   | ``slope_coeff``  | ``mli_tensor *``     | [IN] Pointer to tensor-scalar with negative  |
   |                  |                      | slope coefficient.                           |
   +------------------+----------------------+----------------------------------------------+
   | ``out``          | ``mli_tensor *``     | [IN | OUT] Pointer to output tensor.         |
   |                  |                      | Result is stored here                        |
   +------------------+----------------------+----------------------------------------------+
..

.. table:: List of Available Leaky ReLU Functions
   :align: center
   :widths: auto 
   
   +------------------------------+------------------------------------+
   | **Function Name**            | **Details**                        |
   +==============================+====================================+
   | ``mli_krn_leaky_relu_sa8``   | All tensors data format: **sa8**   |
   +------------------------------+------------------------------------+
   | ``mli_krn_leaky_relu_fx16``  | All tensors data format: **fx16**  |
   +------------------------------+------------------------------------+
..

Conditions
^^^^^^^^^^

Ensure that you satisfy the following general conditions before calling the function:

 - ``in`` and ``out`` tensors must be valid (see :ref:`mli_tnsr_struc`)
   and satisfy data requirements of the selected version of the kernel.

 - ``slope_coeff`` tensor must be a valid tensor-scalar (see data field 
   description in the Table :ref:`mli_tnsr_struc`).

 - ``in`` and ``out`` tensors must be of the same shapes

 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.

For **fx16** versions of kernel, in addition to general conditions, ensure that you satisfy 
the following quantization conditions before calling the function:

 - The number of ``frac_bits`` in the ``in`` and ``out`` tensors must be equal. 

For **sa8** versions of kernel, in addition to general conditions, ensure that you satisfy 
the following quantization conditions before calling the function:

 - ``in`` and ``out`` tensor must be quantized on the tensor level. This implies 
   that each tensor contains a single scale factor and a single zero offset.

 - Zero offset of ``in`` and ``out`` tensors must be within [-128, 127] range.

 - Zero offset of ``slope_coeffs`` tensor must be within [-16384, 16383] range.

Ensure that you satisfy the platform-specific conditions in addition to those listed above 
(see the :ref:`platform_spec_chptr` chapter).

Result
^^^^^^

These functions only modify the memory pointed by ``out.data.mem`` field. 
It is assumed that all the other fields of ``out`` tensor are properly populated 
to be used in calculations and are not modified by the kernel.

The kernel supports in-place computation. It means that ``out`` and ``in`` tensor structures 
can point to the same memory with the same memory strides but without shift.
It can affect performance for some platforms.

.. warning::

  Only an exact overlap of starting address and memory stride of the ``in`` and ``out`` 
  tensors is acceptable. Partial overlaps result in undefined behavior.
..

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

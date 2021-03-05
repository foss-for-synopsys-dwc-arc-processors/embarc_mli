.. _leaky_relu_prot:

Leaky ReLU Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel performs Rectified Linear unit (ReLU) with a negative slope activation function. 
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

This kernel outputs tensor of the same shape and type as input. This kernel supports in-place 
computation: output and input can point to exactly the same memory (the same starting address
and memory strides). If the starting address and memory stride of the 
input and output tensors are set in such a way that memory regions are overlapped, 
the behavior is undefined.

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
   | ``out``          | ``mli_tensor *``     | [OUT] Pointer to output tensor. Result is    |
   |                  |                      | stored here.                                 |
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

Ensure that you satisfy the following conditions before calling the function:

 - ``in`` and ``slope_coeff`` tensors must be valid.
 
 - ``slope_coeff`` tensor must be a valid tensor-scalar (see data field description in the 
   Table :ref:`t_tfm_data_conv`).
 
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity (that is, 
   the total amount of elements in input tensor). Other fields are filled by kernel (shape, 
   rank and element specific parameters).
   
For **sa8** versions of kernel, in addition to the preceding conditions, ensure that you 
satisfy the following conditions before calling the function: 

 - ``in`` tensor must be quantized on the tensor level. It implies that the tensor contains a 
   single scale factor and a single zero offset.
   
Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

.. _param_relu_prot:

Parametric ReLU (PReLU) Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel performs Parametric Rectified Linear unit (PReLU) with a negative slope activation 
function. It transforms each element of input tensor according to the following formula:

.. math::

   y_{i} = \Big\{ { \begin{matrix}
   x_{i}\text{ if }x_{i} \geq 0 \\
   {\alpha}*x_{i}\text{ if }x_{i} < 0 \\
   \end{matrix}} 

Where:

    :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input data subset*

    :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output data subset*

    :math:`\alpha` - coefficient of the negative slope for the specific
    data subset
	
While for Leaky ReLU, the whole tensor shares only the :math:`\alpha` coefficient, for PRelu an 
array of slope coefficients is shared across an axis.  Hence, for each slice along the 
specified axis an individual :math:`\alpha` slope coefficient is used. 

The “shared axis” feature found in some frameworks is not supported in MLI. This functionality can 
instead be achieved in several iterations using the PReLU kernel and the mem_strides feature. 
One iteration implies creating subtensors from input and alpha tensors using memstrides and applying 
the PReLU kernel on them.

This kernel outputs tensor of the same shape and type as input. This kernel can perform in-place 
computation: output and input can point to exactly the same memory (the same starting address
and memory strides). 

If the starting address and memory stride of the input and output tensors are set in such a way 
that memory regions are overlapped, the behavior is undefined.

Kernels which implement Leaky ReLU functions have the following prototype:

.. code:: c

   mli_status mli_krn_leaky_relu_<data_format>(
      const mli_tensor  *in,
      const mli_tensor  *slope_coeffs,
      const mli_prelu_cfg  *cfg,
      mli_tensor  *out);

where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function parameters 
are shown in the following table:

.. table:: Parametric ReLU Function Parameters
   :align: center
   :widths: auto
   
   +------------------+-----------------------+-----------------------------------------------------------+
   | **Parameter**    | **Type**              | **Description**                                           |
   +==================+=======================+===========================================================+
   | ``in``           | ``mli_tensor *``      | [IN] Pointer to constant input tensor.                    |
   +------------------+-----------------------+-----------------------------------------------------------+
   | ``slope_coeff``  | ``mli_tensor *``      | [IN] Pointer to tensor with negative slope coefficients.  |
   +------------------+-----------------------+-----------------------------------------------------------+
   | ``cfg``          | ``mli_prelu_cfg *``   | [IN] Pointer to PReLU parameters structure.               |
   +------------------+-----------------------+-----------------------------------------------------------+
   | ``out``          | ``mli_tensor *``      | [OUT] Pointer to output tensor. Result is stored here.    |
   +------------------+-----------------------+-----------------------------------------------------------+
..

``mli_prelu_cfg`` is defined as:

.. code:: c

   typedef struct {
       int32_t axis;
   } mli_prelu_cfg;
..

.. _t_mli_prelu_cfg_desc:
.. table:: mli_prelu_cfg Structure Field Description
   :align: center
   :widths: auto
   
   +-----------------+----------------+--------------------------------------------------------------+
   |                 |                |                                                              |
   | **Field Name**  | **Type**       | **Description**                                              |
   +=================+================+==============================================================+
   |                 |                | An axis along which the function is computed. Axis           |
   |                 |                | corresponds to index of tensor’s dimension starting from 0.  |
   | ``axis``        | ``int32_t``    | For instance, having feature map in HWC layout, axis == 0    |
   |                 |                | corresponds to H dimension. If axis < 0, the function is     |
   |                 |                | applied to the whole tensor.                                 |
   +-----------------+----------------+--------------------------------------------------------------+
..

.. table:: List of Available PReLU Functions
   :align: center
   :widths: auto
   
   +-------------------------+------------------------------------+
   | **Function Name**       | **Details**                        |
   +=========================+====================================+
   | ``mli_krn_prelu_sa8``   | All tensors data format: **sa8**   |
   +-------------------------+------------------------------------+
   | ``mli_krn_prelu_fx16``  | All tensors data format: **fx16**  |
   +-------------------------+------------------------------------+
..

Ensure that you satisfy the following conditions before calling the function:

 - ``in`` and ``slope_coeff`` tensors must be valid.
 
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity 
   (that is, the total amount of elements in input tensor). Other fields are filled by 
   kernel (shape, rank and element specific parameters).
   
For **sa8** versions of kernel, in addition to the preceding conditions, ensure that you 
satisfy the following conditions before calling the function: 

 - ``in`` ``out`` and ``slope_coeff`` tensors must be quantized on the tensor level. It implies 
   that the tensor contains a single scale factor and a single zero offset.
   
Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

.. _param_relu_prot:

Parametric ReLU (PReLU) Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^

This kernel performs Parametric Rectified Linear Unit (PReLU) with a negative slope activation 
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
One iteration implies creating subtensors from ``in`` and ``slope_coeff`` tensors using memstrides and applying 
the PReLU kernel on them.

Functions
^^^^^^^^^

Kernels which implement Parametric ReLU functions have the following prototype:

.. code:: c

   mli_status mli_krn_prelu_<data_format>(
      const mli_tensor  *in,
      const mli_tensor  *slope_coeff,
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
   | ``out``          | ``mli_tensor *``      | [IN | OUT] Pointer to output tensor.                      |
   |                  |                       | Result is stored here                                     |
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

Conditions
^^^^^^^^^^

Ensure that you satisfy the following general conditions before calling the function:

 - ``in``, ``out`` and ``slope_coeff`` tensors must be valid (see :ref:`mli_tnsr_struc`).

 - ``in`` and ``out`` tensors must be of the same shape.

 - ``slope_coeff`` tensor must satisfy the following shape requirements depending
   on the ``axis`` parameter of ``cfg`` structure:
   
    - ``axis < 0`` : ``slope_coeff`` tensor must be a valid tensor-scalar (see data field 
      description in the Table :ref:`mli_tnsr_struc`).

    - ``axis >= 0`` : ``slope_coeff`` is a one-dimensional tensor (rank==1). 
      Its length must be equal to ``axis`` dimension of ``in`` tensor (e.g. ``in.shape[cfg.axis]``).

 - ``axis`` parameter of ``cfg`` structure can be negative and must be less than ``in`` tensor rank.
 
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.

For **fx16** versions of kernel, in addition to general conditions, ensure that you satisfy 
the following quantization conditions before calling the function:

 - The number of ``frac_bits`` in the ``in`` and ``out`` tensors must be equal. 

For **sa8** versions of kernel, in addition to general conditions, ensure that you satisfy 
the following quantization conditions before calling the function:

 - ``in``, ``out`` and ``slope_coeff`` tensors must be quantized on the tensor level. This implies 
   that the tensor contains a single scale factor and a single zero offset.

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

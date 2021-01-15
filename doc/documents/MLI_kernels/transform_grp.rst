Transform (Activation) Group
-----------------------------

The Transform Group describes operations for transforming data according to 
particular functions (or activations).  In general, transformations do not 
alter data size or shape and only change the values element-wise.

 - :ref:`relu_prot`

 - :ref:`leaky_relu_prot`
 
 - :ref:`param_relu_prot` 
 
 - :ref:`sigmoid_prot`

 - :ref:`tanh_prot`
 
 - :ref:`softmax_prot`
 
 - :ref:`l2_norm_prot` 

 
.. _relu_prot:

ReLU Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel represents Rectified Linear unit (ReLU). It performs various types 
of the rectifier activation on input. The following types of ReLU supported by 
this type of kernel:

-  *General ReLU:* :math:`y_{i} = MAX(x_{i},\ 0)`

-  *ReLU1:* :math:`y_{i} = MAX(MIN\left( x_{i},1 \right),\  - 1)`

-  *ReLU6:* :math:`y_{i} = MAX(MIN\left( x_{i},6 \right),\ 0)`

   Where:

-  :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input tensor*

-  :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output tensor*

This kernel outputs tensor of the same shape and type as input. This kernel performs 
in-place computation: output and input can point to exactly the same memory (the same 
starting address).

Kernels which implement ReLU functions have the following prototype:

.. code::

   mli_status mli_krn_relu_<data_format>(
      const mli_tensor *in,
      const mli_relu_cfg *cfg,
      mli_tensor *out);
..

where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function 
parameters are shown in the following table:

.. table:: Data Format Naming Convention Fields
   :align: center
   :widths: auto 
   
   +----------------+----------------------+----------------------------------------------------------+
   | **Parameter**  | **Type**             | **Description**                                          |
   +================+======================+==========================================================+
   | ``in``         | ``mli_tensor *``     | [IN] Pointer to constant input tensor.                   |
   +----------------+----------------------+----------------------------------------------------------+
   | ``cfg``        | ``mli_relu_cfg *``   | [IN] Pointer to relu parameters structure.               |
   +----------------+----------------------+----------------------------------------------------------+
   | ``out``        | ``mli_tensor *``     | [OUT] Pointer to output tensor. Result is stored here.   |
   +----------------+----------------------+----------------------------------------------------------+
..

   ``mli_relu_cfg`` is defined as:

.. code::
   
   typedef struct {
      mli_relu_type type;
    } mli_relu_cfg;
..

.. _t_mli_relu_cfg_desc:
.. table:: mli_relu_cfg Structure Field Description
   :align: center
   :widths: auto 
   
   +-----------------+--------------------+------------------------+-------------------------------------------------------+
   | **Field Name**  | **Type**           | **Enumeration Value**  | **Description**                                       |
   +=================+====================+========================+=======================================================+
   |                 |                    | ``MLI_RELU_NONE``      | No ReLU. Identity function                            |
   |                 |                    +------------------------+-------------------------------------------------------+
   |                 |                    | ``MLI_RELU_GEN``       | General Rectifier function with output range from 0   |
   |                 | ``mli_relu_type``  |                        | to value maximum inclusively                          |
   | ``type``        | (enumeration)      +------------------------+-------------------------------------------------------+
   |                 |                    | ``MLI_RELU_1``         | ReLU1 Rectifier function with output range [-1, 1]    |
   |                 |                    +------------------------+-------------------------------------------------------+
   |                 |                    | ``MLI_RELU_6``         | ReLU6 Rectifier function with output range [0, 6]     |
   +-----------------+--------------------+------------------------+-------------------------------------------------------+
..


.. table:: List of Available ReLU Functions
   :align: center
   :widths: auto 
   
   +------------------------+-----------------------------------+
   | **Function Name**      | **Details**                       |
   +========================+===================================+
   | ``mli_krn_relu_sa8``   | All tensors data format: **sa8**  |
   +------------------------+-----------------------------------+
   | ``mli_krn_relu_fx16``  | All tensors data format: **fx16** |
   +------------------------+-----------------------------------+
..

All the listed functions must comply to the following conditions:

 - ``in`` tensor must be valid.
 
 - ``mem_stride`` of the innermost dimension should be equal to 1 for all the tensors.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity 
   (that is, the total amount of elements in input tensor). Other fields are filled 
   by kernel (shape, rank and element specific parameters).

For **sa8** versions of kernel, in addition to the preceding conditions: 

 - ``in`` tensor must be quantized on the tensor level. It implies that the tensor 
   contains a single scale factor and a single zero offset.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and return the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

.. _leaky_relu_prot:

Leaky ReLU Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel performs Rectified Linear unit (ReLU) with a negative slope activation function. 
It transforms each element of input tensor according to next formula:

.. math::

   y_{i} =  \Big\{ {\begin{matrix}
   x_{i}\text{ if }x_{i} \geq 0 \\
   {\alpha}*x_{i}\text{ if }x_{i} < 0 \\
   \end{matrix}} 
..

Where:

-  :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input tensor*

-  :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output tensor*

-  :math:`\alpha` - coefficient of the negative slope

This kernel outputs tensor of the same shape and type as input. This kernel performs in-place 
computation: output and input can point to exactly the same memory (the same starting address).

Kernels which implement Leaky ReLU functions have the following prototype:

.. code::

   mli_status mli_krn_leaky_relu_<data_format>(
      const mli_tensor \*in,
      const mli_tensor \*slope_coeff,
      mli_tensor \*out);
..
   
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the 
function parameters are shown in the following table:

.. table:: Data Format Naming Convention Fields
   :align: center
   :widths: auto
   
   +------------------+----------------------+----------------------------------------------+
   | **Parameter**    | **Type**             | **Description**                              |
   +==================+======================+==============================================+
   | ``in``           | ``mli_tensor \*``    | [IN] Pointer to constant input tensor.       |
   +------------------+----------------------+----------------------------------------------+
   | ``slope_coeff``  | ``mli_tensor \*``    | [IN] Pointer to tensor-scalar with negative  |
   |                  |                      | slope coefficient.                           |
   +------------------+----------------------+----------------------------------------------+
   | ``out``          | ``mli_tensor \*``    | [OUT] Pointer to output tensor. Result is    |
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

All the listed functions must comply to the following conditions:

 - ``in`` and ``slope_coeff`` tensors must be valid.
 
 - ``slope_coeff`` tensor must be a valid tensor-scalar (see data field description in the Table 6).
 
 - ``mem_stride`` of the innermost dimension should be equal to 1 for all the tensors.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity (that is, 
   the total amount of elements in input tensor). Other fields are filled by kernel (shape, 
   rank and element specific parameters).
   
For **sa8** versions of kernel, in addition to the preceding conditions: 

 - ``in`` tensor must be quantized on the tensor level. It implies that the tensor contains a 
   single scale factor and a single zero offset.
   
Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and return the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

.. _param_relu_prot:

Parametric ReLU (PReLU) Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel performs Parametric Rectified Linear unit (PReLU) with a negative slope activation 
function. It transforms each element of input tensor according to next formula:

.. math::

   y_{i} = \Big\{ { \begin{matrix}
   x_{i}\text{ if }x_{i} \geq 0 \\
   {\alpha}*x_{i}\text{ if }x_{i} < 0 \\
   \end{matrix}} 

Where:

 -  :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input data subset*

 -  :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output data subset*

 -  :math:`\alpha` - coefficient of the negative slope for the specific
    data subset
	
While for Leaky ReLU the whole tensor shares the only :math:`\alpha` coefficient, for PRelu an 
array of slope coefficients is shared across an axis.  In other word, for each slice along the 
specified axis an induvidual :math:`\alpha` slope coefficient is used. 

This kernel outputs tensor of the same shape and type as input. This kernel can perform in-place 
computation: output and input can point to exactly the same memory (the same starting address).

Kernels which implement Leaky ReLU functions have the following prototype:

.. code::

   mli_status mli_krn_leaky_relu_<data_format>(
      const mli_tensor \*in,
      const mli_tensor \*slope_coeffs,
      const mli_prelu_cfg \*cfg,
      mli_tensor \*out);

where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function parameters 
are shown in the following table:

.. table:: Data Format Naming Convention Fields
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

.. code::

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
   | ``axis``        | ``int32_t``    | For instance, having future map in HWC layout, axis == 0     |
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

All the listed functions must comply to the following conditions:

 - ``in`` and ``slope_coeff`` tensors must be valid.
 
 - ``mem_stride`` of the innermost dimension should be equal to 1 for all the tensors.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity 
   (that is, the total amount of elements in input tensor). Other fields are filled by 
   kernel (shape, rank and element specific parameters).
   
For **sa8** versions of kernel, in addition to the preceding conditions: 

 - ``in`` and ``slope_coeff`` tensors must be quantized on the tensor level. It implies 
   that the tensor contains a single scale factor and a single zero offset.
   
Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and return the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

.. _sigmoid_prot:

Sigmoid Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel performs sigmoid (also mentioned as logistic) activation function on input tensor 
lement-wise and stores the result to the output tensor.

.. math:: y_{i} = \frac{1}{1 + e^{{- x}_{i}}}

Where:

-  :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input tensor*

-  :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output tensor*

This kernel outputs tensor of the same shape and type as input. This kernel can perform 
in-place computation: output and input can point to exactly the same memory (the same 
starting address).

Kernels which implement Sigmoid functions have the following prototype:

.. code::

   mli_status mli_krn_sigm_<data_format>(
      const mli_tensor \*in,
      mli_tensor \*out);
..
	  
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function 
parameters are shown in the following table:

.. table:: Data Format Naming Convention Fields
   :align: center
   :widths: auto
   
   +----------------+----------------------+-----------------------------------------+
   | **Parameter**  | **Type**             | **Description**                         |
   +================+======================+=========================================+
   | ``in``         | ``mli_tensor *``     | [IN] Pointer to constant input tensor.  |
   +----------------+----------------------+-----------------------------------------+
   | ``out``        | ``mli_tensor *``     | [OUT] Pointer to output tensor.         |
   |                |                      | Result is stored here                   |
   +----------------+----------------------+-----------------------------------------+
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

All the listed functions must comply to the following conditions:

 - ``in`` tensor must be valid.
 
 - ``mem_stride`` of the innermost dimension should be equal to 1 for all the tensors.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity 
   (that is, the total amount of elements in input tensor). Other fields are filled by 
   kernel (shape, rank and element specific parameters).
   
For **sa8** versions of kernel, in addition to the preceding conditions: 

 - ``in`` tensor must be quantized on the tensor level. It implies that the tensor contains 
   a single scale factor and a single zero offset.
   
Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and return the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

The range of this function is (0, 1), and this kernel outputs completely fractional tensor of 
the same shape and type as input. For fx8 type, the output holds 7 fractional bits, and 15 
fractional bits for fx16 type. Therefore, the maximum representable value of SoftMax is equivalent 
to 0.9921875 for **fx8** output tensor, and to 0.999969482421875 for fx16 (not 1.0).

.. _tanh_prot:

TanH Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel performs hyperbolic tangent activation function on input tensor elementwise 
and stores the result to the output tensor.

.. math:: y_{i} = \frac{e^{x_{i}} - e^{{- x}_{i}}}{e^{x_{i}} + e^{{- x}_{i}}}

Where:

-  :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input tensor*

-  :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output tensor*

This kernel outputs tensor of the same shape and type as input. This kernel performs 
in-place computation: output and input can point to exactly the same memory (the same 
starting address).

Kernels which implement TanH functions have the following prototype:

.. code::

   mli_status mli_krn_tanh_<data_format>(
      const mli_tensor *in,
      mli_tensor *out);
	  
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function 
parameters are shown in the following table:

.. table:: Data Format Naming Convention Fields
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

All the listed functions must comply to the following conditions:

 - ``in`` tensor must be valid.
 
 - ``mem_stride`` of the innermost dimension should be equal to 1 for all the tensors.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity 
   (that is, the total amount of elements in input tensor). Other fields are filled 
   by kernel (shape, rank and element specific parameters).

For **sa8** versions of kernel, in addition to the preceding conditions: 

 - ``in`` tensor must be quantized on the tensor level. It implies that the tensor 
   contains a single scale factor and a single zero offset.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and return the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

The range of function is (-1, 1), and kernel outputs a completely fractional tensor of the 
same shape and type as input. Output holds 7 fractional bits for fx8 type, and 15 fractional 
bits for fx16 type. For this reason, the maximum representable value of TanH is equivalent to 
0.9921875 in case of **fx8** output tensor, and to 0.999969482421875 in case of **fx16** (not 1.0).

.. _softmax_prot:

Softmax Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel performs Softmax activation function that is a generalization of the 
logistic function that transforms input vector according to the following formula:

.. math:: y_{i} = \frac{e^{x_{i}}}{\sum_{j}^{}e^{x_{j}}}

Where:

-  :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input data subset*

-  :math:`x_{j}` *–* :math:`j_{\text{th}}` *value in the same input data
   subset*

-  :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output data subset*
	
Softmax function might be applied to the whole tensor, or along a specific axis. 
In first case all input values are involved in calculation of each output value. 
If axis is specified, then softmax function is applied to each slice along the 
specific axis independently. 

This kernel outputs tensor of the same shape and type as input. This kernel performs
in-place computation: output and input can point to exactly the same memory (the same 
starting address).
 
Kernels which implement SoftMax functions have the following prototype:

.. code::

   mli_status mli_krn_softmax_<data_format>(
      const mli_tensor *in,
      const mli_softmax_cfg *cfg,
     mli_tensor *out);
..
	 
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function 
parameters are shown in the following table:

.. table:: Data Format Naming Convention Fields
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

All the listed functions must comply to the following conditions:

 - ``in`` tensor must be valid.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity 
   (that is, the total amount of elements in input tensor). Other fields are filled 
   by kernel (shape, rank and element specific parameters).
   
 - ``mem_stride`` of the innermost dimension should be equal to 1 for all the tensors.
 
 - axis parameter might be negative and must be less than in tensor rank.
 

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and return the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

.. _l2_norm_prot:

L2 Normalization Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel normalizes data across specified dimension using L2 norm according to the following 
formula:

.. math:: y_{i} = \frac{x_{i}}{\sqrt{MAX(epsilon,\sum_{j}{x_{j}}^{2})}}

Where:

-  :math:`x_{i}-i_{th}` *–* value in input data subset*

-  :math:`x_{j}-j_{th}` *–* value in the same input data subset*

-  :math:`y_{i}-i_{th}` *–* value in output data subset*

-  :math:`epsilon` *–* lower bound to prevent division on zero

L2 normalization function might be applied to the whole tensor, or along a specific axis. In the 
first case all input values are involved in calculation of each output value. If axis is specified, 
then the function is applied to each slice along the specific axis independently. 

This kernel outputs tensor of the same shape and type as input. This kernel performs in-place 
computation: output and input can point to exactly the same memory (the same starting address).

Kernels which implement L2 normalization functions have the following prototype:

.. code::

   mli_status mli_krn_L2_normalize_<data_format>(
      const mli_tensor *in,
      const mli_tensor *epsilon,
      const mli_softmax_cfg *cfg,
      mli_tensor *out);
	  
where data_format is one of the data formats listed in Table :ref:`mli_data_fmts` and the function parameters are 
shown in the following table:

.. table:: Data Format Naming Convention Fields
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

mli_L2_normalize_cfg is defined as:

.. code::

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

All the listed functions must comply to the following conditions:

 - ``in`` and ``epsilon`` tensors must be valid.
 
 - ``epsilon`` tensor must be a valid tensor-scalar (see data field 
   description in the Table :ref:`mli_tnsr_struc`).
   
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient 
   capacity (that is, the total amount of elements in input tensor). Other 
   fields are filled by kernel (shape, rank and element specific parameters).

 - ``mem_stride`` of the innermost dimension should be equal to 1 for all the 
   tensors.

 - ``axis`` parameter might be negative and must be less than in tensor rank.

For **sa8** versions of kernel, in addition to the preceding conditions: 

 - ``in`` and ``epsilon`` tensors must be quantized on the tensor level. It 
   implies that the tensor contains a single scale factor and a single zero offset.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and return the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.


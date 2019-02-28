Transform Group
---------------

   Transform group provides operations for transformation data according
   to particular function (activations). In general, transformations do
   not alter data size or shape. They only change the values
   element-wise.

.. _relu:
   
ReLU
~~~~

.. _f_relu_image:
.. figure:: ../pic/images/image_relu.png

   Various ReLU functions. a. General ReLU; b. ReLU1; c. ReLU6

   This kernel represents Rectified Linear unit (ReLU). It performs
   various types of the rectifier activation on input. The following
   types of ReLU supported by this type of kernel:

1) General ReLU: :math:`f(x) = MAX\left( x,\ \ 0 \right)`

2) ReLU1:
      :math:`f(x) = MAX\left( \text{MIN}\left( x,\ 1 \right),\  - 1 \right)`

3) ReLU6:
      :math:`f(x) = MAX\left( \text{MIN}\left( x,\ 6 \right),\ 0 \right)`

..

   Where:

   *x* - input value.

   Kernel outputs a tensor of the same shape, type and format as input
   tensor.

   Kernel might perform in-place computation: output and input might point
   to exactly the same memory (the same starting address). It might affect
   performance for some platforms.

.. _function-configuration-structure-7:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Definition**        | typedef struct {                              |
|                       | mli_relu_type type;                           |
|                       | } mli_relu_cfg;                               |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Fields**            | ``Type``              | Type of ReLu          |
|                       |                       | that is applied       |
|                       |                       | (enumeration)         |
+-----------------------+-----------------------+-----------------------+

\

.. _mli_relu_val_desc:
.. table:: mli_relu_type Values Description
   :widths: auto   

   +-----------------------------------+-----------------------------------+
   | **Value**                         | **Field Description**             |
   +===================================+===================================+
   | ``MLI_RELU_NONE``                 | No ReLU. Identity function.       |
   +-----------------------------------+-----------------------------------+
   | ``MLI_RELU_GEN``                  | General Rectifier function with   |
   |                                   | output range from 0 to value      |
   |                                   | maximum inclusively.              |
   +-----------------------------------+-----------------------------------+
   | ``MLI_RELU_1``                    | ReLU1 Rectifier function with     |
   |                                   | output range [-1, 1]              |
   +-----------------------------------+-----------------------------------+
   | ``MLI_RELU_6``                    | ReLU6 Rectifier function with     |
   |                                   | output range [0, 6]               |
   +-----------------------------------+-----------------------------------+

.. _api-3:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_relu_<data_type>(          |
|                       | const mli_tensor *in,                         |
|                       | const mli_relu_cfg *cfg,                      |
|                       | mli_tensor *out);                             |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``cfg``               | [IN] Pointer to       |
|                       |                       | function parameters   |
|                       |                       | structure             |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output tensor for     |
|                       |                       | storing the result    |
+-----------------------+-----------------------+-----------------------+

.. _kernel-specializations-3:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+--------------------------------------+
| **Function**          | **Description**                      |
+=======================+======================================+
| ``mli_krn_relu_fx8``  | General function; 8bit FX elements;  |
+-----------------------+--------------------------------------+
| ``mli_krn_relu_fx16`` | General function; 16bit FX elements; |
+-----------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-3:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, tensors must be valid (see :ref:`mli_tns_struct`).

   -  Before processing, the output tensor must contain valid pointer to a
      buffer with sufficient capacity enough for storing the result.
      Other fields are filled by kernel (shape, rank and element
      specific parameters)

Leaky ReLU
~~~~~~~~~~

.. _f_leaky_relu:
.. figure:: ../pic/images/image149.png 

   Leaky ReLU Function With Negative Slope 0.3

   This kernel represents Rectified Linear unit (ReLU) with a negative
   slope. It transforms each element of input tensor according to next
   formula:

   :math:`f(x) = \left\{ \begin{matrix}
   x\ \ if\ x \geq 0 \\
   \alpha*x\ \ if\ x < 0 \\
   \end{matrix} \right.\ `

   Where:

   *x* - input value.

   :math:`\alpha\ ` - Coefficient of the negative slope.

   The function accepts two tensors as input and one as output. The
   first input tensor is the feature map to be processed by the kernel,
   and the second input is a tensor-scalar (see :ref:`mli_tns_struct`)
   that holds a negative slope coefficient.

   Ensure that the scalar tensor holds element of the same type as that
   of input tensor (it does not need to have the same format, that is,
   the number of fractional bits).

   Kernel outputs tensor of the same shape, type and format as input
   tensor.

   Kernel can perform in-place computation: output and input might point
   to exactly the same memory (the same starting address). It might affect
   performance for some platforms.

.. _function-configuration-structure-8:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   No configuration structure for leaky ReLU kernel is required. All
   necessary information is provided by tensors.

.. _api-4:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_leaky_relu_<data_type>(    |
|                       | const mli_tensor *in,                         |
|                       | const mli_tensor *slope_coeff,                |
|                       | mli_tensor *out);                             |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``slope_coeff``       | [IN] Pointer to       |
|                       |                       | tensor-scalar with    |
|                       |                       | negative slope        |
|                       |                       | coefficient           |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output tensor for     |
|                       |                       | storing the result    |
+-----------------------+-----------------------+-----------------------+

.. _kernel-specializations-4:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------+--------------------------------------+
| **Function**                | **Description**                      |
+=============================+======================================+
| ``mli_krn_leaky_relu_fx8``  | General function; 8bit FX elements;  |
+-----------------------------+--------------------------------------+
| ``mli_krn_leaky_relu_fx16`` | General function; 16bit FX elements; |
+-----------------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-4:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, and slope coefficient tensors must be valid (see :ref:`mli_tns_struct`
      ).

   -  Slope coefficient must be valid tensor-scalar (see :ref:`mli_tns_struct`).

   -  Before processing, the output tensor must contain a valid pointer to
      a buffer, with sufficient capacity enough for storing the result.
      Other fields are filled by the kernel (shape, rank and element
      specific parameters)

.. _sigmoid:

Sigmoid
~~~~~~~

.. _f_sigm:
.. figure:: ../pic/images/image152.png 

   Logistic (Sigmoid) function

   This kernel performs sigmoid (also mentioned as logistic) activation
   function on input tensor element-wise and stores the result to the
   output tensor:

.. math::

   f(x) = \frac{1}{1 + e^{- x}}

..

   *x* - input value.
   
..

   The range of function is (0, 1), and kernel outputs completely
   fractional tensor of the same shape and type as input. For ``fx8`` type,
   the output holds 7 fractional bits, and 15 fractional bits for ``fx16``
   type. For this reason, the maximum representable value of SoftMax is
   equivalent to 0.9921875 for ``fx8`` output tensor, and to
   0.999969482421875 for ``fx16`` (not 1.0).

   Kernel can perform in-place computation: output and input can point
   to exactly the same memory (the same starting address).

.. _function-configuration-structure-9:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   No configuration structure for sigmoid kernel is required. All
   necessary information is provided by tensors.

.. _api-5:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_sigm_<data_type>(          |
|                       | const mli_tensor *in,                         |
|                       | mli_tensor *out);                             |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output tensor for     |
|                       |                       | storing the result    |
+-----------------------+-----------------------+-----------------------+

.. _kernel-specializations-5:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+--------------------------------------+
| **Function**          | **Description**                      |
+=======================+======================================+
| ``mli_krn_sigm_fx8``  | General function; 8bit FX elements;  |
+-----------------------+--------------------------------------+
| ``mli_krn_sigm_fx16`` | General function; 16bit FX elements; |
+-----------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-5:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, tensors must be valid (see :ref:`mli_tns_struct`).

   -  Before processing, the output tensor must contain a valid pointer to
      a buffer, with sufficient capacity enough for storing the result.
      Other fields are filled by kernel (shape, rank and element
      specific parameters)

.. _tanh:

TanH
~~~~

.. _f_tanh_func:
.. figure:: ../pic/images/image154.png

   Hyperbolic Tangent (TanH) Function

   This kernel performs hyperbolic tangent activation function on input
   tensor element-wise and store result to the output tensor:

.. math::

   f(x) = \frac{e^{x} - e^{- x}}{e^{x} + e^{- x}}

  *x* - input value.

..

   The range of function is (-1, 1), and kernel outputs a completely
   fractional tensor of the same shape and type as input. Output holds 7
   fractional bits for ``fx8`` type, and 15 fractional bits for ``fx16`` type.
   For this reason, the maximum representable value of TanH is
   equivalent to 0.9921875 in case of ``fx8`` output tensor, and to
   0.999969482421875 in case of ``fx16`` (not 1.0).

   The kernel can perform in-place computation: output and input can
   point to exactly the same memory (the same starting address).

.. _function-configuration-structure-10:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   No configuration structure for tanh kernel is required. All
   necessary information is provided by tensors.

.. _api-6:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_tanh_<data_type>(          |
|                       | const mli_tensor *in,                         |
|                       | mli_tensor *out);                             |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output tensor for     |
|                       |                       | storing the result    |
+-----------------------+-----------------------+-----------------------+

.. _kernel-specializations-6:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+--------------------------------------+
| **Function**          | **Description**                      |
+=======================+======================================+
| ``mli_krn_sigm_fx8``  | General function; 8bit FX elements;  |
+-----------------------+--------------------------------------+
| ``mli_krn_sigm_fx16`` | General function; 16bit FX elements; |
+-----------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-6:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, tensors must be valid (see :ref:`mli_tns_struct`).

   -  Before processing, the output tensor must contain a valid pointer to
      a buffer, with sufficient capacity enough for storing the result.
      Other fields are filled by kernel (shape, rank and element
      specific parameters)

SoftMax
~~~~~~~

.. _f_softmax_func:
.. figure:: ../pic/images/image156.png 

   Softmax Function

   This kernel performs activation function which is a generalization of
   the logistic function that transform input vector according to next
   formula:

.. math::

   y_{i} = \frac{e^{x_{i}}}{\sum_{j}^{}e^{x_{j}}}
   
..

   *x* :sub:`i` - i :sub:`th` value in input tensor.

   :math:`y_{i}\ ` - :math:`i_{\text{th}}` value in output tensor.

..
   
   The SoftMax function is often used as the final layer of a neural
   network-based classifier and its output can be considered as a
   probability distribution over N different possible outcomes. The sum
   of all the entries across the last dimension tends to 1.

   For FX data type, the range of output values is [0, 1), and all
   non-sign bits are fractional. Output holds 7 fractional bits for fx8
   type, and 15 fractional bits for fx16 type. For this reason, the
   maximum representable value of SoftMax is equivalent to 0.9921875 in
   case of fx8 output tensor, and to 0.999969482421875 in case of fx16
   (not 1.0).

   The kernel outputs tensor of the same shape and type as input.

   The kernel can perform in-place computation: output and input can
   point to exactly the same memory (the same starting address).

.. _function-configuration-structure-11:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   No configuration structure for softmax kernel is required. All
   necessary information is provided by tensors.

.. _api-7:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_softmax_<data_type>(       |
|                       | const mli_tensor *in,                         |
|                       | mli_tensor *out);                             |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output tensor. Result |
|                       |                       | is stored here        |
+-----------------------+-----------------------+-----------------------+

.. _kernel-specializations-7:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+--------------------------+--------------------------------------+
| **Function**             | **Description**                      |
+==========================+======================================+
| ``mli_krn_softmax_fx8``  | General function; 8bit FX elements;  |
+--------------------------+--------------------------------------+
| ``mli_krn_softmax_fx16`` | General function; 16bit FX elements; |
+--------------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-7:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, tensors must be valid (see :ref:`mli_tns_struct`).

   -  Before processing, the output tensor must contain a valid pointer to
      a buffer, with sufficient capacity enough for storing the result.
      Other fields are filled by kernel (shape, rank and element
      specific parameters)

